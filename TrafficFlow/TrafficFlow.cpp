#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>

// --- Configuration ---
const int ROAD_LENGTH = 1000000;
const int MAX_SPEED = 5;
const int ITERATIONS = 1000;
const double DENSITY = 0.3;
const double PROB_SLOW = 0.2;
const int VISUALIZE_CELLS = 50; // Number of cells to print and save

// --- Helper: Initialize the road ---
void initializeRoad(std::vector<int>& road) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < ROAD_LENGTH; ++i) {
        road[i] = (dist(rng) < DENSITY) ? (rand() % (MAX_SPEED + 1)) : -1;
    }
}

// --- Helper: ASCII Visualization ---
void printRoad(const std::vector<int>& road, int iter) {
    std::cout << "Iter " << std::setw(4) << iter << " | ";
    for (int i = 0; i < VISUALIZE_CELLS; ++i) {
        if (road[i] == -1) std::cout << "[ . ]";
        else std::cout << "[ " << road[i] << " ]";
    }
    std::cout << "\n";
}

// --- SEQUENTIAL IMPLEMENTATION ---
void runSequential(std::vector<int> road) {
    std::vector<int> next_road(ROAD_LENGTH, -1);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int t = 0; t < ITERATIONS; ++t) {
        std::fill(next_road.begin(), next_road.end(), -1);
        for (int i = 0; i < ROAD_LENGTH; ++i) {
            if (road[i] != -1) {
                int v = road[i];
                int d = 1;
                while (road[(i + d) % ROAD_LENGTH] == -1 && d <= MAX_SPEED) d++;

                if (v < MAX_SPEED) v++;
                if (v >= d) v = d - 1;
                if (v > 0 && dist(rng) < PROB_SLOW) v--;

                next_road[(i + v) % ROAD_LENGTH] = v;
            }
        }
        road = next_road;
    }
}

// --- PARALLEL HYBRID IMPLEMENTATION (Non-Blocking MPI + OpenMP) ---
void runParallel(std::vector<int>& local_road, int rank, int size, double& total_compute_time, double& total_comm_time, double& total_io_time) {
    int local_length = local_road.size();
    std::vector<int> local_next(local_length, -1);
    int left_neighbor = (rank - 1 + size) % size;
    int right_neighbor = (rank + 1) % size;

    std::ofstream csv_file;
    if (rank == 0) {
        csv_file.open("traffic_data.csv");
        csv_file << "Time,Position,Speed\n";
    }

    double step_start_time = 0.0;

    for (int t = 0; t < ITERATIONS; ++t) {
        std::fill(local_next.begin(), local_next.end(), -1);

        int left_boundary_send[MAX_SPEED];
        int right_ghost_recv[MAX_SPEED];

        // 1. MEASURE COMMUNICATION (Setup & Async Send/Recv)
        step_start_time = MPI_Wtime();
        for (int i = 0; i < MAX_SPEED; i++) {
            left_boundary_send[i] = local_road[i];
        }
        MPI_Request reqs[2];
        MPI_Isend(left_boundary_send, MAX_SPEED, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(right_ghost_recv, MAX_SPEED, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, &reqs[1]);
        total_comm_time += (MPI_Wtime() - step_start_time);


        // 2. MEASURE COMPUTATION (Inner Cells via OpenMP)
        step_start_time = MPI_Wtime();
#pragma omp parallel 
        {
            std::mt19937 local_rng(rank * 1000 + omp_get_thread_num() + t);
            std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for
            for (int i = 0; i < local_length - MAX_SPEED; ++i) {
                if (local_road[i] != -1) {
                    int v = local_road[i];
                    int d = 1;
                    while (local_road[i + d] == -1 && d <= MAX_SPEED) d++;

                    if (v < MAX_SPEED) v++;
                    if (v >= d) v = d - 1;
                    if (v > 0 && dist(local_rng) < PROB_SLOW) v--;

                    int new_pos = i + v;
                    if (new_pos < local_length) local_next[new_pos] = v;
                }
            }
        }
        total_compute_time += (MPI_Wtime() - step_start_time);


        // 3. MEASURE COMMUNICATION (Wait for boundary data)
        step_start_time = MPI_Wtime();
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
        total_comm_time += (MPI_Wtime() - step_start_time);


        // 4. MEASURE COMPUTATION (Boundary Cells via OpenMP)
        step_start_time = MPI_Wtime();
#pragma omp parallel 
        {
            std::mt19937 local_rng(rank * 9999 + omp_get_thread_num() + t);
            std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for
            for (int i = local_length - MAX_SPEED; i < local_length; ++i) {
                if (local_road[i] != -1) {
                    int v = local_road[i];
                    int d = 1;

                    while (d <= MAX_SPEED) {
                        int check_pos = i + d;
                        int cell_val = (check_pos < local_length) ? local_road[check_pos] : right_ghost_recv[check_pos - local_length];
                        if (cell_val != -1) break;
                        d++;
                    }

                    if (v < MAX_SPEED) v++;
                    if (v >= d) v = d - 1;
                    if (v > 0 && dist(local_rng) < PROB_SLOW) v--;

                    int new_pos = i + v;
                    if (new_pos < local_length) local_next[new_pos] = v;
                }
            }
        }
        local_road = local_next;
        total_compute_time += (MPI_Wtime() - step_start_time);


        // 5. MEASURE I/O (Printing & CSV Export)
        step_start_time = MPI_Wtime();
        if (rank == 0) {
            if (t % 10 == 0) printRoad(local_road, t);
            for (int i = 0; i < VISUALIZE_CELLS; ++i) {
                if (local_road[i] != -1) {
                    csv_file << t << "," << i << "," << local_road[i] << "\n";
                }
            }
        }
        total_io_time += (MPI_Wtime() - step_start_time);
    }

    if (rank == 0) csv_file.close();
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> global_road;
    double start_time, seq_time = 0.0, par_time = 0.0;

    // Bottleneck trackers
    double total_compute_time = 0.0;
    double total_comm_time = 0.0;
    double total_io_time = 0.0;

    if (rank == 0) {
        std::cout << "======================================\n";
        std::cout << "Traffic Flow Simulation (Non-Blocking Hybrid Model)\n";
        std::cout << "Road Length: " << ROAD_LENGTH << " | Iterations: " << ITERATIONS << "\n";
        std::cout << "MPI Nodes: " << size << " | Max OpenMP Threads/Node: " << omp_get_max_threads() << "\n";
        std::cout << "======================================\n";

        global_road.resize(ROAD_LENGTH);
        initializeRoad(global_road);

        std::cout << "[1/2] Running Sequential Baseline...\n";
        start_time = MPI_Wtime();
        runSequential(global_road);
        seq_time = MPI_Wtime() - start_time;
        std::cout << "      Sequential Time: " << std::fixed << std::setprecision(4) << seq_time << " s\n\n";
    }

    int local_length = ROAD_LENGTH / size;
    std::vector<int> local_road(local_length);
    MPI_Scatter(global_road.data(), local_length, MPI_INT, local_road.data(), local_length, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "[2/2] Running Hybrid Parallel Implementation...\n";

    start_time = MPI_Wtime();

    // Run the parallel simulation and pass the bottleneck variables
    runParallel(local_road, rank, size, total_compute_time, total_comm_time, total_io_time);

    MPI_Barrier(MPI_COMM_WORLD);
    par_time = MPI_Wtime() - start_time;

    if (rank == 0) {
        double speedup = seq_time / par_time;
        double efficiency = speedup / size;
        
        // Calculate percentages
        double measured_total = total_compute_time + total_comm_time + total_io_time;
        double comp_percent = (total_compute_time / measured_total) * 100.0;
        double comm_percent = (total_comm_time / measured_total) * 100.0;
        double io_percent   = (total_io_time / measured_total) * 100.0;

        std::cout << "\n";
        std::cout << "==============================================================\n";
        std::cout << "                 FINAL SIMULATION RESULTS                     \n";
        std::cout << "==============================================================\n";
        std::cout << std::left << std::setw(32) << " Hardware Configuration:" << size << " MPI Node(s)\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << " [1] EXECUTION TIMES\n";
        std::cout << std::left << std::setw(32) << "     Sequential Baseline:" << std::fixed << std::setprecision(4) << seq_time << " seconds\n";
        std::cout << std::left << std::setw(32) << "     Hybrid Parallel Time:" << par_time << " seconds\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << " [2] SCALABILITY METRICS\n";
        std::cout << std::left << std::setw(32) << "     Speedup (S):" << speedup << "x\n";
        std::cout << std::left << std::setw(32) << "     Efficiency (E):" << (efficiency * 100.0) << "%\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << " [3] BOTTLENECK PROFILE\n";
        std::cout << std::left << std::setw(32) << "     Compute Time (Work):" << total_compute_time << " s  (" << comp_percent << "%)\n";
        std::cout << std::left << std::setw(32) << "     Comm Time (Overhead):" << total_comm_time << " s  (" << comm_percent << "%)\n";
        std::cout << std::left << std::setw(32) << "     I/O Time (Print/Save):" << total_io_time << " s  (" << io_percent << "%)\n";
        std::cout << "==============================================================\n";
    }

    MPI_Finalize();
    return 0;
}
