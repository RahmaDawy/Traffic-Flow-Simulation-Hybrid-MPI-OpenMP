
# Traffic Flow Simulation — Hybrid MPI + OpenMP

A high-performance parallel traffic flow simulator built in C++ using the **Nagel-Schreckenberg (NaSch) cellular automaton model**. Combines **MPI** for distributed memory parallelism across nodes and **OpenMP** for shared memory parallelism within each node, with **non-blocking MPI communication** to overlap computation and communication.

---

## Model

The Nagel-Schreckenberg model simulates single-lane highway traffic using four rules applied each timestep:

1. **Accelerate** — vehicles below max speed increase by 1
2. **Decelerate** — vehicles slow to avoid collision with the car ahead
3. **Randomize** — vehicles randomly slow down with probability `p` (models human behavior)
4. **Move** — vehicles advance by their updated speed

---

## Parallelization Strategy

### MPI — Distributed Road Segments
The road is partitioned evenly across MPI processes. Each process owns a contiguous segment and exchanges **ghost cells** with its neighbors using `MPI_Isend` / `MPI_Irecv` (non-blocking).

### OpenMP — Thread-Level Vehicle Updates
Within each MPI process, vehicle state updates are parallelized across CPU threads using `#pragma omp parallel for`. Inner cells and boundary cells are handled in separate passes to allow communication overlap.

### Non-Blocking Communication Overlap
```
[Send boundary]  →  [Compute inner cells]  →  [Wait for recv]  →  [Compute boundary cells]
```
Communication and computation run concurrently, minimizing idle time.

---

## Simulation Parameters

| Parameter            | Value       |
|----------------------|-------------|
| Road Length          | 1,000,000   |
| Max Speed            | 5           |
| Iterations           | 1,000       |
| Vehicle Density      | 0.3 (30%)   |
| Slowdown Probability | 0.2 (20%)   |
| Visualization Width  | 50 cells    |

---

## Performance Metrics

The simulation profiles three components separately across all iterations:

| Metric             | Description                                      |
|--------------------|--------------------------------------------------|
| Sequential Time    | Single-threaded baseline (rank 0 only)           |
| Parallel Time      | Wall-clock time for full hybrid execution        |
| Speedup (S)        | `Sequential Time / Parallel Time`                |
| Efficiency (E)     | `Speedup / MPI Processes`                        |
| Compute Time       | Time spent updating vehicle states (OpenMP work) |
| Communication Time | Time spent on MPI boundary exchanges             |
| I/O Time           | Time spent printing and writing CSV              |

---

## Build & Run

### Prerequisites
- MPI implementation (MS-MPI on Windows, OpenMPI/MPICH on Linux)
- C++17 compiler with OpenMP support

### Linux / GCC
```bash
mpicxx -fopenmp -O2 -std=c++17 TrafficFlow.cpp -o traffic_sim
mpirun -np 4 ./traffic_sim
```

### Windows / Visual Studio
Open `TrafficFlow.sln`, set configuration to **Release x64**, and run via the MS-MPI launcher or the built-in MPI project settings.

---

## Output

### Console
- ASCII road visualization every 10 iterations (first 50 cells)
- Final results table with all timing and scalability metrics

### traffic_data.csv
Vehicle positions and speeds sampled from rank 0's segment:
```
Time,Position,Speed
0,3,2
0,7,4
...
```

### performance_results.csv
One row appended per run — useful for scalability experiments across different process counts:
```
MPI_Processes,Sequential_Time,Parallel_Time,Speedup,Efficiency,Compute_Time,Comm_Time,IO_Time
```

---

## Project Structure

```
TrafficFlow/
├── TrafficFlow.cpp        # Main simulation source
├── TrafficFlow.sln        # Visual Studio solution
├── README.md
└── .gitignore

---

## Author

**Rahma Dawy** — AI & Data Science Student
```
