\# Traffic Flow Simulation using MPI + OpenMP



\## Overview



This project implements a hybrid parallel traffic flow simulation based on the Nagel-Schreckenberg traffic model.



The simulation combines:



\- MPI (Message Passing Interface)

\- OpenMP (Shared Memory Parallelism)

\- Non-Blocking MPI Communication



The goal is to evaluate traffic behavior and analyze the performance benefits of hybrid parallel computing.



\---



\## Features



\- Sequential baseline implementation

\- Hybrid MPI + OpenMP implementation

\- Non-blocking MPI communication

\- Performance profiling

\- Speedup and efficiency analysis

\- CSV export of simulation data

\- ASCII traffic visualization



\---



\## Technologies



\- C++

\- MPI

\- OpenMP

\- Visual Studio



\---



\## Simulation Parameters



| Parameter | Value |

|------------|---------|

| Road Length | 1,000,000 |

| Max Speed | 5 |

| Iterations | 1000 |

| Vehicle Density | 0.3 |

| Slowdown Probability | 0.2 |



\---



\## Performance Metrics



The application measures:



\- Sequential Execution Time

\- Parallel Execution Time

\- Speedup

\- Efficiency

\- Communication Overhead

\- Computation Time

\- I/O Time



\---



\## Build \& Run



Compile:



```bash

mpicxx -fopenmp traffic\_simulation.cpp -o traffic\_sim

```



Run:



```bash

mpirun -np 4 ./traffic\_sim

```



\---



\## Output Files



\### traffic\_data.csv



Stores vehicle position and speed information.



\### performance\_results.csv



Stores performance measurements:



\- MPI Processes

\- Sequential Time

\- Parallel Time

\- Speedup

\- Efficiency

\- Compute Time

\- Communication Time

\- I/O Time



\---



\## Parallelization Strategy



\### MPI



The road is divided among MPI processes.



Each process:



\- Handles a local road segment

\- Exchanges boundary information

\- Uses non-blocking communication



\### OpenMP



Each MPI process creates multiple threads to update vehicle states in parallel.



\---



\## Author



Rahma Dawy



AI \& Data Science Student

