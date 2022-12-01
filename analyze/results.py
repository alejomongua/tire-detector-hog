from matplotlib import pyplot as plt
mpi = [
    [
        9.027,
        9.739,
        9.282,
        8.837,
        9.278,
        8.963,
        9.284,
        9.136,
        9.347,
        9.341,
    ],
    [
        5.349,
        5.556,
        5.683,
        5.240,
        5.451,
        5.737,
        5.521,
        5.806,
        5.340,
        5.283,
    ],
    [
        3.369,
        3.647,
        3.490,
        3.263,
        3.222,
        3.591,
        3.087,
        3.256,
        3.567,
        3.368,
    ],
    [
        8.732,
        6.411,
        6.246,
        6.168,
        6.048,
        6.318,
        6.803,
        6.110,
        6.324,
        6.673,
    ],
    [
        5.562,
        4.598,
        4.495,
        4.594,
        4.353,
        4.201,
        5.060,
        4.345,
        4.546,
        4.195,
    ],
    [
        4.980,
        3.571,
        3.319,
        3.250,
        3.158,
        3.136,
        3.234,
        4.013,
        3.705,
        3.245,
    ],
]

x = [1, 2, 4, 8, 16, 32]
times = [sum(i) / 10.0 for i in mpi]
y = [times[0] / i for i in times]
plt.semilogx(x, y)
plt.title("MPI Speedup")
plt.xlabel("Cores")
plt.ylabel("Seconds")
plt.xticks(x, x)
plt.savefig('mpi.svg')

# CUDA

x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10240]
times = [175.15, 97.1, 51.32, 26.69, 14.37, 8.21, 5.27, 4.06, 3.68, 2.77, 2.3, 2.08, 1.94, 1.87, 1.84]
y = [times[0] / i for i in times]
plt.semilogx(x, y)
plt.title("Cuda Speedup")
plt.xlabel("Cores")
plt.ylabel("Seconds")
plt.xticks(x[1:-1:2], x[1:-1:2])
plt.savefig('cuda.svg')

# OpenMP

x = [1, 2, 4, 8, 16, 32, 64]
times = [6.4, 3.52, 2.04, 1.99, 1.74, 1.78, 1.95]
y = [times[0] / i for i in times]
plt.semilogx(x, y)
plt.title("OpenMP Speedup")
plt.xlabel("Cores")
plt.ylabel("Seconds")
plt.xticks(x, x)
plt.savefig('openmp.svg')
