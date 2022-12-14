import statistics
import subprocess
import sys
import time


def ejecutar(programa, args):
    """
    Ejecuta el programa indicado y retorna el tiempo que tardó
    """

    start_time = time.time()

    exit_code = subprocess.call([programa] + args)

    if exit_code:
        raise "Hubo un error ejecutando el programa"

    return time.time() - start_time


def main(args):
    tiempos_cuda = []
    tiempos_omp = []
    tiempos_seq = []
    for i in range(10):
        tiempo = ejecutar('build/TireDetector', args)
        tiempos_cuda.append(tiempo)
        tiempo = ejecutar('build/Threaded', args)
        tiempos_omp.append(tiempo)
        tiempo = ejecutar('build/Secuencial', args)
        tiempos_seq.append(tiempo)

    # print(tiempos_omp)
    # print(tiempos_seq)

    stdev_cuda = statistics.stdev(tiempos_cuda)
    stdev_omp = statistics.stdev(tiempos_omp)
    stdev_seq = statistics.stdev(tiempos_seq)
    mean_cuda = statistics.mean(tiempos_cuda)
    mean_omp = statistics.mean(tiempos_omp)
    mean_seq = statistics.mean(tiempos_seq)

    print(f"Ejecución secuencial: promedio {round(mean_seq, 2)} s, stdev: {round(stdev_seq, 3)}")
    print(f"Ejecución en paralelo: promedio {round(mean_omp, 2)} s, stdev: {round(stdev_omp, 3)}")
    print(f"Ejecución en GPU: promedio {round(mean_cuda, 2)} s, stdev: {round(stdev_cuda, 3)}")


if __name__ == '__main__':
    # print(sys.argv[1:])
    main(sys.argv[1:])
