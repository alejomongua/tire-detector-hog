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
    tiempos_omp = []
    for i in range(10):
        tiempo = ejecutar('build/Threaded', args)
        tiempos_omp.append(tiempo)

    # print(tiempos_omp)
    # print(tiempos_seq)

    stdev_omp = statistics.stdev(tiempos_omp)
    mean_omp = statistics.mean(tiempos_omp)

    print(f"Ejecución en paralelo: promedio {round(mean_omp, 2)} s, stdev: {round(stdev_omp, 3)}")


if __name__ == '__main__':
    # print(sys.argv[1:])
    main(sys.argv[1:])
