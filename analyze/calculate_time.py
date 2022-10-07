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
    tiempos_seq = []
    for i in range(10):
        tiempo = ejecutar('build/TireDetector', args)
        tiempos_omp.append(tiempo)
        tiempo = ejecutar('build/Secuencial', args)
        tiempos_seq.append(tiempo)

    print(tiempos_omp)
    print(tiempos_seq)


if __name__ == '__main__':
    # print(sys.argv[1:])
    main(sys.argv[1:])
