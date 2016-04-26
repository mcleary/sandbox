import sys
import numpy as np


def print_matrix(M):
    print '-' * len(M) * 12     # Linha Horizontal
    for line in M:
        for elem in line:
            sys.stdout.write('{elem: .8f} '.format(elem=elem))
        print
    print '-' * len(M) * 12     # Linha Horizontal


def mat_mat_mul(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]


def mat_vec_mul(A, v):
    n = len(A)
    Av = [0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Av[i] += A[i][j] * v[i]
    return Av;


def hilbert(n):
    H = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            H[i][j] = 1.0 / (i+1 + j+1 - 1)
    return H


def upper_triangular_solver(A):
    n = len(A)
    x = [0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x


def gauss_solver(A, b):
    # Numero de linhas de A
    n = len(A)

    A_b = [[0 for _ in range(n+1)] for _ in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            A_b[i][j] = A[i][j]

    for i in range(0, n):
        A_b[i][n] = b[i]

    # procura o elemento maximo da coluna
    for i in range(0, n):
        max_elem = abs(A_b[i][i])
        max_row = i
        for k in range(i+1, n):
            if abs(A_b[k][i] > max_elem):
                max_elem = abs(A_b[k][i])
                max_row = k

        # troca a linha com o valor maximo com a linha corrente
        for k in range(i, n+1):
            A_b[max_row][k], A_b[i][k] = A_b[i][k], A_b[max_row][k]

        # zera as linhas abaixo da coluna atual
        for k in range(i+1, n):
            m = -A_b[k][i] / A_b[i][i]
            for j in range(i, n+1):
                if i == j:
                    A_b[k][j] = 0.0
                else:
                    A_b[k][j] += m * A_b[i][j]

    # resolve o sistema triangular superior Ax=b
    return upper_triangular_solver(A_b)


def jacobi_approximation(A, b, x0):
    n = len(A)

    x_new = [0 for _ in x0]
    x_old = [x for x in x0]

    for iteration in range(100):
        for i in range(n):
            t1, t2 = 0.0, 0.0
            for j in range(0, i):
                t1 += A[i][j] * x_old[j]
            for j in range(i, n):
                t2 += A[i][j] * x_old[j]

            x_new[i] = (b[i] - t1 - t2) / A[i][i]

        x_old = x_new
        print x_new


def main():
    n = 3

    #A = [[0 for _ in range(n)] for _ in range(n)]
    #for i in range(0, n):
    #    A[i][i] = 2.0

    A = hilbert(n)
    x = [1 for _ in range(n)]
    b = mat_vec_mul(A, x)

    x0 = gauss_solver(A, b)
    jacobi_approximation(A, b, x0)

    x0n = np.linalg.solve(A, b)

    print
    print x0
    print x0n

if __name__ == '__main__':
    main()






