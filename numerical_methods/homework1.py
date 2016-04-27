import sys
import numpy as np


def print_matrix(M):
    print '-' * len(M) * 12     # Linha Horizontal
    for line in M:
        for elem in line:
            sys.stdout.write('{elem: .8f} '.format(elem=elem))
        print
    print '-' * len(M) * 12     # Linha Horizontal


def mat_mul(A, B):
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
    return Av


def vec_minus(x, y):
    n = len(x)
    dx = [0 for _ in range(n)]
    for i in range(n):
        dx[i] = x[i] - y[i]
    return dx


def inf_norm(x):
    max_value = abs(x[0])
    for xi in x:
        if abs(xi) > max_value:
            max_value = abs(xi)
    return max_value


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
            if abs(A_b[k][i]) > max_elem:
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

    x_new = [x for x in x0]
    x_old = [x for x in x0]

    Ax = mat_vec_mul(A, x0)
    r = vec_minus(b, Ax)

    nr = 0

    while abs(inf_norm(r)) > 10E-2 and nr < 1000:
        for i in range(n):
            t1, t2 = 0.0, 0.0
            for j in range(0, i):
                t1 += A[i][j] * x_old[j]
            for j in range(i+1, n):
                t2 += A[i][j] * x_old[j]

            x_new[i] = (b[i] - t1 - t2) / A[i][i]

        Ax = mat_vec_mul(A, x_new)
        r = vec_minus(b, Ax)

        x_old = x_new
        nr += 1

    return x_new, nr


def main():
    for n in range(3, 20):
        #A = [[0 for _ in range(n)] for _ in range(n)]
        #for i in range(0, n):
        #    A[i][i] = 2.0

        H = hilbert(n)
        x = [1 for _ in range(n)]
        b = mat_vec_mul(H, x)

        x0 = gauss_solver(H, b)
        x0p, n_iter = jacobi_approximation(H, b, x0)

        dx = inf_norm(vec_minus(x0p, x))

        print "n: {n: <4} dx: {dx: .20f}   nr: {nr: < 4}   cond(H): {cond_H: <10}".format(n=n, dx=dx, nr=n_iter, cond_H='---')


if __name__ == '__main__':
    main()






