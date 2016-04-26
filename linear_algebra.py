import sys


def print_matrix(M):
    print '-' * len(M) * 12     # Linha Horizontal
    for line in M:
        for elem in line:
            sys.stdout.write('{elem: .8f} '.format(elem=elem))
        print
    print '-' * len(M) * 12     # Linha Horizontal


def hilbert(n):
    H = [[0 for x in range(n)] for y in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            H[i][j] = 1.0 / (i+1 + j+1 - 1)
    return H


def upper_triangular_solver(A):
    n = len(A)
    x = [0 for x in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x


def gauss_solver(A, b):
    # Numero de linhas de A
    n = len(A)

    A_b = [[0 for x in range(n+1)] for y in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            A_b[i][j] = A[i][j]

    for i in range(0, n):
        A_b[i][n] = b[i]

    # procura o elemento maximo da coluna
    for i in range(0, n):
        max_elem = abs(A_b[i][i])
        max_row = i
        for k in range(0, n):
            if abs(A_b[k][j] > max_elem):
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