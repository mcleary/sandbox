#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

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
    return C


def mat_vec_mul(A, v):
    n = len(A)
    Av = [0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            Av[i] += A[i][j] * v[i]
    return Av


def vec_plus(x, y):
    return [x[i] + y[i] for i in range(len(x))]


def vec_minus(x, y):
    return [x[i] - y[i] for i in range(len(x))]


def inf_norm_vector(x):
    """
    Norma infinito de um vetor
    :param x:
    :return:
    """
    max_value = abs(x[0])
    for xi in x:
        if abs(xi) > max_value:
            max_value = abs(xi)
    return max_value


def inf_norm_matrix(X):
    """
    Norma infinito de uma matriz
    :param X:
    :return:
    """
    n = len(X)
    max_column_sum = 0.0
    for i in range(n):
        column_sum = 0.0
        for j in range(n):
            column_sum += X[i][j]
        if abs(column_sum) > max_column_sum:
            max_column_sum = abs(column_sum)
    return max_column_sum


def identity_matrix(n):
    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I


def hilbert_matrix(n):
    H = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            H[i][j] = 1.0 / (i+1 + j+1 - 1)
    return H


def check_for_all_zeros(X, i, j):
    """
    Verifica se todos os elementos abaixo da linha i coluna j são zeros
        zero_sum - Quantos
        first_non_zero - index of the first non value
    :param X:
    :param i: Linha
    :param j: Coluna
    :return: O número de zeros encontrados e a primeira linha que contém um elemento diferente de zero
    """
    non_zeros = []
    first_non_zero = -1
    for m in range(i, len(X)):
        non_zero = X[m][j] != 0
        non_zeros.append(non_zero)
        if first_non_zero == -1 and non_zero:
            first_non_zero = m
    zeros_count = sum(non_zeros)
    return zeros_count, first_non_zero


def invert_matrix(X):
    """
    Inverte uma matriz usando a eliminação de gauss-jordan
    A matriz é colocada na forma de echelon

    A matriz identidade é concatenada na matriz que se deseja inverter e ao final
    do processo ela será a inversa

    :param X: Matriz a ser invertida
    :return: A inversa da matriz X
    """
    n = len(X)

    # Faz uma cópia da matriz para evitar modificar o original
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = X[i][j]

    # Concatena a matriz inversa na matriz original para fazer a eliminação de gauss-jordan
    I = identity_matrix(n)
    for i in range(n):
        A[i] += I[i]

    i = 0
    for j in range(n):
        # Verifica se existe algum valor não nulo abaixo da linha e coluna correntes
        zero_sum, first_non_zero = check_for_all_zeros(A, i, j)

        # Se só tem zeros, o algoritmo para aqui
        if zero_sum == 0:
            if j == n:
                return A
            print "Matrix é singular"
            return A

        # Se o pivot é zero mas tem um valor não nulo, troca as linhas
        if first_non_zero != i:
            A[first_non_zero], A[i] = A[i], A[first_non_zero]

        # Divide A[i] por A[i][j] para fazer X[i][j] igual a 1
        A[i] = [m / A[i][j] for m in A[i]]

        # Reescala as outras linhas para fazer os valores abaixo de A[i][j] nulos
        for q in range(n):
            if q != i:
                scaled_row = [A[q][j] * m for m in A[i]]
                A[q] = [A[q][m] - scaled_row[m] for m in range(0, len(scaled_row))]

        if i == n or j == n:
            break
        i += 1

    # Retorna o lado direito da matriz montada no incio do algoritmo
    for i in range(n):
        A[i] = A[i][n:len(A[i])]

    return A


def transpose_matrix(A):
    n = len(A)
    T = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            T[i][j] = A[j][i]
    return T


def upper_triangular_solver(U, b):
    n = len(U)
    x = [0 for _ in range(n)]

    x[n - 1] = b[n - 1] / U[n - 1][n - 1]
    for i in range(n-1, -1, -1):
        tmp = 0.0
        for j in range(i+1, n):
            tmp += U[i][j] * x[j]
        x[i] = (b[i] - tmp) / U[i][i]

    return x


def lower_triangular_solver(L, b):
    n = len(L)
    x = [0 for _ in range(n)]

    x[0] = b[0] / L[0][0]
    for i in range(1, n):
        tmp = 0.0
        for j in range(0, i):
            tmp += L[i][j] * x[j]
        x[i] = (b[i] - tmp) / L[i][i]

    return x


def ldlt_solver(A, b):
    n = len(A)
    L = identity_matrix(n)
    D = [0 for _ in range(n)] # Matriz D na forma de um vetor

    # Decomposição LDLT
    for j in range(n):
        tmp = 0.0
        for k in range(j):
            tmp += L[j][k]**2 * D[k]
        D[j] = A[j][j] - tmp

        for i in range(j+1, n):
            tmp = 0.0
            for k in range(0, j):
                tmp += L[i][k] * D[k] * L[j][k]
            L[i][j] = (A[i][j] - tmp) / D[j]

    # Solução do sistema linear LDL(T) x = b
    w = lower_triangular_solver(L, b)
    y = [w[i] / D[i] for i in range(n)]
    LT = transpose_matrix(L)
    return upper_triangular_solver(LT, y)


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
    new_A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            new_A[i][j] = A_b[i][j]
    new_b = [0 for _ in range(n)]
    for i in range(n):
        new_b[i] = A_b[n-1][i]
    return upper_triangular_solver(new_A, new_b)


def heder_iterative_method(H, x, b):
    """

    :param H:
    :param x: Solução Inicial
    :param b:
    :return:
    """
    # r = b - H*x
    H_x = mat_vec_mul(H, x)
    r = vec_minus(b, H_x)

    x_star = [xi for xi in x]

    iter_count = 0

    while inf_norm_vector(r) > 1E-5 and iter_count <= 50:
        z = gauss_solver(H, r)
        x_star = vec_plus(x_star, z)
        A_x_star = mat_vec_mul(H, x_star)
        r = vec_minus(b, A_x_star)

        iter_count += 1

    return x_star, iter_count


def main():
    print_latex_table = False

    # Resolve o sistema linear
    for n in range(2, 13):
        H = hilbert_matrix(n)
        x = [1000 for _ in range(n)]
        b = mat_vec_mul(H, x)

        x0 = ldlt_solver(H, b)
        x_iter, n_iter = heder_iterative_method(H, x0, b)

        dx = inf_norm_vector(vec_minus(x_iter, x))

        x_iter_norm = inf_norm_vector(x_iter)

        if not print_latex_table:
            print "n: {n: <4} dx: {dx: .20f}   dx_iter: {x_iter: .20f}  nr: {nr: < 4}".format(n=n,
                                                                                              dx=dx,
                                                                                              x_iter=x_iter_norm,
                                                                                              nr=n_iter)
        else:
            print "{n: <4} & {dx: .20f} & {x_iter: .20f} & {nr: < 4} \\\\".format(n=n,
                                                                             dx=dx,
                                                                             x_iter=x_iter_norm,
                                                                             nr=n_iter)

    print
    print "------------------------"
    print

    # Calcula o número de condicionamento da matriz de Hilbert
    for n in range(2, 21):
        H = hilbert_matrix(n)
        H_inv = invert_matrix(H)

        cond_H = inf_norm_matrix(H) * inf_norm_matrix(H_inv)

        if not print_latex_table:
            print "n: {n: <4} cond(H): {cond_H: .20f}".format(n=n, cond_H=cond_H)
        else:
            print '{n: <4} & {cond_H: .20f} \\\\'.format(n=n, cond_H=cond_H)


if __name__ == '__main__':
    main()






