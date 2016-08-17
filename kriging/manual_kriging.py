
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from manual_kriging_data import generate_data


def print_matrix(M):
    print '-' * len(M) * 12     # Linha Horizontal
    for line in M:
        for elem in line:
            sys.stdout.write('{elem: .8f} '.format(elem=elem))
        print
    print '-' * len(M) * 12     # Linha Horizontal


def dist(p, q):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


def dist_matrix(m):
    distances_matrix = []

    for p in m:
        distances = []
        for q in m:
            distances.append(dist(p, q))
        distances_matrix.append(distances)

    return distances_matrix


def mean(x):
    sum = 0.0
    for i in x:
        sum += i
    return sum / len(x)


def cov(x, y):
    mean_of_x = mean(x)
    mean_of_y = mean(y)

    cov_sum = 0.0
    for i in xrange(len(x)):
        cov_sum += (x[i] - mean_of_x) * (y[i] - mean_of_y)

    cov_sum /= len(x) - 1
    return cov_sum


def cov_matrix(data):
    number_of_variables = len(data[0])
    cov_matrix_values = []

    for var_i in xrange(number_of_variables):
        cov_values = []
        var_i_samples = data[:, var_i]

        for var_j in xrange(number_of_variables):
            var_j_samples = data[:, var_j]
            cov_values.append(cov(var_i_samples, var_j_samples))

        cov_matrix_values.append(cov_values)

    return cov_matrix_values


def semi_variogram(data):
    semi_var = {}
    distances_matrix = dist_matrix(data)

    # x = data[:, 0]
    # y = data[:, 1]
    z = data[:, 2]

    for i in xrange(len(distances_matrix)):

        distances = distances_matrix[i]
        i_value = z[i]

        for j in xrange(len(distances)):
            if i != j:
                j_value = z[j]

                dist_i_j = distances[j]

                if dist_i_j not in semi_var:
                    semi_var[dist_i_j] = set()

                semi_var_value = (i_value - j_value)**2
                semi_var[dist_i_j].add(semi_var_value)

    new_semi_var = []
    for h in semi_var.keys():
        semi_var_values = semi_var[h]
        actual_semi_var = 0.5 * mean(semi_var_values)
        new_semi_var.append([h, actual_semi_var])

    return new_semi_var


def semi_variogram_lags(data, lags=10):
    semi_var = {}

    for lag in xrange(1, lags):
        semi_var[lag] = {
            'dist': [],
            'semivar': []
        }

    distances_matrix = dist_matrix(data)
    number_of_points = len(distances_matrix)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    cutoff = math.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2) / 3.0

    for lag in xrange(1, lags):

        for i in xrange(number_of_points):
            for j in xrange(number_of_points):
                dist_i_j = distances_matrix[i][j]

                lag_range = [(lag - 1) * cutoff / lags, lag * cutoff / lags]

                if lag_range[0] < dist_i_j < lag_range[1]:
                    semi_var_value = (z[i] - z[j])**2
                    semi_var[lag]['dist'].append(dist_i_j)
                    semi_var[lag]['semivar'].append(semi_var_value)

    new_semi_var = []
    for lag in semi_var.keys():
        dist_values = semi_var[lag]['dist']
        semi_var_values = semi_var[lag]['semivar']

        if len(dist_values) == 0 or len(semi_var_values) == 0:
            continue

        h = mean(dist_values)
        actual_semi_var = 0.5 * mean(semi_var_values)

        new_semi_var.append([h, actual_semi_var])

    return new_semi_var


def sphermodel(h, nugget, range, sill):
    if h >= range:
        return sill

    return (sill-nugget) * (1.5 * (h / range) - 0.5 * math.pow(h / range, 3)) + nugget


def sphere_covar(dist, nugget, sill, range):
    n = len(dist)

    sigma = sill + nugget
    covar = [[0 for _ in xrange(n)] for _ in xrange(n)]

    for i in xrange(n):
        for j in xrange(n):
            dist_i_j = dist[i][j]

            if dist_i_j > range:
                covar[i][j] = 0.0
            elif dist_i_j == 0.0:
                covar[i][j] = sigma
            else:
                covar[i][j] = sphermodel(dist_i_j, nugget, range, sill)


# spher.covar <- function(dist, nugget, sill, range) {
#     sigma <- sill + nugget
#     # Covariance(h) = Sigma + Gamma(h)
#     covar <- matrix(0, nrow=nrow(dist), ncol=ncol(dist))
#     for (i in 1:nrow(covar)) {
#         for (j in 1:ncol(covar)) {
#             if (dist[i,j] > range)
#             covar[i,j] <- 0
#         else if (dist[i,j] == 0)
#             covar[i,j] <- sigma
#             else covar[i,j] <- sigma - (nugget + (sigma - nugget) *
#                         (((3*dist[i,j])/(2*range)) - ((dist[i,j]^3)/(2*range^3))))
#         }
#     }
#     return(covar)
# }


def main():
    raw_data = generate_data(1)

    print np.cov(raw_data, rowvar=0)

    # print_matrix(cov_matrix(raw_data))
    # print_matrix(dist_matrix(raw_data))

    # semi_var = np.array(semi_variogram(raw_data))
    semi_var = np.array(semi_variogram_lags(raw_data, lags=10))
    print semi_var

    h_values = semi_var[:, 0]
    semi_var_values = semi_var[:, 1]

    print h_values
    print semi_var_values

    # model_x = np.arange(0, h_values.max(), 0.1)
    # nugget = 1.242169
    # sill = 1.217841
    # range = 18.95621
    # model_y = [sphermodel(x, nugget, range, sill) for x in model_x]
    # plt.plot(model_x, model_y, 'b-')

    plt.plot(h_values, semi_var_values, 'ro')
    plt.axis([0.0, h_values.max(), 0, semi_var_values.max()])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()