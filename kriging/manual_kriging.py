
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
from bisect import bisect
from manual_kriging_data import generate_data


def dist(p, q):
    return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)


class DistancesMatrix:
    distances_matrix = []

    def __init__(self, data):
        print 'Creating Distances Matrix ...'

        number_of_points = len(data)

        percentage = 0.0
        for i in xrange(number_of_points):
            distances = []

            current_percentage = (float(i) / float(number_of_points)) * 100.0
            if int(current_percentage) % 5 == 0 and int(current_percentage) != percentage:
                percentage = int(current_percentage)
                print percentage

            for j in xrange(0, i):
                dist_i_j = dist(data[i], data[j])
                distances.append(dist_i_j)

            self.distances_matrix.append(distances)

    def distance(self, i, j):
        if i == j:
            return 0.0

        if j > i:
            # swap(i, j)
            i, j = j, i

        return self.distances_matrix[i][j]


def print_matrix(M):
    print '-' * len(M) * 12     # Linha Horizontal
    for line in M:
        for elem in line:
            sys.stdout.write('{elem: .8f} '.format(elem=elem))
        print
    print '-' * len(M) * 12     # Linha Horizontal


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


def semi_variogram_lags(data, distances_matrix, lag_count=10):
    number_of_points = len(data)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    cutoff = math.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2) / 3.0

    lags = []
    lag_ranges = []
    for lag_index in xrange(1, lag_count + 1):
        lags.append(
            {
                'range': [(lag_index - 1) * cutoff / lag_count, lag_index * cutoff / lag_count],
                'semivardata': {
                    'dist': [],
                    'values': []
                }
            }
        )
        lag_ranges.append((lag_index - 1) * cutoff / lag_count)

    for i in xrange(number_of_points):
        print i

        for j in xrange(i, number_of_points):
            dist_i_j = distances_matrix.distance(i, j)

            lag_index = bisect(lag_ranges, dist_i_j) - 1
            lag = lags[lag_index]
            lag_range = lags[lag_index]['range']

            if lag_range[0] < dist_i_j < lag_range[1]:
                semi_var_value = (z[i] - z[j]) ** 2

                lag['semivardata']['dist'].append(dist_i_j)
                lag['semivardata']['values'].append(semi_var_value)

    new_semi_var = []
    for lag in lags:
        dist_values = lag['semivardata']['dist']
        semi_var_values = lag['semivardata']['values']

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


def expmodel(h, nugget, range, sill, a):
    return (sill-nugget) * (1 - math.exp(-h/(range*a))) + nugget


def gaussmodel(h, nugget, range, sill, a):
    if h > range:
        return nugget

    return (sill-nugget) * (1 - math.exp(-(pow(h, 2))/(pow(range, 2)*a))) * nugget


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


def linear_model_fit(x, y):
    mean_y = mean(y)
    mean_x = mean(x)

    sum1 = 0.0
    sum2 = 0.0
    for i in xrange(len(x)):
        sum1 += (x[i] - mean_x) * (y[i] - mean_y)
        sum2 += (x[i] - mean_x)**2

    b = sum1 / sum2
    a = mean_y - b * mean_x

    return a, b


def krig_fit(data):
    distances_matrix = DistancesMatrix(data)
    semi_var = np.array(semi_variogram_lags(data, distances_matrix=distances_matrix))

    h_values = semi_var[:, 0]
    semi_var_values = semi_var[:, 1]

    lm_a, lm_b = linear_model_fit(h_values, semi_var_values)

    nugget = lm_a
    range = h_values.max()
    sill = nugget + lm_b * range

    n = len(data)
    a = np.ones((n+1, n+1), np.float)

    a[n, n] = 0.0

    print 'Covariance ...'
    percentage = 0.0
    for i in xrange(n):

        current_percentage = (float(i) / float(n)) * 100.0
        if int(current_percentage) % 5 == 0 and int(current_percentage) != percentage:
            percentage = int(current_percentage)
            print percentage

        for j in xrange(n):
            dist_i_j = distances_matrix.distance(i, j)
            a[i, j] = sphermodel(dist_i_j, nugget, range, sill)

    return {
        'A': a,
        'nugget': nugget,
        'range': range,
        'sill': sill
    }


def krig_pred(data, x_pred, y_pred, kriging_model):
    A = kriging_model['A']
    nugget = kriging_model['nugget']
    sill = kriging_model['sill']
    range = kriging_model['range']
    inv_A = np.linalg.inv(A)

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    n = len(z)

    R = [0 for _ in xrange(n + 1)]

    for j in xrange(n):
        xdist = dist([x[j], y[j]], [x_pred, y_pred])
        R[j] = sphermodel(xdist, nugget, range, sill)
    R[n] = 1

    invXR = inv_A.dot(R)
    z = np.append(z, 1)
    pred = z.transpose().dot(invXR)

    return pred


def main():
    raw_data = generate_data(4)

    x = raw_data[:, 0]
    y = raw_data[:, 1]

    ng = 30

    grid_dx = abs(x.max() - x.min()) / ng
    grid_dy = abs(y.max() - y.min()) / ng

    print 'Fitting ...'
    kriging_model = krig_fit(raw_data)

    print 'Predicting ...'
    grid = []
    x_min = x.min()
    y_min = y.min()
    for i in xrange(ng):
        for j in xrange(ng):
            grid_x = x_min + i * grid_dx
            grid_y = y_min + j * grid_dy
            grid_z = krig_pred(raw_data, grid_x, grid_y, kriging_model)
            grid.append([grid_x, grid_y, grid_z])

    grid = np.array(grid)
    grid_x = grid[:, 0]
    grid_y = grid[:, 1]
    grid_z = grid[:, 2]

    print 'Exporting ... '
    with open('/Users/mcleary/Desktop/dtm_manual_kriging.xyz', 'w') as output:
    # with open(r'D:\Dropbox\Doutorado\arvores\dtm_manual_kriging.xyz', 'w') as output:
        for i in xrange(len(grid_x)):
            output.write(str(grid_x[i]))
            output.write(' ')
            output.write(str(grid_y[i]))
            output.write(' ')
            output.write(str(grid_z[i]))
            output.write('\n')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='r')
    # # ax.scatter(x_pred, y_pred, z_pred, c='g')
    # ax.scatter(grid_x, grid_y, grid_z, c='g')
    # plt.show()


if __name__ == '__main__':
    main()