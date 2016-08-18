
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    semi_var = np.array(semi_variogram_lags(data, lags=50))

    h_values = semi_var[:, 0]
    semi_var_values = semi_var[:, 1]

    lm_a, lm_b = linear_model_fit(h_values, semi_var_values)

    nugget = lm_a
    range = h_values.max()
    sill = nugget + lm_b * range

    distances = np.array(dist_matrix(data))

    n = len(data)
    a = np.ones((n+1, n+1), np.float)

    a[n, n] = 0.0

    for i in xrange(n):
        for j in xrange(n):
            dist_i_j = distances[i, j]
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

    R = [0 for _ in xrange(n+1)]

    for j in xrange(n):
        xdist = dist([x[j], y[j]], [x_pred, y_pred])
        R[j] = sphermodel(xdist, nugget, range, sill)
    R[n] = 1

    invXR = inv_A.dot(R)
    z = np.append(z, 1)
    pred = z.transpose().dot(invXR)

    return pred


def main():
    raw_data = generate_data(3)

    x = raw_data[:, 0]
    y = raw_data[:, 1]
    z = raw_data[:, 2]

    ng = 30

    grid = []

    grid_dx = abs(x.max() - x.min()) / ng
    grid_dy = abs(y.max() - y.min()) / ng

    kriging_model = krig_fit(raw_data)

    for i in xrange(ng):
        for j in xrange(ng):
            grid_x = x.min() + i * grid_dx
            grid_y = y.min() + j * grid_dy
            grid_z = krig_pred(raw_data, grid_x, grid_y, kriging_model)
            grid.append([grid_x, grid_y, grid_z])

    grid = np.array(grid)
    grid_x = grid[:, 0]
    grid_y = grid[:, 1]
    grid_z = grid[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r')
    # ax.scatter(x_pred, y_pred, z_pred, c='g')
    ax.scatter(grid_x, grid_y, grid_z, c='g')
    plt.show()

    return

    # print_matrix(cov_matrix(raw_data))
    # print_matrix(dist_matrix(raw_data))

    # semi_var = np.array(semi_variogram(raw_data))
    semi_var = np.array(semi_variogram_lags(raw_data, lags=10))

    h_values = semi_var[:, 0]
    semi_var_values = semi_var[:, 1]

    lm_a, lm_b = linear_model_fit(h_values, semi_var_values)

    my_nugget = lm_a
    my_range = h_values.max()
    my_sill = my_nugget + lm_b * my_range

    print my_nugget, my_range, my_sill

    print h_values
    print semi_var_values

    model_x = np.arange(0, h_values.max(), 0.1)
    # nugget = 1.242169
    # sill = 1.217841
    # range = 18.95621
    nugget = my_nugget
    sill = my_sill
    range = my_range
    a = 1.0 / 3.0
    model_y = [sphermodel(x, nugget, range, sill) for x in model_x]
    model_y1 = [gaussmodel(x, nugget, range, sill, a) for x in model_x]
    model_y2 = [expmodel(x, nugget, range, sill, a) for x in model_x]
    model_y3 = [-0.001283 * x + 1.242169 for x in model_x]
    plt.plot(model_x, model_y, 'b-')
    # plt.plot(model_x, model_y1, 'g-')
    # plt.plot(model_x, model_y2, 'g-')
    plt.plot(model_x, model_y3, 'r-')

    plt.plot(h_values, semi_var_values, 'ro')
    plt.axis([0.0, h_values.max(), 0, semi_var_values.max()])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()