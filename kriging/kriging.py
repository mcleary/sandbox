import numpy as np
from sklearn import gaussian_process
import math


def sklearn_example():

    def f(x):
        return x * np.sin(x)

    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y = f(X).ravel()
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(X, y)
    y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)

    print y_pred


def filter_points():
    xyz_file = open('/Users/tsabino/devel/mangroves-data/1/Arvore1-3leituras.xyz', 'r')

    print 'Lendo arquivo ...'
    xyz_contents = []
    for xyz_line in xyz_file.readlines():
        xyz_contents.append(xyz_line.strip('\n').strip('\r'))

    xyz_file.close()

    points = []

    print 'Extraindo dados ...'
    for xyz_entry in xyz_contents:
        xyz_data = xyz_entry.split()
        points.append([float(xyz_data[2]), float(xyz_data[3]), float(xyz_data[4])])

    np_points = np.array(points)

    x_min = np_points[:, 0].min()
    x_max = np_points[:, 0].max()
    y_min = np_points[:, 1].min()
    y_max = np_points[:, 1].max()
    z_min = np_points[:, 2].min()
    z_max = np_points[:, 2].max()

    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    grid_res = 0.25
    grid_x_size = x_size / grid_res
    grid_y_size = y_size / grid_res

    X = np_points[:, 0]
    Y = np_points[:, 1]
    Z = np_points[:, 2]

    z_min = Z.min()
    Z = Z - z_min
    np_points[:, 2] = Z

    filtered_points = []

    print 'Coletando pontos minimos...'

    total_blocks = int(math.floor(grid_x_size)) * int(math.floor(grid_y_size))
    current_block = 0

    for x_grid_idx in xrange(0, int(math.floor(grid_x_size))):
        for y_grid_idx in xrange(0, int(math.floor(grid_y_size))):
            grid_block_x_min = x_grid_idx * grid_res + x_min
            grid_block_x_max = grid_block_x_min + grid_res
            grid_block_y_min = y_grid_idx * grid_res + y_min
            grid_block_y_max = grid_block_y_min + grid_res
            grid_block_center = [grid_block_x_min + (grid_block_x_max - grid_block_x_min) / 2.0, grid_block_y_min + (grid_block_y_max - grid_block_y_min) / 2.0, 0.0]

            print current_block, 'de', total_blocks, 'blocos'
            current_block += 1

            grid_block_points = []

            for point in np_points:
                if (point[0] >= grid_block_x_min) and (point[0] < grid_block_x_max) and (point[1] >= grid_block_y_min) and (point[1] < grid_block_y_max):
                    grid_block_points.append(point)

            if len(grid_block_points) > 0:
                np_grid_block = np.array(grid_block_points)
                grid_block_min_z_index = np_grid_block[:, 2].argmin()
                grid_block_min = np_grid_block[grid_block_min_z_index]

                if grid_block_min[2] < 1.0:
                    # se estiver abaixo de 1m, considera como um ponto de chao
                    filtered_points.append(grid_block_min)
                else:
                    filtered_points.append(grid_block_center)
            else:
                filtered_points.append(grid_block_center)


    print 'Escrevendo arquivo de pontos filtrados..'
    filtered_points_file = open('/Users/tsabino/Desktop/teste.xyz', 'w')
    for p in filtered_points:
        filtered_points_file.write(str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n')
    filtered_points_file.close()


def kriging():
    points_file = open('/Users/tsabino/Desktop/teste.xyz', 'r')
    points = []
    for point_entry in points_file.readli
        points.append([float(point_data[0]), float(point_data[1]), float(point_data[2])])
    points_file.close()nes():
        point_data = point_entry.strip('\n').split()
    np_points = np.array(points)

    X = np_points[:, [0, 1]]
    y = np_points[:, 2]

    print 'Fitting data...'
    gp = gaussian_process.GaussianProcess(theta0=2e-1, regr='quadratic', )
    gp.fit(X, y)

    x_min = np_points[:, 0].min()
    x_max = np_points[:, 0].max()
    y_min = np_points[:, 1].min()
    y_max = np_points[:, 1].max()
    z_min = np_points[:, 2].min()
    z_max = np_points[:, 2].max()

    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    points_to_predict = []
    for x in np.linspace(x_min, x_max, 100):
        for y in np.linspace(y_min, y_max, 100):
            points_to_predict.append([x, y])

    Z_predicted = gp.predict(points_to_predict)

    dtm_file = open('/Users/tsabino/Desktop/dtm.xyz', 'w')
    for i in xrange(len(points_to_predict)):
        dtm_file.write(str(points_to_predict[i][0]) + ' ' + str(points_to_predict[i][1]) + ' ' + str(Z_predicted[i]) + '\n')
    dtm_file.close()



if __name__ == '__main__':
    #filter_points()
    kriging()