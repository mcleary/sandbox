import numpy as np
from sklearn import gaussian_process
import math
import time


class GridBlock:
    def __init__(self, x_min, x_max, y_min, y_max):
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max

        x_size = x_max - x_min
        y_size = y_max - y_min

        self.center = [self._x_min + x_size / 2.0, self._y_min + y_size / 2.0]
        self.points = []

    def contains(self, x, y):
        return (x >= self._x_min) and (x < self._x_max) and (y >= self._y_min) and (y < self._y_max)

    def add_point(self, x, y, z):
        if self.contains(x, y):
            self.points.append([x, y, z])

    def add_point_no_check(self, x, y, z):
        self.points.append([x, y, z])


class Grid2D:
    resolution = 0.25

    def __init__(self, x_min, x_max, y_min, y_max):
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._x_size = int(math.floor((x_max - x_min) / self.resolution))
        self._y_size = int(math.floor((y_max - y_min) / self.resolution))
        self._total_blocks = self._x_size * self._y_size

        self.grid_blocks = []

        for x_idx in xrange(self._x_size):
            for y_idx in xrange(self._y_size):
                block_x_min = x_idx * self.resolution + x_min
                block_x_max = block_x_min + self.resolution
                block_y_min = y_idx * self.resolution + y_min
                block_y_max = block_y_min + self.resolution

                self.grid_blocks.append(GridBlock(block_x_min, block_x_max, block_y_min, block_y_max))

    def find_block(self, x, y):
        start_block = 0
        end_block = len(self.grid_blocks) - 1

    def add_points(self, points):
        total_points = len(points)
        current_progress = 0
        current_point = 0

        for point in points:
            for block in self.grid_blocks:
                x = point[0]
                y = point[1]
                z = point[2]
                if block.contains(x, y):
                    block.add_point_no_check(x, y, z)

            progress = int((current_point / float(total_points)) * 100.0)
            current_point += 1

            if progress % 10 == 0:
                if current_progress != progress:
                    current_progress = progress
                    print 'Adicionando pontos no grid ... {progress}'.format(progress=progress)


def generate_dummy_grid():
    grid_size = 20
    grid = Grid2D(0.0, grid_size, 0.0, grid_size)
    xyz_file = open('D:\\Dropbox\\Doutorado\\arvores\\dummy_grid.xyz', 'w')
    for block in grid.grid_blocks:
        x = block.center[0]
        y = block.center[1]
        z = math.sin(x) + math.cos(y) * 1.5
        xyz_file.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
    xyz_file.close()


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

    start = time.clock()

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

    end = time.clock()
    print 'Time: ' + str(end - start)


    print 'Escrevendo arquivo de pontos filtrados..'
    filtered_points_file = open('/Users/tsabino/Desktop/teste.xyz', 'w')
    for p in filtered_points:
        filtered_points_file.write(str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]) + '\n')
    filtered_points_file.close()


def filter_points_v2():
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

    X = np_points[:, 0]
    Y = np_points[:, 1]
    Z = np_points[:, 2]

    z_min = Z.min()
    Z = Z - z_min
    np_points[:, 2] = Z

    filtered_points = []

    print 'Coletando pontos minimos...'
    grid = Grid2D(x_min, x_max, y_min, y_max)

    start = time.clock()
    grid.add_points(np_points)
    end = time.clock()
    print 'Time: ' + str(end - start)


def kriging():
    #points_file = open('/Users/tsabino/Desktop/teste.xyz', 'r')
    points_file = open('D:\\Dropbox\\Doutorado\\arvores\\terrain_grid.xyz', 'r')
    #points_file = open('D:\\Dropbox\\Doutorado\\arvores\\dummy_grid.xyz', 'r')

    points = []
    for point_entry in points_file.readlines():
        point_data = point_entry.strip('\n').split()
        points.append([float(point_data[0]), float(point_data[1]), float(point_data[2])])
    points_file.close()

    np_points = np.array(points)

    X = np_points[:, [0, 1]]
    y = np_points[:, 2]

    print 'Fitting data...'
    gp = gaussian_process.GaussianProcess(theta0=20.0)
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

    grid_res = 0

    points_to_predict = []
    for x in np.linspace(x_min, x_max, 200):
        for y in np.linspace(y_min, y_max, 200):
            points_to_predict.append([x, y])
            points_to_predict.append([x, y])

    print 'Predicting...'
    Z_predicted = gp.predict(points_to_predict)

    #dtm_file = open('/Users/tsabino/Desktop/dtm.xyz', 'w')
    dtm_file = open('D:\\Dropbox\\Doutorado\\arvores\\dtm2.xyz', 'w')
    for i in xrange(len(points_to_predict)):
        dtm_file.write(str(points_to_predict[i][0]) + ' ' + str(points_to_predict[i][1]) + ' ' + str(Z_predicted[i]) + '\n')
    dtm_file.close()


if __name__ == '__main__':
    #generate_dummy_grid()
    #filter_points()
    #filter_points_v2()
    kriging()