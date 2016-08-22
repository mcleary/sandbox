import numpy as np


def generate_data(data_index):
    if data_index == 1:
        raw_data = np.array([[2, 2, 3],
                             [3, 7, 4],
                             [9, 9, 2],
                             [6, 5, 4],
                             [5, 3, 6]])
        return raw_data

    if data_index == 2:
        raw_data = np.array([[4.0, 2.0, 0.6],
                             [4.2, 2.1, 0.59],
                             [3.9, 2.0, 0.58],
                             [4.3, 2.1, 0.62],
                             [4.1, 2.2, 0.63]])
        return raw_data

    if data_index == 3:
        x = [-113.24195, -71.99837, -81.89808, -85.55240, -109.26020,
             -81.34538, -119.17687, -83.15635, -98.28343, -97.65484,
             -117.88467, -115.25322, -115.46664, -75.67949, -115.87845,
             -120.80413, -97.25267, -87.91998, -112.61898, -123.33534,
             -116.23355, -98.61034, -97.24387, -76.30576, -99.07240,
             -117.47637, -118.37720, -123.65208, -115.90662, -120.66805,
             -89.07153, -98.86473, -107.32864, -90.17405, -81.54257,
             -100.15700, -88.94991, -104.93632, -116.43452, -68.96927,
             -81.35424, -121.94638, -78.42431, -78.12326, -81.68185,
             -69.62688, -87.61224, -100.95793, -69.80093, -73.76623]

        y = [40.01692, 38.22831, 28.22333, 49.05601, 25.85637, 46.20760, 40.99168,
             25.97721, 30.04281, 27.95561, 27.94152, 45.86130, 28.86185, 31.59320,
             30.28169, 41.34342, 41.44540, 35.79267, 34.18067, 34.16108, 39.51218,
             38.34534, 48.62712, 30.45645, 33.74617, 35.60941, 35.24219, 45.11175,
             38.25033, 29.73833, 45.99965, 39.71045, 25.36877, 25.78541, 38.80088,
             39.95517, 34.40401, 49.08081, 38.36950, 39.08793, 45.36972, 37.61989,
             44.63971, 47.45942, 48.30961, 41.63830, 32.61844, 44.63154, 31.80567,
             35.94512]

        z = [-2.07465272, -0.58196695, -1.54724761, 0.99257163, 0.91583913,
             -0.25059097, 1.36857840, -1.47103697, -0.31374941, 1.61368903,
             -1.15440618, -0.42662456, 1.29879333, -0.22339007, -1.63228467,
             -0.08514583, 0.33372904, 0.64368165, 2.00167176, 1.39925425,
             -0.43725294, 0.32029398, -2.15905422, 1.04484000, 0.50208078,
             -1.91530492, -1.61636516, -0.10247989, 0.75882837, -0.46380019,
             1.74592039, 0.35598781, -1.22399710, 1.10366976, 0.32875505,
             -0.99202180, -0.79344508, 0.05729978, -0.56246305, -1.21235859,
             -0.24844030, 0.70272852, -1.53597275, 0.25796196, -0.04232428,
             0.51174422, 0.24669877, -1.95680816, -0.72088498, -1.54924964]

        raw_data = []
        for i in xrange(len(x)):
            raw_data.append([x[i], y[i], z[i]])

        return np.array(raw_data)

    if data_index == 4:
        filepath = '/Users/mcleary/Desktop/mpi-dtm2.xyz'
        xyz_file = open(filepath, 'r')

        print 'Lendo arquivo ' + filepath + ' ...'
        xyz_contents = []
        for xyz_line in xyz_file.readlines():
            xyz_contents.append(xyz_line.strip('\n').strip('\r'))

        xyz_file.close()

        points = []

        print 'Extraindo dados ...'
        for xyz_entry in xyz_contents:
            xyz_data = xyz_entry.split()

            if len(xyz_data) == 3:
                points.append([float(xyz_data[0]), float(xyz_data[1]), float(xyz_data[2])])
            else:
                points.append([float(xyz_data[2]), float(xyz_data[3]), float(xyz_data[4])])

        return np.array(points)