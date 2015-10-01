
import math

for i_counter in range(0, 15):
    h = 0.1 / math.pow(10.0, i_counter)
    lim1 = (math.pow(2.7, h) - 1.0) / h
    lim2 = (math.pow(2.8, h) - 1.0) / h
    lim3 = (math.pow(2.71, h) - 1.0) /h
    print h, lim1, lim2, lim3;