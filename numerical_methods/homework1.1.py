import numpy as np
from scipy.linalg import hilbert


def main():
    for n in range(12, 20):
        H = hilbert(n)
        x = [1000 for _ in range(n)]
        b = H.dot(x)

        x = np.linalg.solve(H, b)

        Hx = H.dot(x)
        r = b - Hx

        print np.linalg.norm(r, np.inf)

        iter = 0
        while np.linalg.norm(r, np.inf) > 1e-5 and iter < 50:
            z = np.linalg.solve(H, r)
            x = x + z
            Ax = H.dot(x)
            r = b - Ax
            iter += 1

        print iter, x


if __name__ == '__main__':
    main()