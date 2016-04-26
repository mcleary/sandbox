import numpy as np
from scipy.linalg import hilbert

use_choleschy = True

for n in range(1, 4):
    H = hilbert(n)
    x = np.ones(n)
    b = H.dot(x)    # H * x

    if use_choleschy:
        L = np.linalg.cholesky(H)
        T = L.T.conj()

        y = np.linalg.solve(L, b)
        x_hat = np.linalg.solve(T, y)
    else:
        x_hat = np.linalg.solve(H, b)

    cond_H = np.linalg.cond(H)
    dx = np.linalg.norm(x_hat, np.inf)

    print "n: {n: <5} dx: {dx: <15} cond(H): {cond_H: <10}".format(n=n, dx=dx, cond_H=cond_H)