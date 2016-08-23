import numpy as np

def F(x):
    def f1(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1+c2+c3+c4-2.0

    def f2(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1*x1 + c2*x2 + c3*x3 + c4*x4

    def f3(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1*x1**2 + c2*x2**2 + c3*x3**2 + c4*x4**2 - 2.0/3.0

    def f4(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1*x1**3 + c2*x2**3 + c3*x3**3 + c4*x4**3

    def f5(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1*x1**4 + c2*x2**4 + c3*x3**4 + c4*x4**4 - 2.0/5.0

    def f6(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1*x1**5 + c2*x2**5 + c3*x3**5 + c4*x4**5

    def f7(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1**x1**6 + c2*x2**6 + c3*x3**6 + c4*x4**6 - 2.0/7.0

    def f8(c1, c2, c3, c4, x1, x2, x3, x4):
        return c1**x1**7 + c2*x2**7 + c3*x3**7 + c4**x4**7

    f_list = [f1, f2, f3, f4, f5, f6, f7, f8]
    result = []
    for f in f_list:
        result += [f(x[0], x[1], x[2], x[3], -1.0, x[5], x[6], 1.0)]
    return result


def JacF(x):
    def grad_f1(c1, c2, c3, c4, x1, x2, x3, x4):
        return [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    def grad_f2(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1, x2, x3, x4, c1, c2, c3, c4]

    def grad_f3(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1**2, x2**2, x3**2, x4**2, 2.0*c1*x1, 2.0*c2*x2, 2.0*c3*x3, 2.0*c4*x4]

    def grad_f4(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1**3, x2**3, x3**3, x4**3, 3.0*c1*x1**2, 3.0*c2*x2**2, 3.0*c3*x3**2, 3.0*c4*x4**2]

    def grad_f5(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1**4, x2**4, x3**4, x4**4, 4.0*c1*x1**3, 4.0*c2*x2**3, 4.0*c3*x3**3, 4.0*c4*x4**3]

    def grad_f6(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1**5, x2**5, x3**5, x4**5, 5.0*c1*x1**4, 5.0*c2*x2**4, 5.0*c3*x3**4, 5.0*c4*x4**4]

    def grad_f7(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1**6, x2**6, x3**6, x4**6, 6.0*c1*x1**5, 6.0*c2*x2**5, 6.0*c3*x3**5, 6.0*c4*x4**5]

    def grad_f8(c1, c2, c3, c4, x1, x2, x3, x4):
        return [x1**7, x2**7, x3**7, x4**7, 7.0*c1*x1**6, 7.0*c2*x2**6, 7.0*c3*x3**6, 7.0*c4*x4**6]

    grad = [grad_f1, grad_f2, grad_f3, grad_f4, grad_f5, grad_f6, grad_f7, grad_f8]
    result = []
    for f in grad:
        result += [f(x[0], x[1], x[2], x[3], -1.0, x[5], x[6], 1.0)]
    return result


def main():
    x0 = [0.5, 0.5, 0.5, 0.5, -1.0, -0.5, 0.5, 1.0]

    x_new = np.array(x0)
    x_old = x_new

    for iter in range(50):
        f_x = np.array(F(x_old))
        jf_x = JacF(x_old)
        y = np.linalg.solve(jf_x, -f_x)
        x_new = x_old + y
        tol = np.linalg.norm(x_new - x_old)
        x_old = x_new
        if tol < 0.3:
            print iter, x_new
            break

if __name__ == '__main__':
    main()