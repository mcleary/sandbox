


#xk, yk = grad_descent_fixed_step(100, 100, max_iter=20)
xk, yk = grad_descent_btls(100, 100, max_iter=20)

graph_range = [-200, 200]

X = np.arange(graph_range[0], graph_range[1], 1)
Y = np.arange(graph_range[0], graph_range[1], 1)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

fig = plt.figure()
plt.contour(X, Y, Z)
plt.plot(xk, yk)
plt.axis([graph_range[0], graph_range[1], graph_range[0], graph_range[1]])
plt.show()

