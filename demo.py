from mxnet import nd

a: nd.NDArray = nd.array([[1, 2], [3, 4]])
b: nd.NDArray = nd.array([2, 2])
c: nd.NDArray = nd.array([1, 1])
print(a * b + c)
