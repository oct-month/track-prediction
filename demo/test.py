from mxnet.gluon import data
from mxnet.gluon.data.vision import datasets, transforms


mnist_train = datasets.FashionMNIST(train=True)

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)
])
mnist_train = mnist_train.transform_first(transformer, lazy=False)

batch_size = 256
train_data = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

for X, y in train_data:
    print(X.shape, y.shape)
    break
