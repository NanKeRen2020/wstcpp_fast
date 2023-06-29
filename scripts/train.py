import json
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import os
import struct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_mnist(path, kind="train"):
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte")

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# Load the data 
# **NOTE** not all training data is in this repo. However the  MNIST data is 
# readily available online and should be places in the data folder
print("Loading data...")
x_train, y_train = load_mnist("data/", kind="train")
x_test, y_test = load_mnist("data/", kind="t10k")

# print shape of train and test data
print("Shape of x_train: ", x_train.shape)
print("Shape of x_test: ", x_test.shape)

# Normalize the input data
print("Normalizing data...")
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the labels data
print("Reshaping labels...")
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# print first label
print("First label: ", y_train[0])

# print first image as grayscale in the terminal
print("First image: ")
for i in range(28):
    for j in range(28):
        if x_train[0][i * 28 + j] > 0.90:
            print("*", end=" ")
        else:
            print("_", end=" ")
    print()


from kymatio import Scattering1D, Scattering2D

x = Scattering2D(
    J=2,
    shape=(28, 28),
    L=8,
    max_order=2,
)

# print shape of x_train and x and y_train and y
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y: ", y_train.shape)

# scatter each training image
print("Scattering each training image...")
x_scattered = np.array([x(x_train[i].reshape(28, 28)) for i in range(len(x_train))])

# reshape x_scattered
x_scattered = x_scattered.reshape(x_scattered.shape[0], -1)

# print shape of x_scattered
print("Shape of x_scattered: ", x_scattered.shape)

# Fit a logistic regression model
print("Fitting a logistic regression model...")
clf = LogisticRegression(
    random_state=0, solver="saga", multi_class="multinomial", max_iter=100
)

# actually train the model
print("Training the model...")
clf.fit(x_scattered, y_train.ravel())

# write coefficients and intercepts to a single file one line for intercepts and one line for coefficients
print("Writing coefficients and intercepts to a single file...")
with open("data/model.txt", "w") as f:
    c = clf.coef_
    c = np.round(c, 16)
    c = c.tolist()
    f.write(str(c))
    f.write("\n")
    i = clf.intercept_
    i = np.round(i, 16)
    i = i.tolist()
    f.write(str(i))
    f.write("\n")
