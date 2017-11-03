import numpy as np
from sklearn.svm import SVC

rawData = np.genfromtxt('./data.csv', delimiter=',', skip_header=True)
target = rawData[:, [0]].transpose()[0]
data = rawData[:, 1:]

plant = dict()

plant['data'] = data
plant['target'] = target


