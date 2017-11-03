import numpy as np
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix


rawData = np.genfromtxt('./data.csv', delimiter=',', skip_header=True)
target = rawData[:, [0]].transpose()[0]
data = rawData[:, 1:]

plant = dict()

plant['data'] = data
plant['target'] = target

x = plant['data']
y = plant['target']

x_train, x_test, y_train, y_test = train_test_split(x, y)

scaler = StandardScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPRegressor(hidden_layer_sizes=(30, 30, 30))
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)
print(y_test[0:5])
print(predictions[0:5])
#
#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))
