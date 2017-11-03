import numpy as np
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score


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
print(y_test[0:5].transpose())
print(predictions[0:5].transpose())
difference = np.absolute(y_test-predictions)
print("Number of tests:" + str(np.alen(difference)))
print("Sample difference: " + str(difference[0:5]))
print("Average difference: " + str(np.average(difference)))
print("Maximum difference: " + str(np.max(difference)))
print("Minimum difference: " + str(np.min(difference)))

results = np.hstack((predictions, y_test.transpose))
np.savetxt('results.csv', results, delimiter=',')

# Accuracy metric used:
# https://en.wikipedia.org/wiki/Coefficient_of_determination
# closer to 1.0 is better
print(r2_score(y_test, predictions))
