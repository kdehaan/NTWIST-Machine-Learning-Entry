import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error


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

start = time.time()

mlp = MLPRegressor(hidden_layer_sizes=(35, 35, 30),
                   activation='relu',     # relu converges fastest and works well on this dataset
                   solver='adam',         # can use lbfgs for slightly greater accuracy, or adam for ~1/3 runtime
                   learning_rate_init=0.0008,  # seems to be the most consistently accurate
                   )
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

end = time.time()

#print(y_test[0:5])
#print(predictions[0:5])
difference = np.absolute(y_test-predictions)
print("Number of tests: " + str(np.alen(difference)))
#print("Sample differences: " + str(difference[0:5]))
print("Average difference: " + str(np.average(difference)))
print("Maximum difference: " + str(np.max(difference)))
print("Minimum difference: " + str(np.min(difference)))
print("Took " + str(end-start) + " seconds to converge")
results = np.hstack((predictions[np.newaxis].T, y_test[np.newaxis].T, difference[np.newaxis].T))

np.savetxt('results.csv', results, delimiter=',', header='Predicted,Actual,Difference')
print("Results saved to 'results.csv'")

# Accuracy metric used:
# https://en.wikipedia.org/wiki/Coefficient_of_determination
# closer to 1.0 is better
print("R2 Score: " + str(r2_score(y_test, predictions)))
print("Mean Squared Error: " + str(mean_squared_error(y_test, predictions)))
