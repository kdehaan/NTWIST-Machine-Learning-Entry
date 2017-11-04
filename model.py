import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

data_type = '-1'
while True:
    data_type = input("Are you entering data with known actual resulting values? (i.e., a 30 column csv) [y/n]: ").lower()
    if data_type == 'y':
        break
    if data_type == 'n':
        break
    print('Invalid selection')

if data_type == 'n':
    model_file = input('Model file (default is ./sample_model.pkl): ')
    if not model_file:
        model_file = './sample_model.pkl'

    mlp = joblib.load(model_file)

    testing_data = input("Data file (29 column csv, default is ./sample_tests.csv): ")
    if not testing_data:
        testing_data = './sample_tests.csv'

    x_test = np.genfromtxt(testing_data, delimiter=',')

    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    predictions = mlp.predict(x_test)
    np.savetxt('predictions.csv', predictions, delimiter=',', header='Predicted Output')
    print('Predictions saved in predictions.csv')


if data_type == 'y':
    model_file = input('Model file (default is ./sample_model.pkl): ')
    if not model_file:
        model_file = './sample_model.pkl'

    mlp = joblib.load(model_file)

    testing_data = input("Data file (30 column csv, default is ./data.csv): ")
    if not testing_data:
        testing_data = './data.csv'

    rawData = np.genfromtxt(testing_data, delimiter=',', skip_header=True)
    target = rawData[:, [0]].transpose()[0]
    data = rawData[:, 1:]

    x_test = data
    y_test = target

    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    predictions = mlp.predict(x_test)

    difference = np.absolute(y_test - predictions)

    print("Number of tests: " + str(np.alen(difference)))
    print("Average difference: " + str(np.average(difference)))
    print("Maximum difference: " + str(np.max(difference)))
    print("Minimum difference: " + str(np.min(difference)))

    results = np.hstack((predictions[np.newaxis].T, y_test[np.newaxis].T, difference[np.newaxis].T))

    # Accuracy metric used:
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    # closer to 1.0 is better
    print("R2 Score: " + str(r2_score(y_test, predictions)))
    print("Mean Squared Error: " + str(mean_squared_error(y_test, predictions)))

    np.savetxt('model_results.csv', results, delimiter=',', header='Predicted,Actual,Difference')
    print("Results saved to 'model_results.csv'")
