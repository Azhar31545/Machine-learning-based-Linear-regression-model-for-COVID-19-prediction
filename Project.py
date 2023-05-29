import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv('deathcases.csv', sep=',')
data = data[['id', 'cases']]
print('-' * 30)
print('HEAD')
print('-' * 30)
print(data.head())

# Prepare data
print('-' * 30)
print('Prepare Data')
print('-' * 30)
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['cases']).reshape(-1, 1)

plt.xlabel("Days")
plt.ylabel("Cases")
plt.title("Plot for Confirmed Cases")
plt.plot(y, '-m', label="plot")
plt.legend(loc="best")

polyFeat = PolynomialFeatures(degree=4)
x_poly = polyFeat.fit_transform(x)

# Training data
print('-' * 30)
print('Training Data')
print('-' * 30)
model = LinearRegression()
model.fit(x_poly, y)
accuracy = model.score(x_poly, y)
print(f'Accuracy: {round(accuracy * 100, 3)}%')

y_predicted = model.predict(x_poly)
plt.title("Plot for Death Cases")
plt.plot(y_predicted, '--b', label="Best fit line")
plt.legend(loc="best")

# Prediction
days = 1
print('-' * 30)
print('PREDICTION')
print('-' * 30)
x_pred = np.array(list(range(1, 100 + days))).reshape(-1, 1)
x_pred_poly = polyFeat.transform(x_pred)
y_pred = model.predict(x_pred_poly)

plt.title("Prediction for Death Cases")
plt.plot(y_pred, '--r', label="Prediction")
plt.plot(y_predicted, '--b',)
plt.legend(loc="best")

plt.show()
