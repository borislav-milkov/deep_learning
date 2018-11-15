import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('challenge_dataset.txt', names=('x','y'))
x = df[['x']].values.tolist()
y = df[['y']].values.tolist()

model = linear_model.LinearRegression()
model.fit(x, y)

plt.scatter(x, y)
plt.plot(x, model.predict(x))
plt.show()