#! /usr/bin/python3

import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

df = pd.read_fwf('animal.txt')

X = df[['Brain']]

y = df[['Body']]

lm = LinearRegression()

lm.fit(X, y)

plt.scatter(X, y)

plt.plot(X, lm.predict(X))

plt.show()
