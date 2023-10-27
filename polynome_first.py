import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)

#creation of dataset
x,y = make_regression(n_samples=1000, n_features=1, noise=10)
y = y**2 #y ne varie plus linéairement selon x!

#on ajoute des variables polynomiales dans notre dataset
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x = poly_features.fit_transform(x)
plt.scatter(x[:,0], y)
print(x.shape)

plt.show()

# Création du modèle et entrainement
model = SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x, y)
print('coeff R2 = ', model.score(x, y))

plt.scatter(x[:,0], y, marker='o')
plt.plot(x[:,0], model.predict(x), c='r',  marker='+')
plt.show()