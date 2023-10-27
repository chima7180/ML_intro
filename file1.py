import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor

# Génération du Dataset
np.random.seed(0)
x, y = make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x, y)
plt.show()

# Création du modèle et entrainement
model = SGDRegressor(max_iter=1000, eta0=0.001)
model.fit(x, y)

# Score du modèle et génération de prédiction
print('coeff R2 = ', model.score(x, y))
plt.scatter(x, y)
plt.plot(x, model.predict(x), c='red', lw=3)
plt.show()


