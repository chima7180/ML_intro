import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# Etape 1 :  creation DS

np.random.seed(0) # pour toujours reproduire le meme dataset
x, y = make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x, y) # afficher les résultats. X en abscisse et y en ordonnée
plt.show()

print(x.shape)
print(y.shape)

# redimensionner y
y = y.reshape(y.shape[0], 1)
print(y.shape)

# Création de la matrice X qui contient la colonne de Biais.
# Création de la matrice X qui contient la colonne de Biais. Pour ca, on colle l'un contre l'autre le vecteur
#x et un vecteur 1 (avec np.ones) de dimension égale a celle de x
X = np.hstack((x, np.ones(x.shape)))
print(X.shape)

np.random.seed(0) # pour produire toujours le meme vecteur theta aléatoire
theta = np.random.randn(2, 1)
print(theta)

# modèle linéaire
def model(X, theta):
    return X.dot(theta)
plt.scatter(x, y)
plt.plot(x, model(X, theta), c='r')
plt.show()


# Etape 3 : Fonction cout : Erreur quadratique moyenne

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

checkF = cost_function(X, y, theta)
print(checkF)

# Etape 4 Gradient descent and Gradients

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(
        n_iterations)  # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele

    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y,theta)  # mise a jour du parametre theta (formule du gradient descent)
        cost_history[i] = cost_function(X, y, theta)  # on enregistre la valeur du Cout au tour i dans cost_history[i]

    return theta, cost_history


# Etape 5 machine Learning

n_iterations = 1000
learning_rate = 0.01

theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
print(theta_final)

# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, theta_final)

# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x, y)
plt.plot(x, predictions, c='r')
plt.show()

#Bonus and Troubleshooting
# Courbe d'apprentissage
plt.plot(range(n_iterations), cost_history)
plt.show()

# note
def coef_determination(y, pred):
    u = ((y - pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1 - u/v

checkc = coef_determination(y, predictions)
print(checkc)






