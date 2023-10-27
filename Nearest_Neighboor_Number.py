import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

# Import de la base de données digits
digits = load_digits()

x = digits.data
y = digits.target

print('Dimension de X:', x.shape)

# Affichage d'un des chiffres
plt.imshow(digits.images[0], cmap='gray')
plt.show()

# Entraînement du modèle
model = KNeighborsClassifier()
model.fit(x, y)
score = model.score(x, y)
print('Score du modèle sur les données d\'entraînement:', score)

# Test du modèle sur l'image 100
test = digits.images[100].reshape(1, -1)
plt.imshow(digits.images[100], cmap='gray')
predicted = model.predict(test)
print(predicted)
plt.show()
