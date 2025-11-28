import pandas as pd
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical

def load_data():
    # Chargement des données
    train = pd.read_csv("dataset/sign_mnist_train.csv")
    test = pd.read_csv("dataset/sign_mnist_test.csv")

    # Séparation des caractéristiques et des étiquettes
    y_train = train['label'].values
    X_train = train.drop('label', axis=1).values

    y_test = test['label'].values
    X_test = test.drop('label', axis=1).values
    
    # Reshape(nb_images, hauteur, largeur, canaux) et normalisation(/255)
    
    X_train = X_train.reshape(-1,28,28,1)/255.0
    #-1 → Python calcule automatiquement combien il y a d’images
    #28,28 → forme de l’image
    #1 → noir et blanc
    #Diviser par 255 → valeur du pixel devient entre 0 et 1, ce qui aide le modèle à apprendre.
    X_test = X_test.reshape(-1,28,28,1)/255.0

    # Calculer dynamiquement le nombre de classes
    num_classes = np.max(y_train) + 1  # Ajoute 1 pour inclure la valeur maximale
    
    #Transformation One-Hot Encoding (si on a 5 classes, alors le label "2" devient: [0,0,1,0,0])
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, y_train, X_test, y_test
