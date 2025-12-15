import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data():
    #  Charger les données
    train = pd.read_csv("dataset/sign_mnist_train.csv")
    test = pd.read_csv("dataset/sign_mnist_test.csv")

    #  Labels choisis pour ton projet
    # poing fermé, feuille, ciseaux(en se basent sur le data_visualization)
    #cet un dictionnaire qui contien kle/Value
    selected_labels = {
        0: 0,   # poing fermé
        22: 1,  # feuille
        2: 2    # ciseaux
        }

    #  Filtrer uniquement les 3 classes supprimer 
    # tout le 22 labels et garder juste les 3 
    train = train[train['label'].isin(selected_labels.keys())]
    test = test[test['label'].isin(selected_labels.keys())]

    #  Séparer X images/ y labels
    y = train['label'].map(selected_labels).values
    X = train.drop('label', axis=1).values

    y_test = test['label'].map(selected_labels).values
    X_test = test.drop('label', axis=1).values

    # Reshape + normalisation en reshape car le CNN comprend
    # juste les images nes pas les plates
    X = X.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # 6️⃣ One-hot encoding (3 classes) en trensfere chqaue lqbel 
    # soit 1 ou bien 2 ou bien 0 a des chaine 1->[0,1,0]
    num_classes = 3
    y = to_categorical(y, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 7️⃣ Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y.argmax(axis=1)
    )

    # 8️⃣ Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    return X_train, y_train, X_val, y_val, X_test, y_test, datagen
