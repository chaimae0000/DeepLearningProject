from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_cnn(num_classes):

    model = Sequential([

        # -------------------- Entrée --------------------
        Input(shape=(28, 28, 1)),  # forme des images MNIST

        # -------------------- Bloc Convolutionnel 1 --------------------
        Conv2D(32, (3,3), activation='relu'),
        BatchNormalization(),          # stabilise et accélère l’apprentissage
        MaxPooling2D(2,2),

        # -------------------- Bloc Convolutionnel 2 --------------------
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        # -------------------- Passage à un vecteur --------------------
        Flatten(),

        # -------------------- Couche dense --------------------
        Dense(128, activation='relu'),
        Dropout(0.4),       # légerement réduit pour éviter trop de perte d’information

        # -------------------- Sortie --------------------
        Dense(num_classes, activation='softmax')  # probas des classes
    ])

    # Compilation du modèle
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
