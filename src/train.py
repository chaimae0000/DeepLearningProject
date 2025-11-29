from load_data import load_data
from model_cnn import create_cnn
import numpy as np

X_train, y_train, X_test, y_test = load_data()

# Détecter dynamiquement le nombre de classes
num_classes = y_train.shape[1]

model = create_cnn(num_classes)

# Entraînement
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Évaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Sauvegarder le modèle
model.save("hand_gesture_cnn.h5")
