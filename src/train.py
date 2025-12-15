from load_data import load_data
from model_cnn import create_cnn
import os
model_path = "../hand_gesture_cnn.h5"
if os.path.exists(model_path):
    os.remove(model_path)
    print("Ancien modèle supprimé !")

# Charger les données
X_train, y_train, X_val, y_val, X_test, y_test, datagen = load_data()

num_classes = y_train.shape[1]

model = create_cnn(num_classes)

# Entraînement avec data augmentation
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=10
)

# Évaluation finale
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

# Sauvegarde
model.save("../hand_gesture_cnn_3classes.h5")

