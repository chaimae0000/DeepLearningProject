import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

model = load_model("../hand_gesture_cnn.h5")
num_classes = model.output_shape[-1]
labels = {i: chr(65+i) for i in range(num_classes)}  # A à Y sans J et Z


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (28,28)).reshape(1,28,28,1)/255.0

    pred = model.predict(img)
    pred_class = np.argmax(pred)
    label = labels.get(pred_class, "Unknown")  # "Unknown" si clé manquante


    cv2.putText(frame, f'Prediction: {label}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
