import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model

# --- Configuration ---
MODEL_PATH = "../hand_gesture_cnn.h5" 

# Dictionnaire de mapping GESTES -> INDEX du modèle (basé sur l'ASL : 0=A, 1=B, etc.)
# Ceci est l'élément clé pour afficher le nom du geste à la place de la lettre.
GESTURES_MAP = {
    0: 'POING FERMÉ (A)',      # Index 0
    1: 'PAUME OUVERTE (B)',    # Index 1
    2: 'MAIN EN C (C)',        # Index 2
    3: 'DOIGT POINTÉ (D)',     # Index 3
    4: 'MAIN COURBÉE (E)',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'PETIT DOIGT LEVÉ (I)',
    # 9 (J) est sauté dans le dataset MNIST
    10: 'K',
    11: 'L (Pouce/Index levés)',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'POING FERMÉ (S)',
    19: 'T',
    20: 'U',
    21: 'SIGNE DE VICTOIRE (V)', # Index 21
    22: 'W',
    23: 'X',
    24: 'I LOVE YOU (Y)'       # Index 24
}

try:
    # Chargement du modèle
    model = load_model(MODEL_PATH)
    num_classes = model.output_shape[-1]
    
    # Nous vérifions que le nombre de classes correspond à notre mapping
    if num_classes != len(GESTURES_MAP):
        print(f"ATTENTION : Le modèle a {num_classes} sorties, mais le mapping en a {len(GESTURES_MAP)}.")
    
    print(f"Modèle chargé avec {num_classes} classes.")
except Exception as e:
    print(f"Erreur de chargement du modèle : {e}")
    print("Veuillez vérifier le nom du fichier et le chemin.")
    exit()

cap = cv2.VideoCapture(0)

# Définition des coordonnées de la Région d'Intérêt (ROI)
roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 200 # (x, y, largeur, hauteur)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Dessiner le rectangle de la ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    
    # 2. Extraire la région d'intérêt de l'image
    hand_roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # --- 3. PRÉTRAITEMENT OPTIMISÉ (Binarisation) ---
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, processed_image = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Redimensionnement et préparation pour le modèle (28x28x1 normalisé)
    img = cv2.resize(processed_image, (28, 28)).reshape(1, 28, 28, 1) / 255.0

    # 4. Prédiction
    pred = model.predict(img, verbose=0)
    pred_class = np.argmax(pred)
    confidence = np.max(pred) 

    # Obtention du label (Utilisation du nouveau dictionnaire GESTURES_MAP)
    label = GESTURES_MAP.get(pred_class, f"INDEX {pred_class} (Inconnu)") 

    # 5. Affichage du résultat
    text_prediction = f'Geste: {label} ({confidence*100:.2f}%)'
    
    # Positionner le texte au-dessus de la ROI
    cv2.putText(frame, text_prediction, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Afficher la fenêtre de la caméra
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()