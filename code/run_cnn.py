import cv2
import numpy as np
import onnxruntime as ort

import os

# Définition de la fonction pour recadrer une image centrée
def center_crop(frame):
    h, w, _ = frame.shape
    start = abs(h - w) // 2
    if h > w:
        return frame[start: start + w, :]
    else:
        return frame[:, start: start + h]

# Définition de la fonction principale
def main():

    print(os.listdir())
    # Constantes
    index_to_letter = list('ABCDEFGHIKLMNOPQRSTUVWXY')
    mean = 0.485 * 255.
    std = 0.229 * 255.

    # Chargement du modèle ONNX spécifié
    ort_session = ort.InferenceSession("./code/model_cnn.onnx")  # Assurez-vous que le nom du modèle est correct

    # Capture vidéo à partir de la webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Capture d'une image
        ret, frame = cap.read()
        if not ret:
            break  # Sortie de la boucle si la capture a échoué

        # Prétraitement de l'image
        frame = center_crop(frame)  # Recadrage au centre
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
        x = cv2.resize(frame, (28, 28))  # Redimensionnement à la taille requise
        x = (x - mean) / std  # Normalisation

        # Mise en forme de l'image pour correspondre à l'entrée du modèle
        x = x.reshape(1, 1, 28, 28).astype(np.float32)

        # Exécution du modèle sur l'image prétraitée
        y = ort_session.run(None, {'input': x})[0]

        # Interprétation des résultats pour obtenir la lettre prédite
        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        # Affichage de la lettre prédite sur l'image
        cv2.putText(frame, letter, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), thickness=2)

        # Affichage de l'image avec la lettre prédite
        cv2.imshow("Sign Language Translator", frame)

        # Sortie de la boucle si la touche 'q' est enfoncée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération de la capture vidéo et fermeture des fenêtres
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
