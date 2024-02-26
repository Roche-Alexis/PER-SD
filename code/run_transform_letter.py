import cv2
import numpy as np
from torchvision import transforms
import onnxruntime as ort
from PIL import Image

# Initialiser la session ONNX
ort_session = ort.InferenceSession('./code/fine_tuned_mobileVIT.onnx')

# Définir les transformations (comme durant notre partie training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_and_predict(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.merge([gray_frame, gray_frame, gray_frame]) #convertie en gray scale et duplique channel comme pour training 
    
    #Convertit en PIL pour transform et return un numpy array
    image = Image.fromarray(gray_frame)
    image = transform(image)
    image = image.unsqueeze(0).numpy()

    # Prediction 
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    pred = np.argmax(ort_outs[0], axis=1)

    return chr(pred[0] + 65)  # Convertir en lettre

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    letter = preprocess_and_predict(frame)
    # Afficher la lettre prédite
    cv2.putText(frame, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
