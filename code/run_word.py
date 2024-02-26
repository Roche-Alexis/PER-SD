import tkinter as tk
from tkinter import Label, Button, Frame
import cv2
from PIL import Image, ImageTk
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import torch


video_path = "video.mp4"



def get_pred():

    # Initialiser MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

    def extract_keypoints(results, hands_only=True):
        if hands_only:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([lh, rh])
        else:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
            return np.concatenate([pose, face, lh, rh])
          

    def process_video(hands_only=True):
        cap = cv2.VideoCapture(video_path)
        all_keypoints_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            frame_kp = extract_keypoints(results, hands_only=hands_only)
            all_keypoints_list.append(frame_kp)

        cap.release()
        holistic.close()

        keypoints_array = np.array(all_keypoints_list)
        return keypoints_array

    # Chemin vers le modèle ONNX exporté et la vidéo d'entrée
    model_path = './code/model_simplifie.onnx'

    # Créer une session ONNX Runtime
    ort_session = ort.InferenceSession(model_path)

    # Extraire les keypoints de la vidéo
    video_kp = process_video(video_path)

    max_length = 90

    # Prétraitement des keypoints (padding ou troncature pour obtenir 124 frames)
    if video_kp.shape[0] < max_length:
        padding = np.zeros(max_length - video_kp.shape[0], video_kp.shape[1])
        video_kp_padded = np.vstack((video_kp, padding))
    elif video_kp.shape[0] >max_length:
        video_kp_padded = video_kp[max_length]
    else:
        video_kp_padded = video_kp

    # Vérification de la forme des données
    assert video_kp_padded.shape == (max_length, 126), "La forme ajustée des données n'est pas correcte."

    # Convertir en tensor PyTorch et préparer les données d'entrée pour ONNX Runtime
    video_kp_tensor = torch.tensor(video_kp_padded, dtype=torch.float32).unsqueeze(0)
    ort_inputs = {ort_session.get_inputs()[0].name: video_kp_tensor.numpy()}

    # Exécuter l'inférence
    ort_outs = ort_session.run(None, ort_inputs)
    predicted_class = np.argmax(ort_outs[0], axis=1)
    res = int(predicted_class[0])
    print(res)
    return res




class VideoRecorderApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_frame = Frame(window)
        self.btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_start = Button(self.btn_frame, text="Enregistrer", width=15, command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT)

        self.is_recording = False
        self.frame_count = 0
        self.total_frames = 90
        self.out = None  # Variable pour l'objet VideoWriter

        self.label_info = Label(self.btn_frame, text="Prêt")
        self.label_info.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.update()

        self.bind_keys()  # Lier les touches à la fonction start_recording



        self.window.mainloop()


    def bind_keys(self):
        self.window.bind('<space>', lambda event: self.start_recording())
  
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.frame_count = 0
            self.label_info.config(text=f"Enregistrement... {self.frame_count}/{self.total_frames}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
            self.out = cv2.VideoWriter(video_path, fourcc, 20.0, (int(self.vid.get(3)), int(self.vid.get(4))))

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            if self.is_recording and self.frame_count < self.total_frames:
                if self.out is not None:
                    self.out.write(frame)
                self.frame_count += 1
                self.label_info.config(text=f"Enregistrement... {self.frame_count}/{self.total_frames}")

            if self.frame_count == self.total_frames and self.is_recording:
                self.is_recording = False
                if self.out is not None:
                    self.out.release()
                    self.out = None
                self.label_info.config(text="Enregistrement terminé")
                self.run_prediction()

        self.window.after(5, self.update)

    def run_prediction(self):
        self.stop_camera()  # Arrêter le flux de la caméra et afficher "Traitement..."
        print('lance getting prediction')
        prediction = get_pred()
        mapping = {0: 'yes', 1: 'no', 2: 'eat'}
        predicted_word = mapping.get(prediction, "Inconnu")
        self.label_info.config(text=f"Prédiction : {predicted_word}", font=('Helvetica', 18, 'bold'), fg="aquamarine")

        self.start_camera()  # Reprendre le flux de la caméra

    def stop_camera(self):
        print('call stoo camera')
        if self.vid.isOpened():
            self.vid.release()  # Libérer l'objet VideoCapture
        print('cam record released')

        # Effacer le canvas et afficher un écran noir
        self.canvas.delete("all")  # Supprime tout du canvas
        self.canvas.create_rectangle(0, 0, self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT), fill="black")
        
        # Afficher le texte "Traitement en cours" au centre de l'écran noir
        self.canvas.create_text(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) / 2+140, self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2+140, text="Traitement en cours", fill="black", font=('Helvetica', 16, 'bold'))
        print('end fonction stop cam')
        self.canvas.update_idletasks()  # Forcer la mise à jour de l'interface graphique


    def start_camera(self):
        self.vid = cv2.VideoCapture(self.video_source)  # Réinitialiser l'objet VideoCapture

    def __del__(self):
        self.stop_camera()
        if self.out is not None:
            self.out.release()


# Créer une fenêtre et passer une instance de VideoRecorderApp
root = tk.Tk()
app = VideoRecorderApp(root, "Enregistreur vidéo Tkinter")