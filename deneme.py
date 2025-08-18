import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

model_path = r"C:\Users\alptekin\Desktop\proje\models\gender_model_finetuned.h5"
model = load_model(model_path)

def has_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def predict_gender(image_path):
    if not has_face(image_path):
        return "Yüz algılanamadı!"  

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Erkek" if pred < 0.5 else "Kadın"

# GUI 
def select_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("JPEG dosyaları", "*.jpg"), ("PNG dosyaları", "*.png")]
    )
    if not file_path:
        return

   
    result = predict_gender(file_path)
    result_label.config(text=f"Tahmin: {result}")

    
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


root = tk.Tk()
root.title("Cinsiyet Tahmin Uygulaması")

btn = tk.Button(root, text="Fotoğraf Seç", command=select_image)
btn.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()
