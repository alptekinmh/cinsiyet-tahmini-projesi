import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Modeli yükle
MODEL_PATH = "models/gender_model_finetuned.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Sınıf isimleri
class_names = ["Erkek", "Kadın"]

# Resmi işleme fonksiyonu
def preprocess_image(img_path, target_size=(128, 128)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Tahmin fonksiyonu
def predict_image():
    global panel
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return


    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk

    # Tahmin
    processed_img = preprocess_image(file_path)
    prediction = model.predict(processed_img)[0][0]

    if prediction < 0.5:
        label = class_names[0]  # Erkek
        confidence = (1 - prediction) * 100
    else:
        label = class_names[1]  # Kadın
        confidence = prediction * 100

    messagebox.showinfo("Tahmin Sonucu", f"{label} (%{confidence:.2f} güven)")

# GUI
root = tk.Tk()
root.title("Cinsiyet Tahmini")
root.geometry("400x400")

btn = tk.Button(root, text="Fotoğraf Yükle ve Tahmin Et", command=predict_image)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

root.mainloop()
