# Yüz Algılama Destekli Cinsiyet Tahmin Sistemi (Gender Prediction System)

Bu proje, görüntü işleme (Computer Vision) ve derin öğrenme (Deep Learning) tekniklerini kullanarak insan yüzlerinden yüksek doğrulukla cinsiyet tahmini yapan uçtan uca bir masaüstü yapay zeka uygulamasıdır. 

Sistem, önceden eğitilmiş **MobileNetV2** mimarisinin ince ayar (fine-tuning) yöntemiyle optimize edilmesiyle oluşturulmuş olup, hatalı tahminleri engellemek için **OpenCV (Haar Cascades)** tabanlı bir ön işleme (yüz doğrulama) katmanı içermektedir.

## 🚀 Teknolojiler ve Mimariler
* **Dil:** Python 3.x
* **Derin Öğrenme:** TensorFlow, Keras (MobileNetV2)
* **Görüntü İşleme:** OpenCV, PIL (Pillow)
* **Arayüz (GUI):** Tkinter
* **Veri İşleme:** NumPy, ImageDataGenerator

## 🧠 Özellikler
1. **Transfer Learning (MobileNetV2):** ImageNet ağırlıkları kullanılarak modelin ilk 100 katmanı dondurulmuş (frozen) ve özellik çıkarımı için kullanılmıştır. Sınıflandırma katmanları projeye özel olarak yeniden eğitilmiştir.
2. **Yüz Doğrulama Katmanı (OpenCV):** `deneme.py` modülü, modele veri beslemeden önce fotoğrafta bir yüz olup olmadığını denetler. Yüz yoksa işlem reddedilir, bu sayede modelin rastgele nesnelere cinsiyet ataması engellenir.
3. **Overfitting Engelleme:** Eğitim sırasında `ImageDataGenerator` ile veri çoğaltma (rotasyon, kaydırma, yansıma) ve `EarlyStopping`, `ModelCheckpoint` callback'leri kullanılarak modelin ezberlemesi (overfitting) önlenmiştir.
4. **İnteraktif GUI:** Kullanıcıların teknik bilgiye ihtiyaç duymadan fotoğraf yükleyip anlık tahmin ve güven skoru alabileceği masaüstü arayüzü.

## 📂 Proje Yapısı
* `train.py`: Veri setini yükler, MobileNetV2 mimarisini kurar, modeli eğitir ve en iyi ağırlıkları `.h5` formatında kaydeder.
* `deneme.py`: OpenCV yüz algılama algoritması ile entegre edilmiş, tahmin kararlılığı yüksek ana grafiksel arayüz (GUI).
* `main.py`: Yüz algılama filtresi olmadan, doğrudan yüklenen görüntüyü işleyip tahmin yapan alternatif arayüz.
* `requirements.txt`: Projenin çalışması için gerekli kütüphanelerin listesi.

## ⚙️ Kurulum (Installation)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin. 

1. Repoyu bilgisayarınıza klonlayın:
```bash
git clone [https://github.com/KULLANICI_ADIN/DEPO_ADIN.git](https://github.com/KULLANICI_ADIN/DEPO_ADIN.git)
cd DEPO_ADIN
