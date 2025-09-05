import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import gc
import  libs.utils.fileHelper as fileHelper

# --- Özellik çıkarım fonksiyonu ---
import cv2
import numpy as np
import base64
from skimage.measure import shannon_entropy  # veya alternatif entropy fonksiyonun

model = None
model_path = ""

# def simple_entropy(image):
#     """Grayscale görüntü için Shannon Entropy hesapla"""
#     hist = cv2.calcHist([image], [0], None, [256], [0,256])
#     hist = hist.ravel() / hist.sum()
#     hist = hist[hist > 0]
#     return -np.sum(hist * np.log2(hist))

def loadModel(modelPath):
    global model, model_path
    model_path = modelPath
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise e

def extract_enhanced_features_from_base64(imageBase64_str):
    img = resized = binary = edges = np_arr = img_data = None  # önceden tanımla
    try:
        # --- Base64'ü çöz ---
        img_data = base64.b64decode(imageBase64_str)
        np_arr = np.frombuffer(img_data, np.uint8)

        # --- Görseli OpenCV ile çözümle ---
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("❌ Görsel base64 verisi çözülemedi.")

        resized = cv2.resize(img, (800, 1000))
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        total_pixels = binary.size
        black_pixels = total_pixels - cv2.countNonZero(binary)
        ink_ratio = black_pixels / total_pixels

        entropy_val = shannon_entropy(binary)

        edges = cv2.Canny(binary, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        num_labels, _ = cv2.connectedComponents(255 - binary)
        connected_components = num_labels

        return [ink_ratio, entropy_val, edge_density, connected_components]

    except Exception as e:
        print(f"[HATA] Özellik çıkarımı sırasında bir hata oluştu: {str(e)}")
        return [0.0, 0.0, 0.0, 0]  # hata durumunda nötr değerler dönebilir

    finally:
        # --- Bellek temizliği ---
        del img, resized, binary, edges, np_arr, img_data, imageBase64_str
        gc.collect()  # zorunlu bellek temizliği

def extract_enhanced_features_from_path(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img, (800, 1000))  # normalize boyut

        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        total_pixels = binary.size
        black_pixels = total_pixels - cv2.countNonZero(binary)
        ink_ratio = black_pixels / total_pixels

        entropy_val = shannon_entropy(binary)

        edges = cv2.Canny(binary, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        num_labels, _ = cv2.connectedComponents(255 - binary)  # tersle -> siyahlar 255 olsun
        connected_components = num_labels

        return [ink_ratio, entropy_val, edge_density, connected_components]
    except Exception as e:
        print(f"[HATA] Özellik çıkarımı sırasında bir hata oluştu: {str(e)}")
        return [0.0, 0.0, 0.0, 0]  # hata durumunda nötr değerler dönebilir
    finally:
        # --- Bellek temizliği ---
        del img, resized, binary, edges
        gc.collect()  # zorunlu bellek temizliği

# --- Eğitim veri dizinleri ---
def trainModel(empty_dir, filled_dir):
    global model_path
    # empty_dir = "train_pages/empty"
    # filled_dir = "train_pages/filled"
    X, y = [], []

    # Boş sayfaları işle
    for file in os.listdir(empty_dir):
        if file.lower().endswith(('.tif', '.tiff', '.jpg', '.png')):
            path = os.path.join(empty_dir, file)
            X.append(extract_enhanced_features_from_path(path))
            y.append(0)

    # Dolu sayfaları işle
    for file in os.listdir(filled_dir):
        if file.lower().endswith(('.tif', '.tiff', '.jpg', '.png')):
            path = os.path.join(filled_dir, file)
            X.append(extract_enhanced_features_from_path(path))
            y.append(1)

    # --- Model eğitimi ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # --- Değerlendirme ---
    y_pred = clf.predict(X_test)
    print("\n📊 Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=["Boş", "Dolu"]))

    #fileName = "../../mlModels/detectEmptyPageModel/page_classifier_enhanced.pkl"
    fileHelper.delFileIfExists(model_path)
    fileHelper.createDirIfExists(os.path.dirname(model_path))
    # --- Modeli kaydet ---
    joblib.dump(clf, model_path)
    print("✅ Model 'page_classifier_enhanced.pkl' olarak kaydedildi.\n")
    return model_path

# --- Örnek tahmin (test amaçlı) ---
def predict_page(imageBase64_str, model_path="../../mlModels/detectEmptyPageModel/page_classifier_enhanced.pkl"):
    global model
    try:
        if model is None:
            raise Exception("Model Dosyası Yok")
        # model = joblib.load(model_path)
        feat = extract_enhanced_features_from_path(imageBase64_str)  if imageBase64_str.lower().endswith((".tif", ".jpeg", ".jpg")) else extract_enhanced_features_from_base64(imageBase64_str)
        if feat[0] == 0.0 and feat[1] ==0.0 and feat[2]==0.0 and feat[3]==0:
            print("🛑🛑🛑 Ölçüm Yapılamadı")
            return ("Ölçüm Yapılamadı", 0.0,0.0)
        pred = model.predict([feat])[0]
        prob = model.predict_proba([feat])[0][1]
        prob = prob*100
        print(f"{pred},{prob:.2f}")  # stdout: örn: 1,93.27
        print(f" ➜ Tahmin: {'Dolu' if pred == 1 else 'Boş'} (%{prob:.2f} güven)")
        return ('Dolu' if pred == 1 else 'Boş', pred, prob)
    except Exception as e:
        print(f"Hata : {str(e)}")
        return ("Ölçüm Yapılamadı", 0.0, 0.0)

# trainModel("train_pages/empty","train_pages/filled")
#
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("0,0.00")  # hata gibi davran
#         sys.exit(1)
#     predict_page(sys.argv[1])
#
# Test et (isteğe bağlı dosya yolları girin)
# predict_page("test_image.tif")
#
#
# # print( datetime.datetime.now())
# # # Kullanım örneği
# label, pred, prop = predict_page("../../sampleImages/1.tif")
# print(f"Durum : {label} Pred : {pred} Prop : {prop}")
# # # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
# #
# label, pred, prop = predict_page("../../sampleImages/2.tif")
# print(f"Durum : {label} Pred : {pred} Prop : {prop}")
# # # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
# #
# label, pred, prop = predict_page("../../sampleImages/3.tif")
# print(f"Durum : {label} Pred : {pred} Prop : {prop}")
# # # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
# #
# label, pred, prop = predict_page("../../sampleImages/4.tif")
# print(f"Durum : {label} Pred : {pred} Prop : {prop}")
# # # print("Sayfa durumu:", "Dolu" if result == 1 else "Boş", f"(%{prob*100:.2f} güven)")
# #
# label, pred, prop = predict_page("../../sampleImages/5.tif")
# print(f"Durum : {label} Pred : {pred} Prop : {prop}")
# #
# label, pred, prop = predict_page("../../sampleImages/6.tif")
# print(f"Durum : {label} Pred : {pred} Prop : {prop}")
# #
# # print( datetime.datetime.now())