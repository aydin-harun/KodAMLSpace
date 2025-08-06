import os
import gc
import cv2
import numpy as np
import joblib
from typing import Optional, Tuple, List, Literal
import base64

class BarcodePresenceClassifier:
    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        min_area: int = 500,
        min_ratio: float = 1.0,
        max_ratio: float = 10.0,
        deskew: bool = True,   # hem eğitimde hem tespitte deskew kullanılsın mı?
        verbose: bool = False
    ):
        self.model = None
        self.model_path = model_path
        self.img_size = img_size
        self.threshold = threshold
        self.min_area = min_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.deskew = deskew
        self.verbose = verbose

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    # ----- EĞİM DÜZELTME (DESKEW) -----
    @staticmethod
    def deskew_min_area_rect(img: np.ndarray) -> np.ndarray:
        # Otomatik olarak eğimi düzelt
        _, threshed = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return img
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    # ----- EĞİTİM (TRAIN) -----
    def train(
        self,
        dataset_dir_barcode: str,
        dataset_dir_nobarcode: str,
        model_type: Literal["rf", "logreg", "svm"] = "rf",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> float:
        X, y = [], []
        label_map = {dataset_dir_barcode: 1, dataset_dir_nobarcode: 0}
        for folder, label in label_map.items():
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                if self.deskew:
                    img = self.deskew_min_area_rect(img)
                img_proc = cv2.resize(img, self.img_size)
                X.append(img_proc.flatten())
                y.append(label)
        X = np.array(X)
        y = np.array(y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000)
        elif model_type == "svm":
            from sklearn.svm import SVC
            clf = SVC(probability=True)
        else:
            raise ValueError("model_type: rf, logreg, svm olmalı")

        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        if self.verbose:
            print(f"Test set accuracy: {acc:.2%}")

        self.model = clf
        if self.model_path:
            joblib.dump(clf, self.model_path)
            if self.verbose:
                print(f"Model kaydedildi: {self.model_path}")

        # Bellek temizliği
        del X, y, X_train, X_test, y_train, y_test
        gc.collect()
        return acc

    # ----- MODEL YÜKLE/UNLOAD -----
    def load_model(self, path: str):
        self.model = joblib.load(path)
        self.model_path = path

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if self.verbose:
                print("Model bellekten temizlendi.")

    # ----- CONTOUR YÖNTEMİ -----
    def _preprocess_page(self, page_gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(page_gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return thresh

    def _find_candidates(self, thresh_img: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _extract_roi(self, page: np.ndarray, contour, deskew: Optional[bool] = None) -> Optional[np.ndarray]:
        if deskew is None:
            deskew = self.deskew
        # Klasik boundingRect ile crop
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return None
        roi = page[y:y+h, x:x+w]
        if deskew:
            roi = self.deskew_min_area_rect(roi)
        roi_resized = cv2.resize(roi, self.img_size)
        return roi_resized

    def getImageData(self, imageBase64_str:str):
        if imageBase64_str.lower().endswith((".tif", ".jpeg", ".jpg")):
            img = cv2.imread(imageBase64_str, cv2.IMREAD_GRAYSCALE)
            return img
        else:
            # --- Base64'ü çöz ---
            img_data = base64.b64decode(imageBase64_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            # --- Görseli OpenCV ile çözümle ---
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return img

    # ----- TESPİT -----
    def detect_on_page(
        self,
        tif_path: str,
        return_locations: bool = False,
        verbose: Optional[bool] = None,
    ) -> bool | Tuple[bool, Optional[List[Tuple[int, int, int, int]]]]:
        if self.model is None:
            raise RuntimeError("Önce model yüklenmeli veya eğitilmeli!")
        if verbose is None:
            verbose = self.verbose

        page = self.getImageData(tif_path)
        if page is None:
            raise FileNotFoundError(f"Sayfa yüklenemedi: {tif_path}")

        thresh = self._preprocess_page(page)
        contours = self._find_candidates(thresh)
        if verbose:
            print(f"🔍 {len(contours)} aday contour bulundu.")

        found = False
        found_locs = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            ratio = w / h if h != 0 else 0

            if area < self.min_area:
                continue
            if ratio < self.min_ratio or ratio > self.max_ratio:
                continue

            roi_resized = self._extract_roi(page, contour, self.deskew)
            if roi_resized is None:
                continue

            roi_flat = roi_resized.flatten().reshape(1, -1)
            prob = self.model.predict_proba(roi_flat)[0][1]
            if prob > self.threshold:
                if verbose:
                    print(f"✅ Barkod bulundu! Rect=({x},{y},{w},{h}) Güven={prob:.2f}")
                    found_locs.append((x, y, w, h))
                found = True

        # Bellek temizliği
        del page, thresh, contours
        gc.collect()

        if return_locations:
            return found, found_locs if found else []
        else:
            return found

# -------------------------
# Kullanım örneği:
# -------------------------
if __name__ == "__main__":
    # Eğitim ve model kaydı (barcode ve nobarcode klasörleriyle)
    clf = BarcodePresenceClassifier(
        img_size=(64, 64),
        deskew=True,   # Hem train hem detect için deskew aktif
        verbose=True
    )
    acc = clf.train(
        dataset_dir_barcode="dataset/barkod",
        dataset_dir_nobarcode="dataset/nobarkod",
        save_path="barcode_model.joblib",
        model_type="rf",
        test_size=0.2
    )
    clf.unload_model()

    # Modeli yükleyip test etme (deskew açık)
    detector = BarcodePresenceClassifier(
        img_size=(64, 64),
        model_path="barcode_model.joblib",
        threshold=0.5,
        deskew=True,  # testte de deskew aktif
        verbose=True
    )
    has_barcode, regions = detector.detect_on_page(
        "example_a4_page.tif",
        return_locations=True,
        deskew=True,
        verbose=True
    )
    print(f"\nBarkod var mı? {'Evet' if has_barcode else 'Hayır'}")
    if has_barcode:
        print("Barkod bölge(ler)i:", regions)

    detector.unload_model()
