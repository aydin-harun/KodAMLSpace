import os
import gc
import cv2
import numpy as np
import joblib
from typing import Optional, Tuple, List, Literal
import base64
import libs.utils.fileHelper as fileHelper


class BarcodePresenceClassifier:
    def __init__(
        self,
        img_size: Tuple[int, int] = (128, 128),
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        min_area: int = 0,          # 0 => sadece çok ufak gürültüyü ele, asıl clustering’e bırak
        min_ratio: float = 0.7,     # kare: en/boy oranı ~1
        max_ratio: float = 1.3,
        deskew: bool = True,        # hem eğitimde hem tespitte deskew kullanılsın mı?
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
        _, threshed = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
        deskewed = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    # ----- EĞİTİM (TRAIN) -----
    def train(
            self,
            dataset_dir_barcode: str,
            dataset_dir_nobarcode: str,
            model_type: Literal["rf", "logreg", "svm"] = "rf",
            test_size: float = 0.2,
            random_state: int = 42
    ) -> str:
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

        # ✅ Model güvenli protokol ile kaydediliyor
        if self.model_path:
            fileHelper.delFileIfExists(self.model_path)
            fileHelper.createDirIfExists(os.path.dirname(self.model_path))
            joblib.dump(clf, self.model_path, compress=3, protocol=4)
            if self.verbose:
                print(f"Model kaydedildi: {self.model_path} (protocol=4)")

        # Bellek temizliği
        del X, y, X_train, X_test, y_train, y_test
        gc.collect()
        return self.model_path

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

    # ----- SAYFA ÖN İŞLEME -----
    def _preprocess_page(self, page_gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(page_gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        # Çok agresif morph uygulamıyoruz; kare modüller kaybolmasın
        return thresh

    # ----- KARE SHAPE TESPİTİ -----
    def _find_square_boxes(self, thresh_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Threshold görüntü üzerinden kareye benzeyen contour'ların
        bounding box'larını döndürür.
        """
        contours, _ = cv2.findContours(
            thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: List[Tuple[int, int, int, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Çok küçük gürültüleri at
            if area < 50:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

            # 4 köşe + konveks olmalı
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue

            ratio = w / h if h != 0 else 0.0
            if ratio < self.min_ratio or ratio > self.max_ratio:
                continue

            # min_area paramı >0 ise onu da uygula (isteğe bağlı)
            if self.min_area and self.min_area > 0 and (w * h) < self.min_area:
                continue

            boxes.append((x, y, w, h))

        return boxes

    # ----- KARELERİ CLUSTER'LA (QR'ın tamamını yakalamak için) -----
    @staticmethod
    def _cluster_boxes(
        boxes: List[Tuple[int, int, int, int]],
        dist_factor: float = 4.0,
        min_cluster_size: int = 2
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        Kare kutuları, merkezleri birbirine yakın olan gruplara ayır.
        Her grup muhtemelen aynı QR'ın parçası (özellikle finder pattern'ler).
        """
        clusters: List[List[Tuple[int, int, int, int]]] = []
        used = [False] * len(boxes)

        centers = []
        sizes = []
        for (x, y, w, h) in boxes:
            cx = x + w / 2.0
            cy = y + h / 2.0
            centers.append((cx, cy))
            sizes.append(max(w, h))

        for i in range(len(boxes)):
            if used[i]:
                continue
            # yeni cluster başlat
            cluster = [boxes[i]]
            used[i] = True
            ci = centers[i]
            si = sizes[i]

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                cj = centers[j]
                sj = sizes[j]
                # merkezler arası mesafe
                dx = ci[0] - cj[0]
                dy = ci[1] - cj[1]
                dist = (dx * dx + dy * dy) ** 0.5
                # referans mesafe: kutu boyutlarının max'ı * dist_factor
                ref = max(si, sj) * dist_factor
                if dist <= ref:
                    used[j] = True
                    cluster.append(boxes[j])

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _extract_roi_from_cluster(
        self,
        page: np.ndarray,
        cluster: List[Tuple[int, int, int, int]],
        deskew: Optional[bool] = None,
        margin_factor: float = 0.4
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Bir cluster içindeki kare kutuların tamamını kapsayan büyük ROI çıkar.
        margin_factor kadar ekstra kenar ekler.
        """
        if deskew is None:
            deskew = self.deskew

        h_page, w_page = page.shape[:2]

        xs = [x for (x, _, w, _) in cluster] + [x + w for (x, _, w, _) in cluster]
        ys = [y for (_, y, _, h) in cluster] + [y + h for (_, y, _, h) in cluster]

        x_min = max(0, min(xs))
        x_max = min(w_page, max(xs))
        y_min = max(0, min(ys))
        y_max = min(h_page, max(ys))

        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            return None, (0, 0, 0, 0)

        # Kenarlara margin ekle (QR biraz daha tam gelsin)
        margin = int(max(width, height) * margin_factor)
        x0 = max(0, x_min - margin)
        y0 = max(0, y_min - margin)
        x1 = min(w_page, x_max + margin)
        y1 = min(h_page, y_max + margin)

        roi = page[y0:y1, x0:x1]
        if roi is None or roi.size == 0:
            return None, (0, 0, 0, 0)

        if deskew:
            roi = self.deskew_min_area_rect(roi)

        roi_resized = cv2.resize(roi, self.img_size)
        return roi_resized, (x0, y0, x1 - x0, y1 - y0)

    def getImageData(self, imageBase64_str: str):
        if imageBase64_str.lower().endswith((".tif", ".tiff", ".jpeg", ".jpg", ".png")):
            img = cv2.imread(imageBase64_str, cv2.IMREAD_GRAYSCALE)
            return img
        else:
            img_data = base64.b64decode(imageBase64_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return img

    # ----- TESPİT -----
    def detect_on_page(
            self,
            tif_path: str,
            deskew: Optional[bool] = None,
            return_locations: bool = False,
            verbose: Optional[bool] = None,
            save_candidates: bool = False,      # debug için
            candidate_dir: str = "temp_candidates"  # debug klasörü
    ) -> bool | Tuple[bool, Optional[List[Tuple[int, int, int, int]]]]:
        if self.model is None:
            raise RuntimeError("Önce model yüklenmeli veya eğitilmeli!")
        if verbose is None:
            verbose = self.verbose
        if deskew is None:
            deskew = self.deskew

        if save_candidates:
            os.makedirs(candidate_dir, exist_ok=True)

        page = self.getImageData(tif_path)
        if page is None:
            raise FileNotFoundError(f"Sayfa yüklenemedi: {tif_path}")

        thresh = self._preprocess_page(page)

        # 1) kare kutuları bul
        boxes = self._find_square_boxes(thresh)
        if verbose:
            print(f"🔍 {len(boxes)} kare candidate bulundu.")

        # 2) kareleri cluster’la (aynı QR’a ait olanları grupla)
        clusters = self._cluster_boxes(
            boxes,
            dist_factor=4.0,     # kare boyutunun 4 katı mesafeye kadar aynı cluster
            min_cluster_size=2   # en az 2 kare beraber olsun
        )
        if verbose:
            print(f"🔗 {len(clusters)} cluster bulundu.")

        found = False
        found_locs: List[Tuple[int, int, int, int]] = []

        # Eğer hiç cluster yoksa, fallback: tek karelerden de deneyebilirsin
        if not clusters:
            clusters = [[b] for b in boxes]

        for idx, cluster in enumerate(clusters):
            roi_resized, rect = self._extract_roi_from_cluster(
                page,
                cluster,
                deskew=deskew,
                margin_factor=0.4
            )
            if roi_resized is None:
                continue

            x0, y0, w, h = rect

            # DEBUG: candidate kaydet
            if save_candidates:
                out_path = os.path.join(
                    candidate_dir,
                    f"cand_cluster_{idx}_x{x0}_y{y0}_w{w}_h{h}.jpg"
                )
                cv2.imwrite(out_path, roi_resized)
                if verbose:
                    print(f"📁 Cluster aday kaydedildi: {out_path} (boxes_in_cluster={len(cluster)})")

            roi_flat = roi_resized.flatten().reshape(1, -1)
            prob = self.model.predict_proba(roi_flat)[0][1]
            if prob > self.threshold:
                if verbose:
                    print(f"✅ Barkod bulundu! ClusterRect=({x0},{y0},{w},{h}) Güven={prob:.2f}")
                found_locs.append((x0, y0, w, h))
                found = True
                # Sayfada tek QR bekliyorsan break açılabilir:
                # break

        del page, thresh
        gc.collect()

        if return_locations:
            return found, found_locs if found else []
        else:
            return found
