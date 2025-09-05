import os
import cv2
import joblib
import base64
import numpy as np
from skimage.measure import shannon_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from enum import Enum
from libs.utils.configReader import EmptyPageModelConfig


class ModelType(Enum):
    rf="rf"
    xgb="xgb"
    lgbm="lgbm"
    catboost="catboost"


class EmptyPageClassifier:
    def __init__(self, emptyPageModelConfig : EmptyPageModelConfig):
        """
        :param model_path: Modelin kayıt yolu
        """

        if emptyPageModelConfig is None:
            self.rfModelUsable: bool = False
            self.xgbModelUsable: bool = False
            self.lgbmModelUsable: bool = False
            self.catboostModelUsable: bool = False
        self.emptyPageModelConfig = emptyPageModelConfig
        self.rfModelUsable : bool = (
            self.getModelUsable(self.emptyPageModelConfig.isRfModelUsing, self.emptyPageModelConfig.rfModelPath))
        self.xgbModelUsable: bool = (
            self.getModelUsable(self.emptyPageModelConfig.isXgbModelUsing, self.emptyPageModelConfig.xgbModelPath))
        self.lgbmModelUsable: bool = (
            self.getModelUsable(self.emptyPageModelConfig.isLgbmModelUsing, self.emptyPageModelConfig.lgbmModelPath))
        self.catboostModelUsable: bool = (
            self.getModelUsable(self.emptyPageModelConfig.isCatboostModelUsing, self.emptyPageModelConfig.catboostModelPath))

        if self.rfModelUsable:
            self.rfModel = joblib.load(self.emptyPageModelConfig.rfModelPath)
        if self.xgbModelUsable:
            self.xgbModel = joblib.load(self.emptyPageModelConfig.xgbModelPath)
        if self.lgbmModelUsable:
            self.lgbmModel = joblib.load(self.emptyPageModelConfig.lgbmModelPath)
        if self.catboostModelUsable:
            self.catboostModel = joblib.load(self.emptyPageModelConfig.catboostModelPath)

    def getModelUsable(self, modelUsable:bool, modelPath:str):
        if modelUsable == True:
            if len(modelPath)>0:
                if os.path.exists(modelPath):
                    return True
        return False

    # --- Özellik çıkarımı ---
    def _extract_features(self, img) -> list:
        try:
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

            mean_val = np.mean(resized)
            std_val = np.std(resized)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)
            non_white_ratio = np.sum(resized < 245) / resized.size

            return [
                ink_ratio, entropy_val, edge_density, connected_components,
                mean_val, std_val, contour_count, non_white_ratio
            ]
        except Exception as e:
            print(f"[!] Özellik çıkarım hatası: {e}")
            return [0.0] * 8

    def extract_features_from_path(self, image_path: str) -> list:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return self._extract_features(img)

    def extract_features_from_base64(self, b64_str: str) -> list:
        try:
            img_data = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Base64 çözümleme başarısız.")
            return self._extract_features(img)
        except Exception as e:
            print(f"[!] Base64 görsel hata: {e}")
            return [0.0] * 8

    # --- Model Eğitimi ---
    def train_rf(self, empty_dir: str, filled_dir: str):
        return self._train_model(empty_dir, filled_dir, model_type='rf')

    def train_xgb(self, empty_dir: str, filled_dir: str):
        return self._train_model(empty_dir, filled_dir, model_type='xgb')

    def train_lgbm(self, empty_dir: str, filled_dir: str):
        return self._train_model(empty_dir, filled_dir, model_type='lgbm')

    def train_catboost(self, empty_dir: str, filled_dir: str):
        return self._train_model(empty_dir, filled_dir, model_type='catboost')

    def _train_model(self, empty_dir: str, filled_dir: str, model_type: str):
        X, y = [], []

        #model_type: 'rf', 'xgb', 'lgbm', 'catboost'
        files = []
        for folder, label in [(empty_dir, 0), (filled_dir, 1)]:
            for file in os.listdir(folder):
                if file.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
                    files.append((os.path.join(folder, file), label))

        total = len(files)
        print(f"📥 Toplam eğitim verisi: {total} sayfa")

        for idx, (path, label) in enumerate(files):
            X.append(self.extract_features_from_path(path))
            y.append(label)
            if (idx + 1) % 50 == 0 or (idx + 1) == total:
                print(f"    > %{((idx + 1) / total) * 100:.1f} tamamlandı ({idx + 1}/{total})")

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
        modelSavePath:str= ""
        if model_type == 'xgb':
            modelSavePath = self.emptyPageModelConfig.xgbModelPath
            clf = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        elif model_type == 'lgbm':
            modelSavePath = self.emptyPageModelConfig.lgbmModelPath
            clf = LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                class_weight='balanced',
                learning_rate=0.05,
                random_state=42
            )
        elif model_type == 'catboost':
            modelSavePath = self.emptyPageModelConfig.catboostModelPath
            clf = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                verbose=100,
                random_state=42
            )
        else:  # default to RandomForest
            modelSavePath = self.emptyPageModelConfig.rfModelPath
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                class_weight='balanced',
                random_state=42
            )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("\n📊 Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred, target_names=["Boş", "Dolu"]))
        print("F1 (macro):", f1_score(y_test, y_pred, average="macro"))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        os.makedirs(os.path.dirname(modelSavePath), exist_ok=True)
        joblib.dump(clf, modelSavePath)
        if model_type == 'xgb':
            self.xgbModel = clf
            self.xgbModelUsable = True
        elif model_type == 'lgbm':
            self.lgbmModel = clf
            self.lgbmModelUsable = True
        elif model_type == 'catboost':
            self.catboostModel = clf
            self.catboostModelUsable = True
        else:  # default to RandomForest
            self.rfModel = clf
            self.rfModelUsable = True

        print(f"✅ Model kaydedildi: {modelSavePath}")
        return modelSavePath

    # --- Tahmin ---
    def predict(self, image_input: str, input_is_path: bool = True):
        features = self.extract_features_from_path(image_input) if input_is_path else self.extract_features_from_base64(image_input)

        if all([f == 0.0 for f in features]):
            return "Ölçüm Yapılamadı", 0.0, 0.0

        remark:str = "Kararsız"
        clas:str = ""
        pred:float = 0.0
        prob:float = 0.0
        results =[]
        if self.rfModelUsable:
            pred = self.rfModel.predict([features])[0]
            prob = self.rfModel.predict_proba([features])[0][1] * 100
            remark, clas = self.getDescriptions(prob, pred)
            print(f"➜ Tahmin: {clas} (%{prob:.2f} güven) → [{remark}]")
            results.append({"pred" : int(pred), "prob" : float(prob), "clas" : clas, "remark" : remark, "lib_type":"rf"})
        if self.xgbModelUsable:
            pred = self.xgbModel.predict([features])[0]
            prob = self.xgbModel.predict_proba([features])[0][1] * 100
            remark, clas = self.getDescriptions(prob, pred)
            print(f"➜ Tahmin: {clas} (%{prob:.2f} güven) → [{remark}]")
            results.append({"pred" : int(pred), "prob" : float(prob), "clas" : clas, "remark" : remark, "lib_type":"xgb"})
        if self.lgbmModelUsable:
            pred = self.lgbmModel.predict([features])[0]
            prob = self.lgbmModel.predict_proba([features])[0][1] * 100
            remark, clas = self.getDescriptions(prob, pred)
            print(f"➜ Tahmin: {clas} (%{prob:.2f} güven) → [{remark}]")
            results.append({"pred" : int(pred), "prob" : float(prob), "clas" : clas, "remark" : remark, "lib_type":"lgbm"})
        if self.catboostModelUsable:
            pred = self.catboostModel.predict([features])[0]
            prob = self.catboostModel.predict_proba([features])[0][1] * 100
            remark, clas = self.getDescriptions(prob, pred)
            print(f"➜ Tahmin: {clas} (%{prob:.2f} güven) → [{remark}]")
            results.append({"pred" : int(pred), "prob" : float(prob), "clas" : clas, "remark" : remark, "lib_type":"catboost"})
        return results

    def getDescriptions(self,prob, pred):
        remark = "Kararsız"
        if prob < 60:
            remark = "Kararsız"
        elif prob < 85:
            remark = "Orta Güven"
        else:
            remark = "Yüksek Güven"
        clas = "Dolu" if pred == 1 else "Boş"
        return  remark, clas