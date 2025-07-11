import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import libs.utils.strHelper as strHelper


class DocumentClassifier:
    def __init__(self, model_path: str, bertModelPath :str):
        """
        :param model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.bertModelPath = bertModelPath
        self.sbert_model = SentenceTransformer(bertModelPath)
        self.classifier = None
        self.label_encoder = None

    def load_data(self, json_path: str):
        """
        Loads training data from JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts =[strHelper.normalize_text(item["text"]) for item in data]
        labels = [item["label"] for item in data]
        return texts, labels

    def train(self, json_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Trains the model and saves it to disk.
        """
        texts, labels = self.load_data(json_path)

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)

        # Generate embeddings
        X = self.sbert_model.encode(texts, batch_size=16, show_progress_bar=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Train classifier
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X_train, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Save model
        self.save_model()

    def save_model(self):
        """
        Saves the trained model and label encoder to disk.
        """
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "classifier": self.classifier,
                "label_encoder": self.label_encoder
            }, f)
        print(f"✅ Model saved: {self.model_path}")

    def load_model(self):
        """
        Loads a trained model and label encoder from disk.
        """
        with open(self.model_path, "rb") as f:
            saved = pickle.load(f)
        self.classifier = saved["classifier"]
        self.label_encoder = saved["label_encoder"]
        print(f"✅ Model loaded: {self.model_path}")

    def predict(self, text: str) -> str:
        """
        Classifies a single text input.
        """
        if self.classifier is None or self.label_encoder is None:
            raise ValueError("Model not loaded or trained. Call `load_model()` or `train()` first.")
        embedding = self.sbert_model.encode(strHelper.normalize_text([text]))
        prediction = self.classifier.predict(embedding)
        return self.label_encoder.inverse_transform(prediction)[0]
