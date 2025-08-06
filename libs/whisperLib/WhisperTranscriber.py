import gc
import tempfile
import base64
import os

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


class WhisperTranscriber:
    """
    Whisper tabanlı ses transkripsiyon sınıfı.
    transformers kütüphanesi ile çalışır.
    Model bir kere yüklenir ve tekrar tekrar kullanılabilir.
    """

    def __init__(self, model_path: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        """
        :param model_path: HuggingFace model adı veya yerel dizin
        :param device: "cpu" veya "cuda"
        :param dtype: torch.float32, torch.float16
        """
        print(f"Loading Whisper model from '{model_path}' …")

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=dtype
        ).to(self.device)

        print("Model loaded successfully.")

    def transcribe_file(self, audio_path: str) -> str:
        """
        Verilen ses dosyasını transkribe eder.
        :param audio_path: Ses dosyasının yolu (.wav, .mp3)
        :return: Transkripsiyon metni
        """
        import librosa

        # Ses dosyasını yükle
        audio, sr = librosa.load(audio_path, sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(self.device)

        # Greedy decoding
        predicted_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        del inputs,input_features, predicted_ids
        gc.collect()
        torch.cuda.empty_cache()
        return transcription

    def __del__(self):
        """
        Model nesnesi silinirken bellek temizliği yapılır.
        """
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        print("Model resources released.")
