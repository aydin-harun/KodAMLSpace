import gc

import easyocr
import torch
import cv2
import numpy as np
import time
import os
import base64
from PIL import Image
import io

class EasyOcrEngine:
    def __init__(self,modelDir):
        os.environ["LRU_CACHE_CAPACITY"] = "1"
        useGpu = torch.cuda.is_available()
        print(f"Ocr İçin CUDA kullanılabilir mi: {useGpu}")
        if useGpu:
            print(f"CUDA cihazı: {torch.cuda.get_device_name(0)}")
        self.reader = easyocr.Reader(['tr'], gpu=useGpu, model_storage_directory=modelDir)
        print(f"Running Device :{self.reader.device}")
        print("Ocr Engine Load Completed....")

    def getImageFromBase64(self, imageBase64):
        img_data = base64.b64decode(str(imageBase64))
        np_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        print("Image Read Completed....")
        return img

    def getImagesFromTiffBase64(self, imageBase64):
        """Tiff dosyasını base64'ten alır, çok sayfalıysa her sayfayı PIL'den OpenCV'ye çevirerek listeler."""
        img_data = base64.b64decode(imageBase64)
        pil_image = Image.open(io.BytesIO(img_data))
        images = []

        try:
            for i in range(pil_image.n_frames):
                pil_image.seek(i)
                rgb_image = pil_image.convert("RGB")
                # PIL to OpenCV
                cv_img = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
                images.append(cv_img)
        except Exception as ex:
            print(f"Tek sayfalı TIFF olarak işleniyor: {ex}")
            rgb_image = pil_image.convert("RGB")
            cv_img = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
            images.append(cv_img)

        print(f"{len(images)} sayfa TIFF dosyası yüklendi.")
        return images

    def OcrImage(self, imageBase64):
        start_time = time.time()
        image = self.getImageFromBase64(imageBase64)
        print(f"Used Device :{self.reader.device}")
        print(self.reader)
        results = self.reader.readtext(image,
                                  paragraph=True,  # Paragraf olarak tanıma için
                                  detail=0,  # Daha detaylı tanıma
                                  contrast_ths=0.1,  # Kontrast eşiği düşük
                                  adjust_contrast=0.5,  # Kontrast ayarı
                                  width_ths=10.0)
        elapsed_time = time.time() - start_time
        print(f"OCR Process Completed in {elapsed_time:.2f} seconds...")
        result = ""
        for text in results:
            result += "\n" + text

        image = None
        del image
        results = None
        del results
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def OcrTiffImage(self, imageBase64):
        start_time = time.time()
        images = self.getImagesFromTiffBase64(imageBase64)
        all_text = []
        for i, image in enumerate(images):
            print(f"OCR processing page {i + 1}...")
            results = self.reader.readtext(image,
                                           paragraph=True,
                                           detail=0,
                                           contrast_ths=0.1,
                                           adjust_contrast=0.5,
                                           width_ths=10.0)
            all_text.append("\n".join(results))
        elapsed_time = time.time() - start_time
        print(f"TIF OCR Process Completed in {elapsed_time:.2f} seconds...")

        del images, results
        gc.collect()
        torch.cuda.empty_cache()
        return "".join(all_text)

    def OcrTiffImageWithDetails(self, imageBase64, useParagraph: bool = True, useWidth_ths: float = 10.0):
        start_time = time.time()
        json_result = []

        try:
            images = self.getImagesFromTiffBase64(imageBase64)

            for i, image in enumerate(images):
                print(f"OCR processing page {i + 1}...")

                results = self.reader.readtext(
                    image,
                    paragraph=useParagraph,
                    detail=1,  # Bu yapı bazen 2, bazen 3 elemanlı dönebilir!
                    contrast_ths=0.1,
                    adjust_contrast=0.5,
                    width_ths=useWidth_ths
                )

                entries = []
                for item in results:
                    if isinstance(item, (list, tuple) ) and len(item) == 3:
                        bbox, text, conf = item
                        bbox = [[int(x), int(y)] for [x, y] in bbox]
                        entry = {
                            "text": text,
                            "bbox": bbox,
                            "confidence": round(conf, 4)
                        }
                        entries.append(entry)
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        bbox, text = item
                        bbox = [[int(x), int(y)] for [x, y] in bbox]
                        entry = {
                            "text": text,
                            "bbox": bbox,
                            "confidence": None  # confidence bilgisi yok
                        }
                        entries.append(entry)
                    else:
                        print(f"⚠️ Beklenmeyen OCR sonucu atlandı: {item}")
                        continue

                json_result.append({
                    "page": i + 1,
                    "entries": entries
                })

            elapsed_time = time.time() - start_time
            print(f"TIF OCR Process Completed in {elapsed_time:.2f} seconds...")

            return json_result

        except Exception as ex:
            print(f"❌ OCR işlemi sırasında hata oluştu: {ex}")
            raise ex
        finally:
            gc.collect()
            torch.cuda.empty_cache()