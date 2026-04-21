import os
import gc
import json
import base64
import threading
from io import BytesIO
from typing import Optional, Dict, Any, Union

import torch
from PIL import Image, ImageOps, ImageFile
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class QWenHelper:
    """
    Tek image -> JSON extraction için optimize edilmiş Qwen Vision helper.

    Amaç:
    - Modeli bir kez yüklemek
    - Aynı instance ile tekrar tekrar inference yapmak
    - 4-bit quantization kullanmak
    - Gereksiz VRAM/CPU bellek baskısını azaltmak
    - Taranmış dokümanlardan alan çıkarımı yapmak

    Uygun modeller:
    - Qwen/Qwen3-VL-30B-A3B-Instruct
    - Qwen/Qwen3-VL-32B-Instruct
    - Qwen/Qwen2.5-VL-32B-Instruct
    """

    def __init__(
        self,
        model_path: str,
        debug_mode: bool = False,
        cuda_device: str = "cuda:0",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        use_flash_attention_if_available: bool = True,
        auto_load: bool = True
    ):
        self.model_path = model_path
        self.debug_mode = debug_mode
        self.cuda_device = cuda_device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_flash_attention_if_available = use_flash_attention_if_available

        self.device = torch.device(
            cuda_device if torch.cuda.is_available() else "cpu"
        )

        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForVision2Seq] = None
        self.is_loaded: bool = False

        # Çok thread'li çağrılarda model.generate çakışmalarını azaltmak için
        self._lock = threading.RLock()

        if self.debug_mode:
            print(f"[INIT] device = {self.device}")
            if torch.cuda.is_available():
                try:
                    print(f"[INIT] cuda device name = {torch.cuda.get_device_name(0)}")
                except Exception:
                    pass

        if auto_load:
            self.load_model()

    # ---------------------------------------------------------
    # MODEL LOAD
    # ---------------------------------------------------------
    def load_model(self) -> None:
        """
        Model ve processor'ı bir kez yükler.
        Tekrar çağrılırsa ikinci kez yüklemez.
        """
        with self._lock:
            if self.is_loaded:
                if self.debug_mode:
                    print("[LOAD] Model zaten yüklü.")
                return

            if self.debug_mode:
                print(f"[LOAD] Loading processor from: {self.model_path}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model_kwargs = {
                "quantization_config": quant_config,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }

            # CUDA varsa device_map otomatik bırakmak genelde daha güvenli.
            # Tek GPU kullanacağın için auto çoğu durumda doğru davranır.
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
                # Destek varsa flash attention açılabilir
                if self.use_flash_attention_if_available:
                    try:
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                    except Exception:
                        pass
            else:
                model_kwargs["device_map"] = {"": "cpu"}

            if self.debug_mode:
                print(f"[LOAD] Loading vision model from: {self.model_path}")

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            self.model.eval()
            self.is_loaded = True

            if self.debug_mode:
                print("[LOAD] Model loaded successfully.")
                self.print_memory_stats()

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------
    def extract_fields(
        self,
        image_input: Union[str, bytes],
        schema: Optional[Dict[str, Any]] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        image_input:
            - base64 string
            - dosya yolu
            - raw bytes

        schema:
            Beklenen JSON şeması. Örn:
            {
                "evrak_no": "",
                "tarih": "",
                "konu": "",
                "gelen_kurum": "",
                "giden_kurum": "",
                "guven_skoru": 0.0
            }

        user_prompt:
            Ek prompt override / detaylandırma

        return:
            {
                "success": True/False,
                "raw_text": "...",
                "data": {...} | None,
                "error": "..."
            }
        """
        with self._lock:
            self._ensure_loaded()

            image = None
            inputs = None
            output_ids = None

            try:
                image = self._load_image(image_input)
                prompt = self._build_prompt(schema=schema, user_prompt=user_prompt)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                input_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    images=image,
                    text=input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )

                model_device = self._get_model_device()
                inputs = {
                    k: v.to(model_device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                if self.debug_mode:
                    print(f"[INFER] model_device = {model_device}")
                    print(f"[INFER] max_new_tokens = {max_new_tokens or self.max_new_tokens}")

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens or self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "use_cache": True
                }

                # do_sample=False iken temperature vermemek daha doğru
                if self.do_sample:
                    generation_kwargs["temperature"] = self.temperature

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )

                raw_text = self.processor.decode(
                    output_ids[0],
                    skip_special_tokens=True
                ).strip()

                parsed = self._try_parse_json(raw_text)

                result = {
                    "success": parsed is not None,
                    "raw_text": raw_text,
                    "data": parsed,
                    "error": None if parsed is not None else "JSON parse edilemedi."
                }

                return result

            except Exception as ex:
                return {
                    "success": False,
                    "raw_text": None,
                    "data": None,
                    "error": str(ex)
                }

            finally:
                # inference sonrası geçici objeleri bırak
                try:
                    del image
                except Exception:
                    pass

                try:
                    del inputs
                except Exception:
                    pass

                try:
                    del output_ids
                except Exception:
                    pass

                self.clear_memory()

    def ask(
        self,
        image_input: Union[str, bytes],
        question: str,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Serbest soru-cevap modu.
        JSON parse zorlamaz.
        """
        with self._lock:
            self._ensure_loaded()

            image = None
            inputs = None
            output_ids = None

            try:
                image = self._load_image(image_input)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question}
                        ]
                    }
                ]

                input_text = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    images=image,
                    text=input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                )

                model_device = self._get_model_device()
                inputs = {
                    k: v.to(model_device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens or self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "use_cache": True
                }

                if self.do_sample:
                    generation_kwargs["temperature"] = self.temperature

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )

                raw_text = self.processor.decode(
                    output_ids[0],
                    skip_special_tokens=True
                ).strip()

                return {
                    "success": True,
                    "answer": raw_text,
                    "error": None
                }

            except Exception as ex:
                return {
                    "success": False,
                    "answer": None,
                    "error": str(ex)
                }

            finally:
                try:
                    del image
                except Exception:
                    pass

                try:
                    del inputs
                except Exception:
                    pass

                try:
                    del output_ids
                except Exception:
                    pass

                self.clear_memory()

    # ---------------------------------------------------------
    # PROMPT / JSON
    # ---------------------------------------------------------
    def _build_prompt(
        self,
        schema: Optional[Dict[str, Any]] = None,
        user_prompt: Optional[str] = None
    ) -> str:
        if schema is None:
            schema = {
                "evrak_no": "",
                "tarih": "",
                "konu": "",
                "gelen_kurum": "",
                "giden_kurum": "",
                "sayilar": [],
                "guven_skoru": 0.0
            }

        schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

        base_prompt = f"""
Aşağıdaki taranmış belge görüntüsünü incele.

Görev:
- Belgeden alan çıkarımı yap.
- Sadece geçerli JSON döndür.
- JSON dışında hiçbir açıklama yazma.
- Okunamayan veya bulunamayan alanları null yap.
- Tahmin etme, sadece görüntüde açıkça görülen bilgiyi yaz.
- Birden fazla aday varsa en güçlü adayı ana alana yaz, diğerlerini "alternatives" altında listele.
- Tarihi mümkünse standart biçimde yaz.
- Kurum adlarını mümkün olduğunca tam çıkar.
- Belgedeki önemli sayı, tarih, konu, gelen kurum, giden kurum gibi alanları bul.

Dönüş şeması:
{schema_json}
""".strip()

        if user_prompt:
            return f"{base_prompt}\n\nEk talimat:\n{user_prompt}"

        return base_prompt

    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Model bazen JSON dışında metin dönebilir.
        Önce direkt parse etmeyi dener, olmazsa ilk JSON bloğunu ayıklar.
        """
        if not text:
            return None

        # 1) Direkt parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2) ```json ... ``` bloğu
        import re
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        # 3) İlk { ... } bloğu
        bracket = re.search(r"(\{.*\})", text, re.DOTALL)
        if bracket:
            candidate = bracket.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return None

    # ---------------------------------------------------------
    # IMAGE HELPERS
    # ---------------------------------------------------------
    def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """
        image_input:
        - dosya yolu
        - base64 string
        - bytes
        """
        if isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
            return self._normalize_image(img)

        if not isinstance(image_input, str):
            raise TypeError("image_input str veya bytes olmalı.")

        # Dosya yolu ise
        if os.path.exists(image_input):
            img = Image.open(image_input)
            return self._normalize_image(img)

        # Base64 ise
        try:
            raw = base64.b64decode(image_input)
            img = Image.open(BytesIO(raw))
            return self._normalize_image(img)
        except Exception as ex:
            raise ValueError(f"Görüntü yüklenemedi. Geçersiz path/base64 olabilir. Detay: {ex}")

    def _normalize_image(self, img: Image.Image) -> Image.Image:
        """
        Taranmış belgeler için güvenli normalize.
        - EXIF orientation düzeltir
        - RGB'a çevirir
        """
        img = ImageOps.exif_transpose(img)

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background

        return img

    # ---------------------------------------------------------
    # MEMORY / DEVICE
    # ---------------------------------------------------------
    def clear_memory(self) -> None:
        """
        Inference sonrası cache temizliği.
        Modeli RAM/VRAM'den atmaz; sadece geçici allocation baskısını azaltır.
        """
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def unload(self) -> None:
        """
        Modeli tamamen bellekten at.
        Uygulama kapanırken veya model değiştirirken kullan.
        """
        with self._lock:
            if self.debug_mode:
                print("[UNLOAD] Releasing model and processor...")

            try:
                del self.model
            except Exception:
                pass

            try:
                del self.processor
            except Exception:
                pass

            self.model = None
            self.processor = None
            self.is_loaded = False

            self.clear_memory()

    def reload_model(self, new_model_path: Optional[str] = None) -> None:
        """
        Model değiştirmek veya temiz reload yapmak için.
        """
        with self._lock:
            self.unload()

            if new_model_path:
                self.model_path = new_model_path

            self.load_model()

    def print_memory_stats(self) -> None:
        if not torch.cuda.is_available():
            print("[MEMORY] CUDA yok.")
            return

        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

            print(
                f"[MEMORY] allocated={allocated:.2f} GB | "
                f"reserved={reserved:.2f} GB | "
                f"max_allocated={max_allocated:.2f} GB"
            )
        except Exception as ex:
            print(f"[MEMORY] okunamadı: {ex}")

    def _get_model_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model yüklü değil.")
        return next(self.model.parameters()).device

    def _ensure_loaded(self) -> None:
        if not self.is_loaded or self.model is None or self.processor is None:
            self.load_model()