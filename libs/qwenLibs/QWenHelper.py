import os
import gc
import json
import time
import base64
import threading
import importlib.util
from io import BytesIO
from typing import Optional, Dict, Any, Union

import torch
from PIL import Image, ImageOps, ImageFile
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class QWenHelper:
    def __init__(
        self,
        model_path: str,
        debug_mode: bool = False,
        cuda_device: str = "cuda:0",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        use_flash_attention_if_available: bool = True,
        auto_load: bool = True,
        require_gpu: bool = True,
    ):
        self.model_path = model_path
        self.debug_mode = debug_mode
        self.cuda_device = cuda_device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.use_flash_attention_if_available = use_flash_attention_if_available
        self.require_gpu = require_gpu

        self.device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

        self.processor: Optional[AutoProcessor] = None
        self.model = None
        self.is_loaded: bool = False
        self._lock = threading.RLock()

        if self.require_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA bulunamadı. Kod GPU zorunlu modda çalıştırılıyor.")

        if self.debug_mode:
            self.debug_cuda_status(prefix="[INIT]")

        if auto_load:
            self.load_model()

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _has_flash_attn(self) -> bool:
        return importlib.util.find_spec("flash_attn") is not None

    def _resolve_cuda_index(self) -> int:
        if not torch.cuda.is_available():
            return -1

        if isinstance(self.cuda_device, str) and self.cuda_device.startswith("cuda:"):
            try:
                idx = int(self.cuda_device.split(":")[1])
                if 0 <= idx < torch.cuda.device_count():
                    return idx
            except Exception:
                pass

        return 0

    def _get_best_attn_implementation(self) -> str:
        if not torch.cuda.is_available():
            return "eager"

        if self.use_flash_attention_if_available and self._has_flash_attn():
            return "flash_attention_2"

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            return "sdpa"

        return "eager"

    def _single_gpu_device_map(self):
        if not torch.cuda.is_available():
            return None
        gpu_index = self._resolve_cuda_index()
        return {"": gpu_index}

    def _assert_model_on_gpu(self) -> None:
        if self.model is None:
            raise RuntimeError("Model yüklü değil.")

        param_device = next(self.model.parameters()).device
        if param_device.type != "cuda":
            raise RuntimeError(f"Model GPU'da değil. Parametre device: {param_device}")

        if hasattr(self.model, "hf_device_map"):
            has_cpu = any(str(v).startswith("cpu") or str(v).startswith("disk")
                          for v in self.model.hf_device_map.values())
            if has_cpu:
                raise RuntimeError(
                    f"Model tamamen GPU'da değil. hf_device_map={self.model.hf_device_map}"
                )

    def _log_model_and_inputs(self, inputs: Dict[str, Any], prefix: str = "[DEBUG]") -> None:
        if not self.debug_mode:
            return

        try:
            model_device = next(self.model.parameters()).device if self.model is not None else "N/A"
            print(f"{prefix} model_device = {model_device}")
        except Exception as ex:
            print(f"{prefix} model_device okunamadı: {ex}")

        if hasattr(self.model, "hf_device_map"):
            print(f"{prefix} hf_device_map = {self.model.hf_device_map}")

        for k, v in inputs.items():
            if hasattr(v, "device"):
                print(f"{prefix} input[{k}] -> {v.device}")

    # ---------------------------------------------------------
    # DEBUG / STATUS
    # ---------------------------------------------------------
    def debug_cuda_status(self, prefix: str = "[CUDA]") -> None:
        print(f"{prefix} torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"{prefix} torch.version.cuda = {torch.version.cuda}")
        print(f"{prefix} device_count = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        print(f"{prefix} selected device = {self.device}")

        if torch.cuda.is_available():
            try:
                idx = self._resolve_cuda_index()
                print(f"{prefix} current_device_index = {idx}")
                print(f"{prefix} device_name = {torch.cuda.get_device_name(idx)}")
                total_vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
                print(f"{prefix} total_vram = {total_vram_gb:.2f} GB")
                self.print_memory_stats(prefix=prefix)
            except Exception as ex:
                print(f"{prefix} gpu status okunamadı: {ex}")

    # ---------------------------------------------------------
    # MODEL LOAD
    # ---------------------------------------------------------
    def load_model(self) -> None:
        with self._lock:
            if self.is_loaded:
                if self.debug_mode:
                    print("[LOAD] Model zaten yüklü.")
                return

            if self.require_gpu and not torch.cuda.is_available():
                raise RuntimeError("GPU zorunlu fakat CUDA aktif değil.")

            if self.debug_mode:
                print(f"[LOAD] Loading processor from: {self.model_path}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            attn_impl = self._get_best_attn_implementation()

            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "attn_implementation": attn_impl,
            }

            if torch.cuda.is_available():
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = self._single_gpu_device_map()
                model_kwargs["dtype"] = torch.float16
            else:
                model_kwargs["dtype"] = torch.float32

            if self.debug_mode:
                print(f"[LOAD] Loading vision model from: {self.model_path}")
                print(f"[LOAD] attn_implementation = {attn_impl}")
                print(f"[LOAD] flash_attn installed = {self._has_flash_attn()}")
                print(f"[LOAD] device_map = {model_kwargs.get('device_map')}")
                self.print_memory_stats(prefix="[LOAD-BEFORE]")

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    **model_kwargs
                )
            except ValueError as ex:
                msg = str(ex)
                if "Some modules are dispatched on the CPU or the disk" in msg:
                    raise RuntimeError(
                        "Model tek GPU VRAM'ine tam sığmıyor. "
                        "Tam GPU kullanım için daha küçük model, daha düşük çözünürlük "
                        "ve daha düşük max_new_tokens kullan."
                    ) from ex
                raise
            except ImportError as ex:
                msg = str(ex).lower()
                if "flashattention2" in msg or "flash_attn" in msg or "flash attention 2" in msg:
                    if self.debug_mode:
                        print("[LOAD] FlashAttention yok. eager fallback deneniyor...")
                    model_kwargs["attn_implementation"] = "eager"
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_path,
                        **model_kwargs
                    )
                else:
                    raise

            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

            self.model.eval()
            self.is_loaded = True

            # Yükleme sonrası model gerçekten GPU'da mı?
            if torch.cuda.is_available():
                self._assert_model_on_gpu()

            if self.debug_mode:
                print("[LOAD] Model loaded successfully.")
                self.print_memory_stats(prefix="[LOAD-AFTER]")

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
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                input_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    text=input_text,
                    images=image,
                    return_tensors="pt"
                )

                model_device = self._get_model_device()
                inputs = {
                    k: v.to(model_device, non_blocking=True) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                self._log_model_and_inputs(inputs, prefix="[EXTRACT]")
                if torch.cuda.is_available():
                    self.print_memory_stats(prefix="[EXTRACT-BEFORE]")

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens or self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "use_cache": True
                }

                if self.do_sample:
                    generation_kwargs["temperature"] = self.temperature

                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                raw_text = self._decode_generated_text(inputs, output_ids)
                parsed = self._try_parse_json(raw_text)

                if self.debug_mode:
                    print(f"[EXTRACT] inference_time = {elapsed:.3f} sn")
                    if torch.cuda.is_available():
                        self.print_memory_stats(prefix="[EXTRACT-AFTER]")

                return {
                    "success": parsed is not None,
                    "raw_text": raw_text,
                    "data": parsed,
                    "error": None if parsed is not None else "JSON parse edilemedi."
                }

            except Exception as ex:
                return {
                    "success": False,
                    "raw_text": None,
                    "data": None,
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

    def ask(
        self,
        image_input: Union[str, bytes],
        question: str,
        max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
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
                            {"type": "image", "image": image},
                            {"type": "text", "text": question}
                        ]
                    }
                ]

                input_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = self.processor(
                    text=input_text,
                    images=image,
                    return_tensors="pt"
                )

                model_device = self._get_model_device()
                inputs = {
                    k: v.to(model_device, non_blocking=True) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                self._log_model_and_inputs(inputs, prefix="[ASK]")
                if torch.cuda.is_available():
                    self.print_memory_stats(prefix="[ASK-BEFORE]")

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens or self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "use_cache": True
                }

                if self.do_sample:
                    generation_kwargs["temperature"] = self.temperature

                start = time.perf_counter()
                with torch.inference_mode():
                    output_ids = self.model.generate(**inputs, **generation_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                raw_text = self._decode_generated_text(inputs, output_ids)

                if self.debug_mode:
                    print(f"[ASK] inference_time = {elapsed:.3f} sn")
                    if torch.cuda.is_available():
                        self.print_memory_stats(prefix="[ASK-AFTER]")

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

    def warmup(self) -> None:
        if not torch.cuda.is_available():
            print("[WARMUP] CUDA yok, warmup atlandı.")
            return

        dummy = Image.new("RGB", (256, 256), (255, 255, 255))
        result = self.ask(dummy.tobytes(), "Bu görsel boş mu?", max_new_tokens=8)
        if self.debug_mode:
            print("[WARMUP] result =", result)

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
        if not text:
            return None

        try:
            return json.loads(text)
        except Exception:
            pass

        import re
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

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
        if isinstance(image_input, bytes):
            img = Image.open(BytesIO(image_input))
            return self._normalize_image(img)

        if isinstance(image_input, Image.Image):
            return self._normalize_image(image_input)

        if not isinstance(image_input, str):
            raise TypeError("image_input str, bytes veya PIL.Image olmalı.")

        if os.path.exists(image_input):
            img = Image.open(image_input)
            return self._normalize_image(img)

        try:
            raw = base64.b64decode(image_input)
            img = Image.open(BytesIO(raw))
            return self._normalize_image(img)
        except Exception as ex:
            raise ValueError(f"Görüntü yüklenemedi. Geçersiz path/base64 olabilir. Detay: {ex}")

    def _normalize_image(self, img: Image.Image) -> Image.Image:
        img = ImageOps.exif_transpose(img)

        # 8 GB GPU için çok büyük görselleri küçült
        max_side = 1280
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / float(max(w, h))
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background

        return img

    def _decode_generated_text(self, inputs, output_ids) -> str:
        if "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_len:]
        else:
            generated_ids = output_ids

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return text.strip()

    # ---------------------------------------------------------
    # MEMORY / DEVICE
    # ---------------------------------------------------------
    def clear_memory(self) -> None:
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
        with self._lock:
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
        with self._lock:
            self.unload()
            if new_model_path:
                self.model_path = new_model_path
            self.load_model()

    def print_memory_stats(self, prefix: str = "[MEMORY]") -> None:
        if not torch.cuda.is_available():
            print(f"{prefix} CUDA yok.")
            return

        try:
            idx = self._resolve_cuda_index()
            allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(idx) / (1024 ** 3)

            print(
                f"{prefix} allocated={allocated:.2f} GB | "
                f"reserved={reserved:.2f} GB | "
                f"max_allocated={max_allocated:.2f} GB"
            )
        except Exception as ex:
            print(f"{prefix} okunamadı: {ex}")

    def _get_model_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model yüklü değil.")
        return next(self.model.parameters()).device

    def _ensure_loaded(self) -> None:
        if not self.is_loaded or self.model is None or self.processor is None:
            self.load_model()