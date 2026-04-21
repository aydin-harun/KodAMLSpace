import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq
import json
import re
from PIL import Image
import base64
from io import BytesIO
import  gc

class LlamaInstructHelper:
    def __init__(self, model_path: str, debugMode : bool = False, model_type:int =1):
        """
        :param model_path: 4-bit quantized Llama-3.2-3B-Instruct model path
        """
        self.model_type = model_type
        self._model_path = model_path
        self.debugMode = debugMode
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        print(f"llama Insruct 3B Model Device :{self.device}")

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     device_map={"": self.device},  # zorla cihaz
        #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        #     trust_remote_code=True
        # )
        # self.model.eval()

        # for name, param in self.model.named_parameters():
        #     print(f"{name} → {param.device}")

        # Vision için opsiyonel alanlar (attach_vision ile set edilecek)
        self.load_v_process : bool = False
        self.v_processor = None
        self.v_model = None
        self.cudaPath = "cuda:0"
        self.maxNewTokenCount = 256

        self.load_i_process: bool = False
        self.tokenizer = None
        self.model = None

        if model_type == 1 or model_type == 2:
            self.attach_instruct()
        elif model_type == 3:
            self.attach_vision()

        # --- instruct model bağlama
    def attach_instruct(self):
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        # Load tokenizer & model once
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            device_map={"": self.device},
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.model.eval()
        self.load_i_process = True

    # --- Vision model/processor bağlama (isteğe bağlı) ---
    def attach_vision(self, maxNewTokenCount: int = 256):
        """
        Vision (multimodal) kullanım için gerekli bileşenleri bağlar.
        processor: AutoProcessor/processor (apply_chat_template destekli)
        model: VLM (ör. Llama-3.2-11B-Vision-Instruct uyumlu)
        """
        # Vision bileşenleri ilk defa gerekiyorsa burada yüklenir
        if self.load_v_process:
            return
        # 4-bit quantization config (kaynak dostu)
        v_bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.v_processor = AutoProcessor.from_pretrained(self._model_path, trust_remote_code=True)
        self.v_model = AutoModelForVision2Seq.from_pretrained(
            self._model_path,
            device_map={"": self.device},
            quantization_config=v_bnb_config,
            trust_remote_code=True
        )
        self.v_model.eval()
        self.load_v_process = True

        if self.debugMode:
            print("[Vision Init] Processor & 4-bit model loaded.")

        self.maxNewTokenCount = maxNewTokenCount
        if self.debugMode:
            dev = self.cudaPath if torch.cuda.is_available() else "cpu"
            print(f"[Vision attached] device={dev}, max_tokens={self.maxNewTokenCount}")
        self.load_v_process = True

    def verify(self, big_text: str, checks: list) -> dict:
        """
        Verifies each 'text' in checks against big_text.
        :param big_text: Large text to search in
        :param checks: List of dicts: { "key": str, "text": str }
        :return: dict with results
        """
        # if not self.load_i_process:
        #     self.attach_instruct()
        results = {}
        for item in checks:
            key = item["key"]
            text = item["text"]

            # 👇 Türkçe prompt
            # prompt = (
            #     f"Belge:\n{big_text}\n\n"
            #     f"Soru: Belge açıkça \"{text}\" ifadesini içeriyor mu?\n"
            #     f"Sadece şu formatta json olarak cevap ver: {{\"bulundu\": true/false}}"
            # )
            # prompt = (
            #     f"{big_text} \n\nyukarıdaki metin içinde aşağıdaki değeri var yada yok şeklinde gerçerli bir JSON nesnesi ver. Lütfen asistan mesajı içinde sadece oluşan JSON ver. JSON syntax kullan. Kontrol edeceğin değer \n\n \"{text}\". \n\n"
            # )
            prompt = (
                f"{big_text} \n\n"
                f"In the text above, check if the following value exists or not, and respond with a valid JSON object. "
                f"Please include only the generated JSON in the assistant's message. Use proper JSON syntax,  sample:```{{\"exists\": true}}```. "                
                f"The value to check is:\n\n\"{text}\".\n\n"
            )

            response = self._generate(prompt)
            if self.debugMode:
                print(response)
            # 👇 JSON parse
            bulundu = False
            try:
                match = re.search(r"(?:```(.*?)```|(?<!\w)\{.*?\}(?!\w))", response.replace("sample:```{{\"exists\": true}}",""), re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                    try:
                        parsed =  json.loads(json_str)
                        bulundu = parsed.get("exists", False)
                    except json.JSONDecodeError:
                        bulundu = False
            except Exception:
                bulundu = False
            results[key] = {"text": text, "bulundu": bulundu}
        self.clear_memory()
        return results

    def answer_question(self, big_text: str, question: str) -> dict:
        """
        Ask a question about the big_text and get the answer in JSON.
        :param big_text: Document text
        :param question: Question in Turkish
        :return: dict with answer
        """

        if self.model_type == 1: #Llama-3.2-3B-Instruct
            prompt = (
                f"Document:\n{big_text}\n\n"
                f"Quesition: {question}\n"
                f"Please include only the generated JSON in the assistant's message. Use proper JSON syntax,  sample:```{{\"result\": \"answer\"}}```. "

            )
            response = self._generate(prompt)
        elif self.model_type == 2: # Llama-3.1-8B-Instruct
            prompt = (
                f"Aşağıda bir belge metni ve bu metinle ilgili bir soru verilmiştir.\n\n"
                f"BELGE METNİ:\n{big_text}\n\n"
                f"SORU:\n{question}\n\n"
                f"CEVAP:\n"
                f"Lütfen sadece JSON formatında yanıt ver. Cevabı belirteçler arasında ver. Doğru JSON syntax kullan. "
                f"Örnek:\n```{{\"cevap\": \"...\"}}```"
            )
            response = self._generate(prompt)
        elif self.model_type == 3: #Llama-3.2-11B-Vision-Instruct
            response = self._generate_vision(img_b64=big_text, question=question)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


        # try:
        #     json_start = response.find('{')
        #     json_str = response[json_start:]
        #     parsed = json.loads(json_str)
        #     cevap = parsed.get("cevap", "")
        # except Exception:
        #     cevap = ""
        self.clear_memory()
        return {"soru": question, "cevap": response}

    def clear_memory(self):
        """
        Free GPU memory if any
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, prompt: str) -> str:
        # if not self.load_i_process:
        #     self.attach_instruct()

        inputs = self.tokenizer(prompt, return_tensors="pt",add_special_tokens=False).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    # --- VISION jenerasyonu (image + text, chat template ile) ---
    def _generate_vision(self, img_b64: str, question: str) -> str:
        max_tokens = self.maxNewTokenCount

        # Chat messages: image + text
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question}\n\n"
                                         # f"Sadece geçerli JSON döndür. Ek açıklama yazma. "
                                         # f"Örnek: ```{{\"cevap\": \"...\"}}```"
                }
            ]}
        ]
        input_text = self.v_processor.apply_chat_template(messages, add_generation_prompt=True)

        # Base64 → PIL Image (TIFF güvenli RGB)
        im = Image.open(BytesIO(base64.b64decode(img_b64)))
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        # Inputs oluştur
        inputs = self.v_processor(
            im,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )

        # 👉 model cihazını bul ve inputları aynı cihaza at
        model_device = next(self.v_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        if self.debugMode:
            print(f"[DEBUG] Vision input device: {inputs[next(iter(inputs))].device}")
            print(f"[DEBUG] Vision model device: {model_device}")

        # Generate
        with torch.no_grad():
            output = self.v_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=False,
            )

        # Decode
        ans = self.v_processor.decode(output[0], skip_special_tokens=True)

        # Cleanup
        del im, inputs, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        gc.collect()

        return ans

