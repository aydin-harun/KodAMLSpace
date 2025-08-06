import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re

class LlamaInstructHelper:
    def __init__(self, model_path: str, debugMode : bool = False):
        """
        :param model_path: 4-bit quantized Llama-3.2-3B-Instruct model path
        """
        self.debugMode = debugMode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"llama Insruct 3B Model Device :{self.device}")
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        # Load tokenizer & model once
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": self.device},
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.model.eval()
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     device_map={"": self.device},  # zorla cihaz
        #     torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        #     trust_remote_code=True
        # )
        # self.model.eval()

        for name, param in self.model.named_parameters():
            print(f"{name} → {param.device}")

    def verify(self, big_text: str, checks: list) -> dict:
        """
        Verifies each 'text' in checks against big_text.
        :param big_text: Large text to search in
        :param checks: List of dicts: { "key": str, "text": str }
        :return: dict with results
        """
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
        prompt = (
            f"Document:\n{big_text}\n\n"
            f"Quesition: {question}\n"
            f"Please include only the generated JSON in the assistant's message. Use proper JSON syntax,  sample:```{{\"result\": \"answer\"}}```. "

        )

        response = self._generate(prompt)

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
