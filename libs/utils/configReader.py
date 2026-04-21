from dataclasses import dataclass
import json


@dataclass
class EmptyPageModelConfig:
    rfModelPath:str
    xgbModelPath:str
    lgbmModelPath:str
    catboostModelPath:str
    cnnModelPath: str
    isRfModelUsing:bool
    isXgbModelUsing:bool
    isLgbmModelUsing: bool
    isCatboostModelUsing: bool
    isCNNModelUsing: bool


@dataclass
class AppConfig:
    apiPort: int
    emptyPageSizeWidth:int
    emptyPageSizeHeight: int
    emptyPageModelPath:str
    dbPath:str
    ocrModelDir:str
    gensimModelDir:str
    spacyModelDir:str
    bertModelDir:str
    bertModelPath:str
    useBertModelOperation:bool
    useGensimModelOperation:bool
    useEasyOcrOperation:bool
    useEmptyPageOperation:bool
    useRapifFuzzOperation:bool
    debugMode:bool
    useLlama:bool
    llamaModelType:int
    llamaModelPath:str
    useWhisperTranscribeOperation:bool
    whisperTranscribeModelPath:str
    useBarcodeDetectOperation:bool
    barcodeDetectModelPath:str
    emptyPageModelConfig:EmptyPageModelConfig
    useQWen: bool
    qwenModelType: int
    qwenModelPath: str

    @classmethod
    def from_dict(cls, data: dict):
        # nested config nesnesini oluştur
        data['emptyPageModelConfig'] = EmptyPageModelConfig(**data['emptyPageModelConfig'])
        return cls(**data)

def loadConfig():
    with open("appConfig.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # appConfig = AppConfig(**data)
    appConfig = AppConfig.from_dict(data)
    return appConfig