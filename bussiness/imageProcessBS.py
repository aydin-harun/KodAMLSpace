import torch

from libs.sentenceTransformerLib.BertDocumentClassifier import BertDocumentClassifier
from libs.llamaLibs.LlamaInstructHelper import LlamaInstructHelper
from libs.utils import strHelper
from libs.utils.configReader import AppConfig, loadConfig
import dataAccessLayer.SqliteDataOperations as dataOperation
import libs.emptyPageDetect.EmptyPageDetectHelperV2 as emptyPageDetectHelperV2
import libs.utils.fileHelper as fileHelper
import  os
import libs.ocrPage.easyOcrEngine as easyOcrEngine
import base64
import libs.gensimLib.gensimOperations as go
import libs.rapidfuzzLib.rapidfuzzOperations as rf
import  logging
from libs.utils.logger_config import setup_logging
from libs.whisperLib.WhisperTranscriber import WhisperTranscriber
from libs.barcodeDetect.BarcodePresenceClassifier import BarcodePresenceClassifier
from libs.emptyPageDetect.CnnEmptyPageDetectHelper import CnnEmptyPageDetector
from libs.qwenLibs.QWenHelper import QWenHelper
import json

# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

setup_logging(log_level="INFO")

logger = logging.getLogger(__name__)

appConfig:AppConfig =None
eOcrEngine: easyOcrEngine.EasyOcrEngine = None
bertDocClassifier :BertDocumentClassifier = None
bertModelCount = 0
emptyPageModelCount = 0
gensimModelCount = 0
llama3BInstructHlp : LlamaInstructHelper = None
whisperTranscriber :WhisperTranscriber = None
barcodeDetectModelCount = 0
brcPresenceClassifier : BarcodePresenceClassifier = None
emptyPageClassifier : emptyPageDetectHelperV2.EmptyPageClassifier = None
emptyPageClassifierCNN : CnnEmptyPageDetector = None
qwenHlp : QWenHelper = None

#region initialize
def loadAppConfig():
    global appConfig
    try:
        appConfig =  loadConfig()
        dataOperation.dbPath = appConfig.dbPath
        loadModels()
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        raise

def loadModels():
    global appConfig, eOcrEngine, bertDocClassifier, emptyPageModelCount, gensimModelCount, \
        bertModelCount, llama3BInstructHlp, whisperTranscriber, \
        barcodeDetectModelPath, barcodeDetectModelCount, brcPresenceClassifier, \
        emptyPageClassifier, emptyPageClassifierCNN,\
        qwenHlp
    try:
        if appConfig.useEmptyPageOperation:
            rows = getEmptyPageModels()
            for row in rows:
                if row[3] == emptyPageDetectHelperV2.ModelType.rf.value:
                    fileHelper.delFileIfExists(appConfig.emptyPageModelConfig.rfModelPath)
                    fileHelper.createDirIfExists(os.path.dirname(appConfig.emptyPageModelConfig.rfModelPath))
                    fileHelper.writeFile(os.path.dirname(appConfig.emptyPageModelConfig.rfModelPath),
                                         os.path.basename(appConfig.emptyPageModelConfig.rfModelPath), row[1])
                elif row[3] == emptyPageDetectHelperV2.ModelType.xgb.value:
                    fileHelper.delFileIfExists(appConfig.emptyPageModelConfig.xgbModelPath)
                    fileHelper.createDirIfExists(os.path.dirname(appConfig.emptyPageModelConfig.xgbModelPath))
                    fileHelper.writeFile(os.path.dirname(appConfig.emptyPageModelConfig.xgbModelPath),
                                         os.path.basename(appConfig.emptyPageModelConfig.xgbModelPath), row[1])
                elif row[3] == emptyPageDetectHelperV2.ModelType.lgbm.value:
                    fileHelper.delFileIfExists(appConfig.emptyPageModelConfig.lgbmModelPath)
                    fileHelper.createDirIfExists(os.path.dirname(appConfig.emptyPageModelConfig.lgbmModelPath))
                    fileHelper.writeFile(os.path.dirname(appConfig.emptyPageModelConfig.lgbmModelPath),
                                         os.path.basename(appConfig.emptyPageModelConfig.lgbmModelPath), row[1])
                elif row[3] == emptyPageDetectHelperV2.ModelType.catboost.value:
                    fileHelper.delFileIfExists(appConfig.emptyPageModelConfig.catboostModelPath)
                    fileHelper.createDirIfExists(os.path.dirname(appConfig.emptyPageModelConfig.catboostModelPath))
                    fileHelper.writeFile(os.path.dirname(appConfig.emptyPageModelConfig.catboostModelPath),
                                         os.path.basename(appConfig.emptyPageModelConfig.catboostModelPath), row[1])
                elif row[3] == emptyPageDetectHelperV2.ModelType.cnn.value:
                    fileHelper.delFileIfExists(appConfig.emptyPageModelConfig.cnnModelPath)
                    fileHelper.createDirIfExists(os.path.dirname(appConfig.emptyPageModelConfig.cnnModelPath))
                    fileHelper.writeFile(os.path.dirname(appConfig.emptyPageModelConfig.cnnModelPath),
                                         os.path.basename(appConfig.emptyPageModelConfig.cnnModelPath), row[1])
                emptyPageModelCount = emptyPageModelCount + 1
            emptyPageClassifier = emptyPageDetectHelperV2.EmptyPageClassifier(appConfig.emptyPageModelConfig)
            if appConfig.emptyPageModelConfig.isCNNModelUsing:
                emptyPageClassifierCNN = CnnEmptyPageDetector(appConfig.emptyPageModelConfig)
            print("✅✅ Emty Page Detect Model(ler) Yüklendi. Model Sayısı:"+str(emptyPageModelCount))
        if appConfig.useEasyOcrOperation:
            eOcrEngine = easyOcrEngine.EasyOcrEngine(appConfig.ocrModelDir)
            print("✅✅ Ocr Model Yüklendi")
        if appConfig.useGensimModelOperation:
            classificatinDatas = getClassificationTypes(0, True)
            classificatinModel = []
            for classificatinData in classificatinDatas.get("Data"):
                t = type(classificatinData)
                fileHelper.createDirIfExists(appConfig.gensimModelDir)
                modelFilePath = os.path.join(appConfig.gensimModelDir,f"{classificatinData['ClassificationTypeName']}.model" )
                fileHelper.delFileIfExists(modelFilePath)
                fileHelper.writeFile(appConfig.gensimModelDir,
                                     os.path.basename(modelFilePath),classificatinData["ModelData"])
                classificatinModel.append([classificatinData["ClassificationTypeName"], modelFilePath])
                gensimModelCount = gensimModelCount + 1
            go.loadModels(classificatinModel)
            print("✅✅ Classificaiton Model(ler) Yüklendi- Gensim")
        if appConfig.useBertModelOperation:
            bertClassificatinData = getBertClassificationModel()
            if len(bertClassificatinData)>0:
                fileHelper.delFileIfExists(appConfig.bertModelPath)
                fileHelper.createDirIfExists(os.path.dirname(appConfig.bertModelPath))
                fileHelper.writeFile(os.path.dirname(appConfig.bertModelPath),
                                     os.path.basename(appConfig.bertModelPath), bertClassificatinData[0][1])
                bertModelCount = bertModelCount + 1
            bertDocClassifier = BertDocumentClassifier(appConfig.bertModelPath, appConfig.bertModelDir)
            if bertModelCount> 0:
                bertDocClassifier.load_model()
            print("✅✅ Classificaiton Model(ler) Yüklendi- Bert")

        if appConfig.useLlama:
            llama3BInstructHlp = LlamaInstructHelper(appConfig.llamaModelPath, appConfig.debugMode, appConfig.llamaModelType)
            print("✅✅ llama 3B Vision Model Yüklendi- Llama")
        if appConfig.useWhisperTranscribeOperation:
            whisperTranscriber = WhisperTranscriber(appConfig.whisperTranscribeModelPath,"cpu", dtype=torch.float32)
            print("✅✅ Whisper-Base Model Yüklendi- OpenAI")
        if appConfig.useBarcodeDetectOperation:
            rows = getBarcodeDetectModel()
            brcPresenceClassifier = BarcodePresenceClassifier(model_path=appConfig.barcodeDetectModelPath, img_size=(128, 128),
                threshold=0.5,
                deskew=True,
                verbose=True)
            if len(rows)>0:
                fileHelper.delFileIfExists(appConfig.barcodeDetectModelPath)
                fileHelper.createDirIfExists(os.path.dirname(appConfig.barcodeDetectModelPath))
                fileHelper.writeFile(os.path.dirname(appConfig.barcodeDetectModelPath),
                                     os.path.basename(appConfig.barcodeDetectModelPath), rows[0][1])
                barcodeDetectModelCount = barcodeDetectModelCount + 1
                brcPresenceClassifier.load_model(appConfig.barcodeDetectModelPath)
            print("✅✅ Barcode Detect Model Yüklendi")

        print("🧙‍♂️🧙‍♂️ KodA AI Space Servis Yüklenmesi Tamamlandı...👍👍")
        print("Kullanılabilir Modüller ->")
        if appConfig.useRapifFuzzOperation:
            print("---> Rapid Fuzz, Veri Doğrulama")
        if appConfig.useGensimModelOperation:
            print("---> Gensim Doküman Sınıflandırma, Veri Doğrulama")
        if appConfig.useEasyOcrOperation:
            print("---> EasyOcr, Doküman Ocrlama ")
        if appConfig.useBertModelOperation:
            print("---> Bert, Doküman Sınıflandırma ")
        if appConfig.useEmptyPageOperation:
            print("---> Sklearn, Boş Sayfa Tespiti ")
        if appConfig.useLlama:
            print("---> llama 3B Instruct, Veri Doğrulama - Soru -> Cevap")
        if appConfig.useWhisperTranscribeOperation:
            print("---> Whisper-Base, Transcript - Ses -> Metin")
        if appConfig.useBertModelOperation:
            print("---> Sklearn, Barkod Tespiti")

        if appConfig.useQWen:
            qwenHlp = QWenHelper(model_path=appConfig.qwenModelPath,debug_mode=appConfig.debugMode)
            print("✅✅ QWen Model Yüklendi- Llama")
    except Exception as e:
        raise e
#endregion

#region classification
def getClassificationTypes(id:int, withModelData = False):
    try:
        if not appConfig.useGensimModelOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        rows = None
        if id>0:
            rows =  getClassificationType(id)
        else:
            rows = dataOperation.getData('''SELECT Id,ClassificationTypeName,ModelData, Deleted FROM ClassificationType''',None)
        data = classificationDataToList(rows, withModelData)
        return {"Data": data, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def getClassificationType(id):
    rows = dataOperation.getData('''SELECT Id,ClassificationTypeName,ModelData, Deleted FROM ClassificationType WHERE Id = ?''', [id])
    return rows

def classificationDataToList(data, withModelData = False):
    detail = []
    for row in data:
        #t = type(row[3])
        # detail.append({"Id" :int(row[0]), "ClassificationTypeName":str(row[1])
        #                   , "ModelData": json.dumps(row[2].decode('utf-8')), "Deleted":int(row[3])})
        if withModelData:
            detail.append({"Id": int(row[0]), "ClassificationTypeName": str(row[1])
                              , "Deleted": int(row[3]), "ModelData": row[2]})
        else:
            detail.append({"Id": int(row[0]), "ClassificationTypeName": str(row[1])
                              , "Deleted": int(row[3])})
    return detail

def insertClassificationType(classificationTypeName, modelData, deleted, id):
    if id>0:
        dataOperation.execCommand("DELETE FROM ClassificationType WHERE Id = ?", [id])

    dataOperation.execCommand("INSERT INTO ClassificationType(ClassificationTypeName,ModelData, Deleted) VALUES(? , ? , ?)",
                   [classificationTypeName, modelData, deleted])
    rows = dataOperation.getData(
        '''SELECT Id,ClassificationTypeName,ModelData, Deleted FROM ClassificationType WHERE ClassificationTypeName = ?''',
        [classificationTypeName])
    return rows

def deleteClassificationType(id):
    try:
        if not appConfig.useGensimModelOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        dataOperation.execCommand('''DELETE FROM ClassificationType WHERE Id = ?''', [id])
        return {"Data": None, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def trainClassificationData(jsonData):
    try:
        if not appConfig.useGensimModelOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        classificationName = jsonData.get("classificationName")
        data = jsonData.get("sampleDatas")
        sampleData = strHelper.jsonArray2StrArray(data)
        #train ve db kayıt
        train_corpus = list(go.read_corpus(sampleData))
        model = go.createNewClassificationModel(train_corpus)
        modelFolder = fileHelper.createDirIfExists(appConfig.gensimModelDir)
        modelFilePath = os.path.join(modelFolder, (classificationName + ".model"))
        fileHelper.delFileIfExists(modelFilePath)
        model.save(modelFilePath)
        modelBytes = fileHelper.readFileAllBytes(modelFilePath)
        dataRecord = getClassificationType(classificationName)
        id = 0
        if len(dataRecord) > 0:
            id = dataRecord[0][0]
        insertClassificationType(classificationName, modelBytes, 0, id)
        return {"Data": True, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def similarClassification(docContent):
    global  appConfig, gensimModelCount
    try:
        if not appConfig.useGensimModelOperation or gensimModelCount == 0:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        similarity = go.similarClassification(docContent)
        return {"Data": similarity, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
#endregion
#region emptyPages
def insertEmptyPageModel(modelData, deleted, modelType):
    dataOperation.execCommand("DELETE FROM EmptyPageModel WHERE ModelType = ?", [modelType])

    new_id = dataOperation.execCommand("INSERT INTO EmptyPageModel(ModelFile, Deleted, ModelType) VALUES(? , ? , ?)",
                   [ modelData, deleted, modelType], True)
    rows = dataOperation.getData(
        '''SELECT EmptyPageModelId,ModelFile,Deleted, ModelType FROM EmptyPageModel WHERE EmptyPageModelId = ?''',
        [new_id])
    return rows

def getEmptyPageModels():
    rows = dataOperation.getData( '''SELECT EmptyPageModelId,ModelFile,Deleted,ModelType FROM EmptyPageModel''', None)
    return rows

def trainEmptyPageModel(emptyPagesDir, filledPagesDir, modelType):
    global appConfig, emptyPageClassifierCNN
    try:
        if not appConfig.useEmptyPageOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        emptyPagesDir = "" if emptyPagesDir == None else emptyPagesDir
        filledPagesDir = "" if filledPagesDir == None else filledPagesDir
        if (emptyPagesDir == "" or filledPagesDir == "" or
                not os.path.isdir(emptyPagesDir) or not os.path.isdir(filledPagesDir)):
            return {"Data": None, "ErrorMessage": "Eğitim Dosyaları Eksik", "Description": "", "Error": True}
        modelFileName: str = ""
        if modelType == emptyPageDetectHelperV2.ModelType.cnn.value and appConfig.emptyPageModelConfig.isCNNModelUsing:
            modelFileName = emptyPageClassifierCNN.train(emptyPagesDir, filledPagesDir)
        else:
            modelFileName = emptyPageClassifier._train_model(emptyPagesDir, filledPagesDir, modelType)
        modelFileBytes = fileHelper.readFileAllBytes(modelFileName)
        insertEmptyPageModel(modelFileBytes,0,modelType)
        return {"Data": None, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}


def detectEmptyPage(imageBase64_srt):
    global appConfig, emptyPageModelCount,emptyPageClassifierCNN
    try:
        if not appConfig.useEmptyPageOperation or emptyPageModelCount == 0:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        if imageBase64_srt is None:
            return {"Data": None, "ErrorMessage": "Geçersiz Image Datası", "Description": "", "Error": True}
        import datetime
        print(datetime.datetime.now())
        data = emptyPageClassifier.predict(imageBase64_srt, False)
        print(datetime.datetime.now())
        if emptyPageClassifierCNN is not None and appConfig.emptyPageModelConfig.isCNNModelUsing and emptyPageClassifierCNN.checkModelExists():
            cnnData = emptyPageClassifierCNN.predict_from_base64(imageBase64_srt)
            print(datetime.datetime.now())
            data.append(cnnData)
        return {"Data": data, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

#endregion
#region ocr
def ocrTifImage(imageBase64_srt):
    global  eOcrEngine
    try:
        if not appConfig.useEasyOcrOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        if imageBase64_srt is None:
            return {"Data": None, "ErrorMessage": "Geçersiz Image Datası", "Description": "", "Error": True}
        ocrResult = eOcrEngine.OcrTiffImage(imageBase64_srt)
        return {"Data": ocrResult, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def ocrTifImageWithDetails(imageBase64_srt,useParagraph: bool = True, useWidth_ths: float = 10.0):
    global  eOcrEngine
    try:
        if not appConfig.useEasyOcrOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        if imageBase64_srt is None:
            return {"Data": None, "ErrorMessage": "Geçersiz Image Datası", "Description": "", "Error": True}
        ocrResult = eOcrEngine.OcrTiffImageWithDetails(imageBase64=imageBase64_srt, useParagraph=useParagraph, useWidth_ths=useWidth_ths)
        return {"Data": ocrResult, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
#endregion
#region data similarity
def dataExistsRatioGensim(data):
    try:
        if not appConfig.useGensimModelOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        content = data.get("content")
        checkValues = data.get("checkValues")
        if content is None or checkValues is None:
            return {"Data": None, "ErrorMessage": "Giriş değerleri geçersiz", "Description": "", "Error": True}
        resultData = go.dataExistsRatio(content, checkValues)
        return {"Data": resultData, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def dataExistsRatioRapidFuzz(data):
    try:
        if not appConfig.useRapifFuzzOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        content = data.get("content")
        checkValues = data.get("checkValues")
        if content is None or checkValues is None:
            return {"Data": None, "ErrorMessage": "Giriş değerleri geçersiz", "Description": "", "Error": True}
        resultData = rf.dataExistsRatio(content, checkValues)
        return {"Data": resultData, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def semanticDataExistsRatioBert(data):
    try:
        if not appConfig.useBertModelOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        content = data.get("content")
        checkValues = data.get("checkValues")
        if content is None or checkValues is None:
            return {"Data": None, "ErrorMessage": "Giriş değerleri geçersiz", "Description": "", "Error": True}
        resultData = bertDocClassifier.semanticSimilarityCalculation(content, checkValues)
        if resultData is None:
            return {"Data": None, "ErrorMessage": "Benzerlik Hesaplanamadı", "Description": "", "Error": True}
        return {"Data": resultData, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def semanticDataExistsRatioLlama(data):
    global llama3BInstructHlp
    try:
        if not appConfig.useLlama:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        content = data.get("content")
        checkValues = data.get("checkValues")
        if content is None or checkValues is None:
            return {"Data": None, "ErrorMessage": "Giriş değerleri geçersiz", "Description": "", "Error": True}
        resultData = llama3BInstructHlp.verify(content, checkValues)
        if resultData is None:
            return {"Data": None, "ErrorMessage": "Benzerlik Hesaplanamadı", "Description": "", "Error": True}
        return {"Data": resultData, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
#endregion
#region bertClassification
def insertBertClassificationModel(modelData, deleted):
    dataOperation.execCommand("DELETE FROM BertClassificationModel", None)

    new_id = dataOperation.execCommand("INSERT INTO BertClassificationModel(ModelFile, Deleted) VALUES(? , ?)",
                   [ modelData, deleted], True)
    rows = dataOperation.getData(
        '''SELECT BertClassificationModelId,ModelFile,Deleted FROM BertClassificationModel WHERE BertClassificationModelId = ?''',
        [new_id])
    return rows

def getBertClassificationModel():
    rows = dataOperation.getData( '''SELECT BertClassificationModelId,ModelFile,Deleted FROM BertClassificationModel''', None)
    return rows

def getBertClassificationModelCount():
    try:
        rows = dataOperation.getData( '''SELECT BertClassificationModelId FROM BertClassificationModel''', None)
        return {"Data": len(rows), "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def trainBertClassificationModel(jSonBase64str):
    try:
        if not appConfig.useBertModelOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        jSonBase64str = "" if jSonBase64str == None else jSonBase64str
        if jSonBase64str == "":
            return {"Data": None, "ErrorMessage": "Eğitim Dosyası Eksik", "Description": "", "Error": True}
        bertDocClassifier.train(jSonBase64str)
        modelFileBytes = fileHelper.readFileAllBytes(appConfig.bertModelPath)
        insertBertClassificationModel(modelFileBytes,0)
        return {"Data": None, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

def detectBertClassification(docContent):
    global appConfig, bertModelCount
    try:
        if bertModelCount == 0:
            return {"Data": None, "ErrorMessage": "Kullanılabilir Model Yok", "Description": "", "Error": True}
        similarity = bertDocClassifier.predict(docContent)
        return {"Data": similarity, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
#endregion

#region whisper transcribe

def transcribeAudio(aoudioBase64_srt):
    global  whisperTranscriber
    try:
        if not appConfig.useWhisperTranscribeOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        if aoudioBase64_srt is None:
            return {"Data": None, "ErrorMessage": "Geçersiz Ses Datası", "Description": "", "Error": True}
        tmpFileName = fileHelper._write_to_tempfile(base64.b64decode(aoudioBase64_srt))
        result = whisperTranscriber.transcribe_file(tmpFileName)
        fileHelper.tryDeleteFile(tmpFileName)
        return {"Data": result, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

#endregion

#region llmaQuestionAnswer
def llamaQuestionAnswer(content:str, quesiton:str)->str:
    try:
        if not appConfig.useLlama:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        result = llama3BInstructHlp.answer_question(content, quesiton)
        return {"Data": result, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
#endregion

#region barcodeDetect
def insertBarcodeDetectModel(modelData, deleted):
    dataOperation.execCommand("DELETE FROM BarcodeDetectModel", None)

    new_id = dataOperation.execCommand("INSERT INTO BarcodeDetectModel(ModelFile, Deleted) VALUES(? , ?)",
                   [ modelData, deleted], True)
    rows = dataOperation.getData(
        '''SELECT BarcodeDetectModelId,ModelFile,Deleted FROM BarcodeDetectModel WHERE BarcodeDetectModelId = ?''',
        [new_id])
    return rows

def getBarcodeDetectModel():
    rows = dataOperation.getData( '''SELECT BarcodeDetectModelId,ModelFile,Deleted FROM BarcodeDetectModel''', None)
    return rows

def trainBarcodeDetectModel(barcodeDir, nobarcodeDir):
    global appConfig, brcPresenceClassifier
    try:
        if not appConfig.useBarcodeDetectOperation:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        barcodeDir = "" if barcodeDir == None else barcodeDir
        nobarcodeDir = "" if nobarcodeDir == None else nobarcodeDir
        if (barcodeDir == "" or nobarcodeDir == "" or
                not os.path.isdir(barcodeDir) or not os.path.isdir(nobarcodeDir)):
            return {"Data": None, "ErrorMessage": "Eğitim Dosyaları Eksik", "Description": "", "Error": True}
        modelFileName = brcPresenceClassifier.train(dataset_dir_barcode= barcodeDir,
                                                    dataset_dir_nobarcode=nobarcodeDir,model_type="rf")
        modelFileBytes = fileHelper.readFileAllBytes(modelFileName)
        insertBarcodeDetectModel(modelFileBytes,0)
        return {"Data": None, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        print(f"🛑🛑🛑Hata : {str(e)}")
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}


def detectBarcode(imageBase64_srt):
    global appConfig, barcodeDetectModelCount, brcPresenceClassifier
    try:
        if appConfig.useBarcodeDetectOperation == False or barcodeDetectModelCount == 0:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        if imageBase64_srt is None:
            return {"Data": None, "ErrorMessage": "Geçersiz Image Datası", "Description": "", "Error": True}
        found = data = brcPresenceClassifier.detect_on_page(imageBase64_srt,
            return_locations=True,
            verbose=True,
            save_candidates=True,
            candidate_dir="temp_candidates")
        return {"Data": {"Found" : found}, "ErrorMessage": "", "Description": "", "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
#endregion

#region qwenQuestionAnswer
def qwenQuestionAnswer(content:str,schema:str, userPromt:str)->str:
    try:
        if not appConfig.useQWen:
            return {"Data": None, "ErrorMessage": "Özellik Kullanılabilir Değil", "Description": "", "Error": True}
        _schema = None
        if schema != None:
            _schema = json.loads(schema)
        result = qwenHlp.extract_fields(image_input= content, max_new_tokens=300, schema=_schema, user_prompt=userPromt)
        return {"Data": result, "ErrorMessage": "", "Description": "",
                "Error": False}
    except Exception as e:
        return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}

#endregion

# def dataExistsRatioSpacy(data):
#     try:
#         content = data.get("content")
#         checkValues = data.get("checkValues")
#         if content is None or checkValues is None:
#             return {"Data": None, "ErrorMessage": "Giriş değerleri geçersiz", "Description": "", "Error": True}
#         resultData = so.dataExistsRatio(content, checkValues)
#         return {"Data": resultData, "ErrorMessage": "", "Description": "", "Error": False}
#     except Exception as e:
#         print(f"🛑🛑🛑Hata : {str(e)}")
#         return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}
# def dataExistsRatioSpacy2(data):
#     try:
#         content = data.get("content")
#         checkValues = data.get("checkValues")
#         if content is None or checkValues is None:
#             return {"Data": None, "ErrorMessage": "Giriş değerleri geçersiz", "Description": "", "Error": True}
#         resultData = so.dataExitstCheck(content, checkValues)
#         return {"Data": resultData, "ErrorMessage": "", "Description": "", "Error": False}
#     except Exception as e:
#         print(f"🛑🛑🛑Hata : {str(e)}")
#         return {"Data": None, "ErrorMessage": str(e), "Description": "", "Error": True}









