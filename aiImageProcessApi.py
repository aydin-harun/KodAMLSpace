from flask import Flask, request, jsonify, render_template
from libs.emptyPageDetect.EmptyPageDetectHelper import trainModel, predict_page
import os
import bussiness.imageProcessBS as imgProcessBS
import  json


app = Flask(__name__)

imgProcessBS.loadAppConfig()

@app.route("/")
def index():
    imgProcessBS.logger.info("Root endpoint called")
    return render_template("index.html")

@app.route("/trainemptymodel")
def train_page():
    return render_template("trainemptymodel.html")

@app.route("/detectemptypage")
def detect_empty_page():
    return render_template("detectemptypage.html")

@app.route("/ocr")
def ocr_page():
    return render_template("ocr.html")

@app.route("/api/trainEmptyPageModel", methods=["POST"])
def api_train_emptyPage_model():
    data = request.get_json()
    empty_dir = data.get("empty_dir")
    filled_dir = data.get("filled_dir")
    model_type = data.get("model_type")
    result = imgProcessBS.trainEmptyPageModel(empty_dir, filled_dir, model_type)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jData , 400 # result, 400
    return  {"Success": True, "Code": 200, "Data": str(result), "Description": "Operation Successful"}, 200

@app.route("/api/detectEmptyPage", methods=["POST"])
def api_detectEmptyPage_page():
    data = request.get_json()
    input_data = data.get("imageBase64")
    result = imgProcessBS.detectEmptyPage(input_data)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jsonify(jData) , 400# result, 400
    data = result.get("Data")
    return jsonify({"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}), 200

@app.route("/api/ocrtifimage", methods=["POST"])
def api_ocr_tif_image():
    data = request.get_json()
    input_data = data.get("imageBase64")
    result = imgProcessBS.ocrTifImage(input_data)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jsonify(jData) , 400# result, 400
    return  jsonify({"Success": True, "Code": 200, "Data": result, "Description": "Operation Successful"}), 200

@app.route("/api/ocrtifimagewithdetails", methods=["POST"])
def api_ocr_tif_image_withdetails():
    data = request.get_json()
    input_data = data.get("imageBase64")
    item_useParagraph = data['useParagraph']
    item_useWidth_ths = data['useWidth_ths']
    result = imgProcessBS.ocrTifImageWithDetails(imageBase64_srt= input_data,useParagraph= item_useParagraph,useWidth_ths= item_useWidth_ths)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jsonify(jData) , 400# result, 400
    jData = {"Success": True, "Code": 200, "Data": result, "Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route("/api/test", methods=["GET"])
def test():
    return "Test Tamamlandı"

#region classification

@app.route("/api/getclassifications", methods=["GET"])
def getclassifications():
    qId =int(request.args.get("id"))
    if qId is None:
        qId=0
    result = imgProcessBS.getClassificationTypes(qId, False)
    data = result.get("Data")
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    jData = {"Success": True, "Code": 200, "Data":data, "Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route("/api/deleteclassification", methods=["DELETE"])
def deleteclassification():
    qId =int(request.args.get("id"))
    if qId is None or qId == 0:
        jData = {"Success": False, "Code": 400, "Data": qId}
        return jsonify(jData), 400  # result, 400

    result = imgProcessBS.deleteClassificationType(qId)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    jData = {"Success": True, "Code": 200, "Data":qId, "Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route('/api/starttrain', methods=['POST'])
def startTrain():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": None, "Description":"Invalid Parameters"}
        return jsonify(jData), 400  # result, 400
    cName = reqContent.get('classificationName')
    result = imgProcessBS.trainClassificationData(reqContent)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage") }
        return jsonify(jData), 400  # result, 400
    # writeText2file("/ClassifierRoot/"+cName+"/TrainData.json",json.dumps(reqContent) )
    jData = {"Success": True, "Code": 200, "Data": f"{cName} - Train Completed","Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route('/api/getclassification', methods=['POST'])
def getClassification():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": None, "Description":"Invalid Parameters" }
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.similarClassification(reqContent.get("content"))
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200
#end region

@app.route('/api/getdataexistsratiog', methods=['POST'])
def getdataexistsratiog():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": "Invalid Parameters"}
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.dataExistsRatioGensim(reqContent)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200  # result, 400

@app.route('/api/getdataexistsratiorf', methods=['POST'])
def getdataexistsratiorf():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": "Invalid Parameters"}
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.dataExistsRatioRapidFuzz(reqContent)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200  # result, 400

@app.route('/api/startbertclassificationtrain', methods=['POST'])
def startBertClassificationTrain():
    data = request.get_json()
    input_data = data.get("trainBase64Data")
    if len(str(input_data)) == 0:
        jData = {"Success": False, "Code": 400, "Data": None, "Description":"Invalid Parameters"}
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.trainBertClassificationModel(str(input_data))
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage") }
        return jsonify(jData), 400  # result, 400
    # writeText2file("/ClassifierRoot/"+cName+"/TrainData.json",json.dumps(reqContent) )
    jData = {"Success": True, "Code": 200, "Data": f"{True} - Train Completed","Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route("/api/getbertclassificationdatacount", methods=["GET"])
def getbertclassificationdatacount():
    result = imgProcessBS.getBertClassificationModelCount()
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    # writeText2file("/ClassifierRoot/"+cName+"/TrainData.json",json.dumps(reqContent) )
    jData = {"Success": True, "Code": 200, "Data": result["Data"], "Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route("/classifications")
def classifications_page():
    return render_template("classifications.html")

@app.route("/trainclassification")
def train_classification_page():
    return render_template("train_classification.html")

@app.route("/getclassification")
def get_classification_page():
    return render_template("get_classification.html")

@app.route("/checkdataexists")
def check_data_exists_page():
    return render_template("check_data_exists.html")

@app.route("/bertclassificationtrain")
def bert_classification_train_page():
    model_count = imgProcessBS.getBertClassificationModelCount()
    return render_template("bert_classification_train.html", model_count=model_count.get("Data"))

@app.route("/bertclassificationdetect")
def bert_classification_detect_page():
    return render_template("bert_classification_detect.html")

@app.route('/api/getbertclassification', methods=['POST'])
def getBertClassification():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": None, "Description":"Invalid Parameters" }
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.detectBertClassification(reqContent.get("content"))
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route('/api/getsemanticdataexistsratiorb', methods=['POST'])
def getSemantikDataExistsRatioB():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": "Invalid Parameters"}
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.semanticDataExistsRatioBert(reqContent)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200  # result, 400

@app.route('/api/getsemanticdataexistsratiol', methods=['POST'])
def getSemantikDataExistsRatioLlama():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": "Invalid Parameters"}
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.semanticDataExistsRatioLlama(reqContent)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200  # result, 400

@app.route("/audiotranscription")
def audio_transcription_page():
    return render_template("audio_transcription.html")


@app.route("/api/transcribeaudio", methods=["POST"])
def api_transcribeAudio():
    data = request.get_json()
    input_data = data.get("audioBase64")
    result = imgProcessBS.transcribeAudio(input_data)
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None , "Description": result.get("ErrorMessage")}
        return jsonify(jData) , 400# result, 400
    return  jsonify({"Success": True, "Code": 200, "Data": str(result), "Description": "Operation Successful"}), 200

@app.route('/api/questionanswer', methods=['POST'])
def getLlamaQuestionAnswer():
    reqContent = request.get_json()
    if reqContent is None:
        jData = {"Success": False, "Code": 400, "Data": None, "Description":"Invalid Parameters" }
        return jsonify(jData), 400  # result, 400
    result = imgProcessBS.llamaQuestionAnswer(reqContent.get("content"), reqContent.get("question"))
    if result.get("Error"):
        jData = {"Success": False, "Code": 400, "Data": None, "Description": result.get("ErrorMessage")}
        return jsonify(jData), 400  # result, 400
    data = result.get("Data")
    jData = {"Success": True, "Code": 200, "Data": data, "Description": "Operation Successful"}
    return jsonify(jData), 200

@app.route("/llamaquestionanswer")
def llama_question_answer_page():
    return render_template("llama_question_answer.html")

@app.route("/ocrtifimagewithdetails")
def ocr_tif_image_with_details_page():
    return render_template("ocr_tif_image_with_details.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=imgProcessBS.appConfig.debugMode,use_reloader=False, port=imgProcessBS.appConfig.apiPort)

