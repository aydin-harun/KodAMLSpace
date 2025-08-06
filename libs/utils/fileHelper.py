import os
import tempfile

current_path = os.getcwd()
modelPath = os.path.join(current_path,"models")
dataPath = os.path.join(current_path,"DataFolder")

def createDirIfExists(dirName):
    fullDirPath = dirName
    if not os.path.isdir(fullDirPath):
        os.makedirs(fullDirPath)
    return  fullDirPath

def delFileIfExists(filePath):
    if os.path.isfile(filePath):
        os.remove(filePath)

def readFileAllBytes(filePath):
    with open(filePath,'rb') as file:
        contents = file.read()
        return  contents

def writeFile(path, fileName, bytes):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, fileName), 'wb') as file:
        file.write(bytes)
    return os.path.join(path, fileName)

def _write_to_tempfile( data: bytes) -> str:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_file.write(data)
    tmp_file.flush()
    tmp_file.close()
    return tmp_file.name

def tryDeleteFile(filePath):
    try:
        delFileIfExists(filePath)
    except Exception as e:
        print(e)
