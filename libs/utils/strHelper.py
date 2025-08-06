def normalizeString(input):
    removeChars =["*",",",":",";","(",")",">","<","|","!","\"","'","#","^","%","&","/","[","]","=","?","\\","{","}","@","\r\n","\n","\t","\r"]
    replaceChars=[["ä","a"],["Ä","A"],["ã","a"],["Ã","A"],["Ğ","G"],["ğ","g"],["Ü","U"],["ü","u"],["Ş","S"],["ş","s"],["Ö","O"],["ö","ö"],["Ç","C"],["ç","c"],["ı","i"],["î","i"],["î","i"],["Î","I"],["ì","i"],["Ì","I"]]
    for rmc in removeChars:
        input =  input.replace(rmc," ")
    for rpc in replaceChars:
        input =  input.replace(rpc[0],rpc[1])
    return twoCharsWordClean(input.lower())

def twoCharsWordClean(sentence):
    words = sentence.split()  # Cümleyi kelimelere ayır
    cleanWords = [word for word in words if len(word) > 2]  # İki karakterli kelimeleri filtrele
    cleanSentence = ' '.join(cleanWords)  # Geri kalan kelimeleri birleştir
    return cleanSentence

def jsonArray2StrArray(jsonArray):
    returnList = []
    for item in jsonArray:
        returnList.append( item.get("docContent"))
    return returnList

def normalizeString_tr(input):
    removeChars =["*",",",":",";","(",")",">","<","|","!","\"","'","#","^","%","&","/","[","]","=","?","\\","{","}","@","\r\n","\n","\t","\r"]
    replaceChars=[["ä","a"],["Ä","A"],["ã","a"],["Ã","A"],["î","i"],["î","i"],["Î","İ"],["ì","i"],["Ì","İ"]]
    for rmc in removeChars:
        input =  input.replace(rmc," ")
    for rpc in replaceChars:
        input =  input.replace(rpc[0],rpc[1])
    return twoCharsWordClean(input.lower())

import re
import  base64

def normalize_text(text: str) -> str:
    # Küçük harf
    text = text.lower()
    # Şapkalı Türkçe karakterleri normal hallerine getir
    text = text.replace('â', 'a').replace('î', 'i').replace('û', 'u')
    # Türkçe karakterleri İngilizce eşlerine çevir
    turkish_map = str.maketrans({
        'ç': 'c',
        'ğ': 'g',
        'ı': 'i',
        'ö': 'o',
        'ş': 's',
        'ü': 'u',
    })
    text = text.translate(turkish_map)
    # Sayıları sil
    text = re.sub(r'\d+', '', text)
    # Noktalama işaretlerini sil
    text = re.sub(r'[^\w\s]', '', text)
    # Fazla boşlukları tek boşluk yap
    text = re.sub(r'\s+', ' ', text).strip()
    # Enter, tab, carriage return karakterlerini boşluğa çevir
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return twoCharsWordClean(text)

def convertBase64StrtoStr(base64str):
    base64_str = base64.b64decode(base64str)
    return base64_str


