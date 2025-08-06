from rapidfuzz import fuzz

# Büyük ve küçük metinler

def dataExistsRatio(content, checkValues):
    # buyuk_metin = "Bu büyük bir metin örneğidir. Yapay zeka harun ile metin analizi yapmak eğlencelidir. Metin büyük olabilir.".lower()
    # kucuk_metin = "harun metin ".lower()
    # Benzerlik oranı
    result =[]
    for text in checkValues:
        ratio = fuzz.partial_ratio(content.lower(), text.get("text").lower())
        result.append({"key": text.get("key"), "text":  text.get("text"), "SimilarityRatio":ratio})
    return result