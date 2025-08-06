import gensim
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric, split_alphanum, remove_stopwords
import libs.utils.strHelper as strHelper
import io
from numpy import dot
from numpy.linalg import norm
import math

models =[]

def read_corpus(sampleDatas, tokens_only=False):
    i = -1
    for sampleData in sampleDatas:
        line = strHelper.normalize_text(sampleData)
        line = remove_stopwords(line)
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            i+=1
            yield tokens
        else:
            # For training data, add tags
            i += 1
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def createNewClassificationModel(train_corpus):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=1000, min_count=3, workers=10, epochs=500, window=2)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def loadModels(modelRows):
    global models
    if modelRows is None:
        return
    for modelRow in modelRows:
        classificationTypeName = modelRow[0]
        modelFile = modelRow[1]
        model =gensim.models.LdaModel.load(modelFile)
        models.append([classificationTypeName, model])

def similarClassification(docContent):
    global  models
    if len(models) == 0:
        return None
    nContent = strHelper.normalize_text(docContent)
    tokens = gensim.utils.simple_preprocess(nContent)
    classificationlist= []
    for modelInfo in models:
        inferred_vector = modelInfo[1].infer_vector(tokens)
        sims = modelInfo[1].dv.most_similar([inferred_vector], topn=len(modelInfo[1].dv))
        classificationResult = {"Oran" :sims[0][1], "Tip" : modelInfo[0]}
        classificationlist.append(classificationResult)
    return classificationlist

def dataExistsRatio(content, checkValues):
    result = []
    # Metni ön işleme tabi tut
    # bigTextTokens = preprocess_string(strHelper.normalizeString_tr(content.lower()))
    bigTextTokens = split_alphanum(strHelper.normalizeString_tr(content.lower()))
    # Word2Vec modelini eğit
    model = gensim.models.doc2vec.Word2Vec([bigTextTokens], vector_size=100, window=5, min_count=1, workers=10, epochs=100)
    # Büyük metindeki her kelimenin vektörü
    bigTextVector = sum([model.wv[token] for token in bigTextTokens if token in model.wv])
    for text in checkValues:
        # Metni ön işleme tabi tut
        smallTextTokens = split_alphanum(strHelper.normalizeString_tr(text.get("text").lower()))
        # Küçük metnin vektörünü elde et
        smallTextVector = sum([model.wv[token] for token in smallTextTokens if token in model.wv])
        if type(smallTextVector) is int:
            result.append({"key": text.get("key"), "text": text.get("text"), "SimilarityRatio": 0})
        else:
            cosine_sim = dot(smallTextVector, bigTextVector) / (norm(smallTextVector) * norm(bigTextVector))
            result.append({"key": text.get("key"), "text": text.get("text"), "SimilarityRatio": math.floor(cosine_sim * 100)})
    return result
