from gensim.models import Word2Vec
import numpy as np


def get_vectors(sentences, model_file, max_sentence_len):
    '''
    获取给定句子的词向量，词向量不存在或者句子长度不够补0
    '''
    w2vModel = Word2Vec.load(model_file)

    all_vectors = []
    embeddingDim = w2vModel.vector_size
    print("max_sentence_len:", max_sentence_len)
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        #如果超过最大长度，则只截取最大长度
        if len(sentence) > max_sentence_len:
            sentence = sentence[:max_sentence_len]
        
        #获取词向量
        this_vector = []
        for word in sentence:
            if word in w2vModel.wv.vocab:
                this_vector.append(w2vModel[word])
            else:
                this_vector.append(embeddingUnknown)
                
        #长度不够补0
        fillLen = max_sentence_len - len(sentence)        
        for a in range(fillLen):
            this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors


def test():
    vectors = get_vectors([['一', '句子','不足'], ['three', '句子']], 'wiki/wiki.model', 4)
    print(vectors)
    vectors = np.array(vectors)
    print(vectors.shape)

if __name__ == "__main__":
    test()
