import numpy as np
import jieba
import zhconv
import re
import io
import pickle

def clean_str(string):
    """
    短信文本预处理
    1.去掉短信文本中的特殊字符用空格代替
    2.繁体字转简体字
    """
    
    #去掉特殊字符
    #string = re.sub(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,。？“”]+'," ",string)
    #string = re.sub(u'[^0-9a-zA-Z\u4e00-\u9fa5]+'," ",string)
    string = re.sub(u'[^\u4e00-\u9fa5]+'," ",string)
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #string = re.sub(r"!", " ! ", string)
    #string = re.sub(r"\(", " \( ", string)
    #string = re.sub(r"\)", " \) ", string)
    #string = re.sub(r"\?", " \? ", string)
    #string = re.sub(r"\s{2,}", " ", string)
    #return string.strip().lower()
    
    #繁体字转换为简体字
    string = zhconv.convert(string.strip(), 'zh-hans')
    return string


def load_data(dataFile):
    """
    从数据文件中加载数据，并进行数据清洗
    返回清洗后的文本和标签.
    """
    lines = list(open(dataFile, "r", encoding="utf-8").readlines())
    y = [line[:1] for line in lines]
    x_text = [clean_str(line[1:]) for line in lines]
    
    #y = [[0,1] if label=='0' else [1,0] for label in y]

    return [x_text, y]



def cut_sentences(sentences):
    '''
    对句子进行中文分词
    返回分词后的句子，和最长的句子长度
    '''
    return [list(cut_sentence(sentence)) for sentence in sentences]

def cut_sentence(sentence):
    words = []
    cut = jieba.cut(sentence)
    for word in cut:
        if word != " ":
            words.append(word)
    return words

def filter_stopword(sentences, stopwords_file):
    '''
    停用词过滤
    ''' 
    # 读停用词
    stopwords = []
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line)>0:
                stopwords.append(line.strip())
    # 过滤停用词
    sentences_new = []
    for sentence in sentences:
        words = []
        for word in sentence:
            if word not in stopwords:
                words.append(word)
        sentences_new.append(words)  
        
    return sentences_new


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    为数据集生成批处理迭代器
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # 每轮对数据进行重新洗牌
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def saveDict(input_dict, dict_file):
    with open(dict_file, 'wb') as f:
        pickle.dump(input_dict, f) 

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict
    

if __name__ == "__main__":
    dataFile = "data/sms.txt"

    load_data(dataFile)
    
