import os
import sys
import gc
import random
import string
import re
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities
import numpy as np
from scipy import spatial


# 去茎化
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    removed_stopwords = [
        stemmer.stem(item) for item in tokens if item not in stopwords.words("english")
    ]

    return removed_stopwords


# 去掉一般连词 / 全部小写 / 复数化成单数形式
def normalize(text):
    remove_punc_map = dict((ord(char), None) for char in string.punctuation)
    removed_punc = text.lower().translate(remove_punc_map)
    tokenized = word_tokenize(removed_punc)
    stemmed_tokens = stem_tokens(tokenized)

    return stemmed_tokens


def VSM(text):
    vectorizer = TfidfVectorizer(min_df=3, token_pattern=r"(?u)\b\w+\b")
    count = vectorizer.fit_transform(text)
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count)
    # print(tfidf_matrix)
    tfidf_array = tfidf_matrix.toarray()
    return tfidf_array


def cosine_sim(text):
    # vectorizer = TfidfVectorizer(tokenizer=normalize, min_df=1, stop_words="english")
    # cv = CountVectorizer()
    # cv_fit = cv.fit_transform([text1, text2])
    # print(cv_fit.toarray())

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(text)
    matrix = tfidf * tfidf.T
    sim = (tfidf * tfidf.T).A[0, 1]

    return sim


def topic_sim(doclist, query):
    texts = []
    for doc in doclist:
        texts += [normalize(doc)]
    # print(texts)
    # print(texts)
    # 抽取一个"词袋（bag-of-words)"，将文档的token映射为id
    dictionary = corpora.Dictionary(texts)
    # print(dictionary)
    print(dictionary.token2id)
    # 将用字符串表示的文档转换为用id表示的文档向量
    corpus = [dictionary.doc2bow(text) for text in texts]
    print("corpus: ", corpus)
    # 计算tf-idf模型
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #     print(doc)
    # print(tfidf.idfs)
    # 跑lda模型
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    # print(lda.print_topics(2))

    # 将文档映射到一个二维的topic空间中
    corpus_lda = lda[corpus_tfidf]
    # for doc in corpus_lda:
    #     print(doc)

    # 对文档建立索引
    index = similarities.MatrixSimilarity(lda[corpus])
    # 将query向量化
    query_bow = dictionary.doc2bow(normalize(query))
    # print(query_bow)
    # 用之前训练好的LDA模型将query映射到二维的topic空间
    query_lda = lda[query_bow]
    # print(query_lda)

    # 计算query和index中doc的余弦相似度
    lda_sims = index[query_lda]
    # print(lda_sims)
    # print(list(enumerate(lda_sims)))
    return lda_sims


def semantic_sim(text1, text2):
    # pip install pyemd
    model = models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    # calculate distance between two sentences using WMD algorithm - Word Mover距离算法
    distance = model.wmdistance(text1, text2)
    # print(distance)
    return distance


# 数据预处理
def preprocess(text):
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    # 分词
    words = word_tokenize(text)

    processed_words = []

    for word in words:
        # 去停止词
        if word.isalnum() and word.lower() not in stop_words:
            # 分割驼峰命名法单词
            split_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', word).split()
            for split_word in split_words:
                # 小写 词性还原
                lemmatized_word = lemmatizer.lemmatize(split_word.lower())
                processed_words.append(lemmatized_word)
    return processed_words


'''
data = pd.read_csv('xxx.csv')
# print(data)

cosine = []
topic = []
semantic = []

for index, row in data.iterrows():
    print(row['BugId'])
    bug_text = row['BugReport_text'] # Bug Report的文本内容 summary+description
    sourcefile_text = row['SourceFile_txt'] # Source File的文本内容 code+comment

    br = preprocess(bug_text)
    cf = preprocess(sourcefile_text)
    text = br + cf

    cosine.append(cosine_sim(text)) # 文本相似性


    doclist = []
    doclist.append(cf)
    doclist.append(br)
    topics = topic_sim(doclist, br) # 主题相似性
    print(topics)
    print(topics[0])
    topic.append(topics[0])

    semantic.append(semantic_sim(br, cf)) # 语义相似性

data['cosine_sim'] = cosine
data['topic_sim'] = topic
data['semantic_sim'] = semantic

print(data)

'''