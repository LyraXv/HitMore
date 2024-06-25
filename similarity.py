import os
# import git
import sys
import gc
import random
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
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

# ！！！
def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(tokenizer=normalize, min_df=1, stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]
    return sim


def topic_sim(doclist, query):
    texts = []
    for doc in doclist:
        texts += [stem_tokens(normalize(doc))]
    print(texts)
    # 抽取一个"词袋（bag-of-words)"，将文档的token映射为id
    dictionary = corpora.Dictionary(texts)
    # print(dictionary)
    # print(dictionary.token2id)
    # 将用字符串表示的文档转换为用id表示的文档向量
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print(corpus)
    # 计算tf-idf模型
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #     print(doc)
    # print(tfidf.idfs)
    # 跑lda模型
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    print(lda.print_topics(2))

    # 将文档映射到一个二维的topic空间中
    corpus_lda = lda[corpus_tfidf]
    for doc in corpus_lda:
        print(doc)

    # 对文档建立索引
    index = similarities.MatrixSimilarity(lda[corpus])
    # 将query向量化
    query_bow = dictionary.doc2bow(stem_tokens(normalize(query)))
    # print(query_bow)
    # 用之前训练好的LDA模型将query映射到二维的topic空间
    query_lda = lda[query_bow]
    print(query_lda)

    # 计算query和index中doc的余弦相似度
    lda_sims = index[query_lda]
    # print(lda_sims)
    # print(list(enumerate(lda_sims)))
    return lda_sims


# pip install pyemd
model = models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
def semantic_sim(text1, text2):
    #calculate distance between two sentences using WMD algorithm - Word Mover距离算法
    distance = model.wmdistance(text1, text2)
    # print(distance)
    return distance


text1 = 'We are close to wrapping up our 10 week Rails Course. This week we will cover a handful of topics commonly encountered in Rails projects.'
text2 = 'We are close to wrapping up our 10 week Rails Course.'


# 文本相似性
# print(stem_tokens(normalize(text1)))
# print(stem_tokens(normalize(text2)))
# print(cosine_sim(text1, text2))

# LDA - 主题相似性
# doclist = []
# doclist.append("Shipment of gold damaged in a fire")
# doclist.append("Delivery of silver arrived in a silver truck")
# doclist.append("Shipment of gold arrived in a truck")
# print(doclist)
# query = "gold silver truck"
# topic = topic_sim(doclist, query)
# for res in topic:
#     print(res)

# Word2Vec - 语义相似性
# t1 = normalize(text1)
# t2 = normalize(text2)
# semantic = semantic_sim(t1, t2)
# print(semantic)
# semantic = semantic_sim(text1, text2)
# print(semantic)

# train = pd.read_csv('data/zookeeper_dat/train_final.csv')
# label = pd.read_csv('data/zookeeper_dat/train_label1_final.csv').rename(columns={'label': 'Label'})
# print(train)
# print(label)
# data = pd.concat([train, label]).drop_duplicates()
data = pd.read_csv('data/tomcat_dat/test_final.csv')
print(data)
# train = train.fillna(
#     0,  # nan的替换值
#     inplace=False  # 是否更换源文件
# )
cosine = []
# topic = []
semantic = []
for index, row in data.iterrows():
    print(row['BugId'])
    bug_text = row['Pre_BugReport']
    sourcefile_text = row['SourceFile_txt']
    print(bug_text)
    print(sourcefile_text)
    print(cosine_sim(bug_text, sourcefile_text))
    cosine.append(cosine_sim(bug_text, sourcefile_text))
#
#     doclist = []
#     doclist.append()
#     # doclist.append(bug_text)
#     topics = topic_sim(doclist, bug_text)
#     # print(topics)
#     # print(topics[0])
#     # topic.append(topics[0])

    print(semantic_sim(bug_text, sourcefile_text))
    semantic.append(semantic_sim(bug_text, sourcefile_text))
#
data['cosine_sim'] = cosine
# train['topic_sim'] = topic
data['semantic_sim'] = semantic
#
# print(data)
# data.to_csv("data/zookeeper_dat/res_final.csv")
data.to_csv("data/tomcat_dat/res_final.csv", index=False)