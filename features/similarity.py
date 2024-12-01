import math
import os
import sys
import gc
import random
import string
import re
from pprint import pprint

import pandas as pd
from gensim.similarities import MatrixSimilarity
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from sympy.physics.control.control_plots import matplotlib

from configx.configx import ConfigX
from features import XMLToDictionary as XD
import numpy as np
from scipy import spatial


# 去茎化
from features.utils_features import searchCodeAndComments, search_bugCmit, readRecList, updateFeatures
from readability.utils_rdFiles import openCodeCorpus


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
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(text)
    sim = (tfidf * tfidf.T).A[0, 1]
    return sim

def cosine_sim_2(text1, text2):
    # vectorizer = TfidfVectorizer(tokenizer=normalize, min_df=1, stop_words="english")
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = (tfidf * tfidf.T).A[0, 1]
    return sim
#
#
# def topic_sim(doclist, query):
#     texts = []
#     for doc in doclist:
#         texts += [preprocess(doc)]
#     # 抽取一个"词袋（bag-of-words)"，将文档的token映射为id
#     dictionary = corpora.Dictionary(texts)
#     # print(dictionary)
#     # print(dictionary.token2id)
#     # 将用字符串表示的文档转换为用id表示的文档向量
#     corpus = [dictionary.doc2bow(text) for text in texts]
#     # print("corpus: ", corpus)
#     # 计算tf-idf模型
#     tfidf = models.TfidfModel(corpus)
#     corpus_tfidf = tfidf[corpus]
#     # 跑lda模型
#     lda = models.LdaModel(corpus, id2word=dictionary, num_topics=5, random_state=42)
#     # lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
#     # print(lda.print_topics(2))
#
#     # 将文档映射到一个二维的topic空间中
#     corpus_lda = lda[corpus_tfidf]
#     # for doc in corpus_lda:
#     #     print(doc)
#
#     # 对文档建立索引
#     index = similarities.MatrixSimilarity(lda[corpus_tfidf])
#     # 将query向量化
#     query_bow = dictionary.doc2bow(preprocess(query))
#     # print(query_bow)
#     # 用之前训练好的LDA模型将query映射到二维的topic空间
#     query_lda = lda[query_bow]
#     # print(query_lda)
#
#     # 计算query和index中doc的余弦相似度
#     lda_sims = index[query_lda]
#     # print(lda_sims)
#     # print(list(enumerate(lda_sims)))
#     return lda_sims[0]


def semantic_sim(text1, text2):
    # pip install pyemd

    # calculate distance between two sentences using WMD algorithm - Word Mover距离算法
    distance = model.wmdistance(text1, text2)
    # print(distance)
    return distance

# 数据预处理
def preprocess(text):
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    # 分词
    #     words = word_tokenize(text)
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    words = text.translate(replace_punctuation)

    words = [s.strip() for s in words.split(" ") if s]

    processed_words = []

    for word in words:
        # 去停止词
        if word.isalnum() not in stop_words and word.lower() not in stop_words:

            # 分割驼峰命名法单词
            split_words = re.sub(r'([a-z])([A-Z])', r'\1 \2', word).split()

            for split_word in split_words:
                # 小写 词性还原
                lemmatized_word = lemmatizer.lemmatize(split_word.lower())
                processed_words.append(lemmatized_word)
    return processed_words

# def topic_similarity(predoc1,predoc2):
#         texts = []
#         texts.append(predoc1)
#         texts.append(predoc2)
#
#         # 构建字典和文档-词频矩阵
#         dictionary = corpora.Dictionary(texts)
#         corpus = [dictionary.doc2bow(text) for text in texts]
#
#         # 训练LDA模型
#         lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=5, random_state=42)
#         print("1 ",dictionary.doc2bow(predoc1))
#         print("2 ",dictionary.doc2bow(predoc2))
#         doc1_topics = lda_model[dictionary.doc2bow(predoc1)]
#         doc2_topics = lda_model[dictionary.doc2bow(predoc2)]
#         print(f"doc1_topics: {doc1_topics}, doc2_topics:{doc2_topics})")
#         similarity = cosine_similarity(doc1_topics, doc2_topics)[0][0]
#         # similarity = cosine_similarity(doc1_topics, doc2_topics)
#         # index = MatrixSimilarity([doc1_topics], num_features=len(dictionary))
#         # similarity = index[doc2_topics][0]
#         return similarity

# LDA TEST
def trainLDA(data,num_topics):
    preprocessed_data = []
    # for code, comments in data:
    #     if pd.notna(code):
    #         preprocessed_code = preprocess(code)
    #         preprocessed_data.append(preprocessed_code)
    #     if pd.notna(comments):
    #         preprocessed_comments = preprocess(comments)
    #         preprocessed_data.append(preprocessed_comments)
    for text in data:
        if pd.notna(text):
            preprocessed_code = preprocess(text)
            preprocessed_data.append(preprocessed_code)

    dictionary = corpora.Dictionary(preprocessed_data)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_data]
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=10, random_state=42)
    # pprint(lda_model.top_topics(corpus))
    return lda_model,dictionary


def topic_similarity(code,comments,lda_model,dictionary):
    predoc1 = preprocess(code)
    predoc2 = preprocess(comments)

    doc1_topics = lda_model[dictionary.doc2bow(predoc1)]
    doc2_topics = lda_model[dictionary.doc2bow(predoc2)]

    index = similarities.MatrixSimilarity([doc1_topics], num_features=len(dictionary))
    similarity = index[doc2_topics][0]

    return similarity

def lda_model_values(num_topics,corpus,dictionary,texts):
    x = []  # x轴
    perplexity_values = []  # 困惑度
    coherence_values = []  # 一致性
    model_list = []  # 存储对应主题数量下的lda模型,便于生成可视化网页

    for topic in range(0,num_topics,5):
        if topic > 0:
            topic-=1
        print("主题数量：", topic + 1)
        lda_model = models.LdaModel(corpus=corpus, num_topics=topic + 1, id2word=dictionary, chunksize=2000, passes=20,
                                    iterations=400)
        model_list.append(lda_model)
        x.append(topic + 1)
        perplexity_values.append(lda_model.log_perplexity(corpus))

        coherencemodel = models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("该主题评价完成\n")
    return model_list, x, perplexity_values, coherence_values

def topic_num_test(data):
    preprocessed_data = []
    for text in data:
        if pd.notna(text):
            preprocessed_code = preprocess(text)
            preprocessed_data.append(preprocessed_code)

    dictionary = corpora.Dictionary(preprocessed_data)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_data]

    # 调用准备函数
    num_topics = 50
    model_list, x, perplexity_values, coherence_values = lda_model_values(num_topics, corpus, dictionary,preprocessed_data)

    # 绘制困惑度和一致性折线图
    fig = plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(x, perplexity_values, marker="o")
    plt.title("主题建模-困惑度")
    plt.xlabel('主题数目')
    plt.ylabel('困惑度大小')
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))  # 保证x轴刻度为1

    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(x, coherence_values, marker="o")
    plt.title("主题建模-一致性")
    plt.xlabel("主题数目")
    plt.ylabel("一致性大小")
    xticks(np.linspace(1, num_topics, num_topics, endpoint=True))

    plt.show()


def get_features_similarity(recList):
    allBugReports = XD.CSVToDictionary(dataset)
    res_list = []
    grouped = recList.groupby('bugId')
    for bugid, group in grouped:
        [report] = list(filter(lambda x: x['bug_id'] == str(bugid), allBugReports))
        br = report["rawCorpus"]
        bugCmit = search_bugCmit(bugid,dataset)
        CodeCorpus = openCodeCorpus(dataset,bugCmit)
        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            # bugFile
            code, comments = searchCodeAndComments(CodeCorpus,filepaths)
            if (code is None) and (comments is None):
                surfaceLexicalSimilarity = 0
                semanticSimilarity = 0

            else:
                if pd.isna(comments):
                    cf = code
                else:
                    cf = code +" "+comments

                pre_br = preprocess(br)
                pre_cf = preprocess(cf)
                str_br = " ".join(pre_br)
                str_cf = " ".join(pre_cf)
                surfaceLexicalSimilarity = cosine_sim([str_br,str_cf])
                semanticSimilarity = semantic_sim(str_br,str_cf)

            # topicSimilarity = topic_similarity(pre_br,pre_cf)

            res.append(index)
            res.append(bugid)
            res.append(surfaceLexicalSimilarity)
            res.append(semanticSimilarity)
            res_list.append(res)
    df_similarity = pd.DataFrame(res_list,columns=['index','bugId',
                                                   'surfaceLexicalSimilarity','semanticSimilarity'])
    # merge with RelationFeatures
    df_relationFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv")
    df_result = updateFeatures(df_relationFeatures,df_similarity)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv",index=False)

if __name__ == "__main__":
    configx = ConfigX()
    model = models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    for dataset, file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====Features_Similarity: {dataset}=====") # semanticSim/surfaveLexicalSim
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            get_features_similarity(readRecList(dataset, i))
