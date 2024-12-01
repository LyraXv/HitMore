import math
import re
import gensim
import numpy as np
import pandas as pd
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from configx.configx import ConfigX
from features.similarity import topic_similarity, preprocess, topic_num_test, trainLDA
from features.utils_features import readRecList, updateFeatures, search_bugCmit, openCodeCorpus, searchCodeAndComments
from features import XMLToDictionary as XD


# def code_and_comments_consistency(code,comments):
#     # similarity = topic_sim([code,comments], comments)
#     similarity = topic_similarity(preprocess(code),preprocess(comments))
#     return format(similarity,'.4f')
def getCodeAndComments(CodeCorpus):
    code_comments = []
    for index,(code,comments) in CodeCorpus[['Code','Comments']].iterrows():
        if pd.isna(comments):
            text = code
        elif pd.isna(code):
            continue
        else:
            text = code + " " + comments
        code_comments.append(text)
    return code_comments

def calculate_topic_similarity(rec_lists):
    allBugReports = XD.CSVToDictionary(dataset)
    # print(allBugReports)
    br_data =[]
    for item in allBugReports:
        br_data.append(item['rawCorpus'])
    grouped = rec_lists.groupby('bugId')
    res_list = [] # output
    for bugId,group in grouped:
        # print("BUGID: ",bugId)
        CodeCorpus_None=0
        [report] = list(filter(lambda x: x['bug_id'] == str(bugId), allBugReports))
        br = report["rawCorpus"]
        bugCmit = search_bugCmit(bugId,dataset)
        CodeCorpus = openCodeCorpus(dataset,bugCmit)
        # 训练LDA模型
        cf_data = getCodeAndComments(CodeCorpus)
        lda_train_data =br_data + cf_data
        # topic_num_test(lda_train_data) # adjust topic_num
        # exit()
        lda_model, dic = trainLDA(lda_train_data,num_topics)

        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            code, comments = searchCodeAndComments(CodeCorpus,filepaths)
            if (code is None) and (comments is None):
                similarity = 0
                CodeCorpus_None +=1
                # print(f"未匹配到CodeCorpus文件：BugId:{bugId},bugCmit:{bugCmit}\nfilepath:{filepaths}")
            else:
                if pd.isna(comments):
                    cf = code
                else:
                    cf = code + " " + comments
                similarity = topic_similarity(br,cf,lda_model,dic)

            res.append(index)
            res.append(bugId)
            res.append(similarity)
            res_list.append(res)
        if CodeCorpus_None != 0:
            print(f"=====BugId:{bugId},CodeCorpus_None:{CodeCorpus_None}")
    df_similarity = pd.DataFrame(res_list,columns=['index','bugId','topicSimilarity'])
    # 与BuggyFileFeatures合并
    # merge with RelationFeatures
    df_relationFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv")
    df_result = updateFeatures(df_relationFeatures,df_similarity)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv",index=False)

if __name__ == '__main__':
    configx = ConfigX()
    num_topic ={'zookeeper':15,'hibernate':20,'Tomcat':15,'openjpa':15,'aspectj':15,'lucene':25}
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: # zookeeper 15
            continue
        num_topics = num_topic[dataset]
        print(f"=====Code and Comments Consisitency: {dataset}=====")
        for i in [0,1,2,3,4,'otherTrulyBuggyFiles']:
            if i !='otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            rec_lists = readRecList(dataset,i)
            calculate_topic_similarity(rec_lists)


