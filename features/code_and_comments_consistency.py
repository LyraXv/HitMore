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

# def code_and_comments_consistency(code,comments):
#     # similarity = topic_sim([code,comments], comments)
#     similarity = topic_similarity(preprocess(code),preprocess(comments))
#     return format(similarity,'.4f')
def getCodeAndComments(CodeCorpus):
    code_comments = []
    for index,(code,comments) in CodeCorpus[['Code','Comments']].iterrows():
        code_comments.append(code)
        code_comments.append(comments)
    return code_comments

def calculate_consistency_features(rec_lists):
    grouped = rec_lists.groupby('bugId')
    res_list = [] # output
    for bugId,group in grouped:
        print("BUGID: ",bugId)
        CodeCorpus_None=0
        bugCmit = search_bugCmit(bugId,dataset)
        CodeCorpus = openCodeCorpus(dataset,bugCmit)
        # 训练LDA模型
        lda_train_data = getCodeAndComments(CodeCorpus)
        # topic_num_test(lda_train_data) # adjust topic_num
        lda_model, dic = trainLDA(lda_train_data,5)

        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            code, comments = searchCodeAndComments(CodeCorpus,filepaths)
            if (code is None) and (comments is None):
                similarity = math.nan
                CodeCorpus_None +=1
                # print(f"未匹配到CodeCorpus文件：BugId:{bugId},bugCmit:{bugCmit}\nfilepath:{filepaths}")
            elif pd.isna(comments):
                similarity = 0
            else:
                similarity = topic_similarity(code,comments,lda_model,dic)

            res.append(index)
            res.append(bugId)
            res.append(similarity)
            res_list.append(res)
        if CodeCorpus_None != 0:
            print(f"=====BugId:{bugId},CodeCorpus_None:{CodeCorpus_None}")
    df_similarity = pd.DataFrame(res_list,columns=['index','bugId','commentsCodesConsistency'])
    # 与BuggyFileFeatures合并
    df_buggyFileFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_result = updateFeatures(df_buggyFileFeatures,df_similarity)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)

if __name__ == '__main__':
    configx = ConfigX()

    print(nltk.data.path)
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: # lucene
            continue
        print(f"=====Code and Comments Consisitency: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            rec_lists = readRecList(dataset,i)
            calculate_consistency_features(rec_lists)


