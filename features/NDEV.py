import math
import re
import gensim
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import cossim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk


from configx.configx import ConfigX
from features.utils_features import readRecList, updateFeatures, search_bugCmit, openCodeCorpus, searchCodeAndComments, \
    openNDEV, searchContributorsNum


def calculate_consistency_features(rec_lists):
    grouped = rec_lists.groupby('bugId')
    res_list = [] # output
    for bugId,group in grouped:
        # print("BUGID: ",bugId)
        NDEV_None=0
        bugCmit = search_bugCmit(bugId,dataset)
        NDEV = openNDEV(dataset,bugCmit)
        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            contributors = searchContributorsNum(NDEV,filepaths,bugCmit)
            if contributors is None:
                contributors = math.nan
                NDEV_None +=1
                # print(f"NDEV：BugId:{bugId},bugCmit:{bugCmit}\nfilepath:{filepaths}")
            res.append(index)
            res.append(bugId)
            res.append(contributors)
            res_list.append(res)
        if NDEV_None != 0:
            print(f"=====BugId:{bugId},CodeCorpus_None:{NDEV_None}")
    df_NDEV = pd.DataFrame(res_list,columns=['index','bugId','ndev'])

    # 与BuggyFileFeatures合并
    df_buggyFileFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_result = updateFeatures(df_buggyFileFeatures,df_NDEV)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)

if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: # all
            continue
        print(f"=====NDEV: {dataset}=====")
        for i in [0,1,2,3,4,'otherTrulyBuggyFiles']:
            if i !='otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            # with open("../data/splited_and_boosted_data/MultipleFilesPaths.txt", 'a') as f:
            #     f.write(f"=======Dataset: {dataset} ==== fold: {i}======\n")
            # f.close()
            rec_lists = readRecList(dataset,i)
            calculate_consistency_features(rec_lists)


