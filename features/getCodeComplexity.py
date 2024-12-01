import io
import math
import pandas as pd
from configx.configx import ConfigX
from features.utils_features import readRecList, updateFeatures, search_bugCmit, searchCCdata

def readCCinfo(bugCmit):
    filePath = f"D:/Dataset/CK_Metrics/{dataset}/{bugCmit}class.csv"
    try:
        df_CC = pd.read_csv(filePath,encoding='utf-8')
    except UnicodeDecodeError:
        with open(filePath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        df_CC = pd.read_csv(io.StringIO(content))
    return df_CC


def getCodeComplexity(rec_lists):
    grouped = rec_lists.groupby('bugId')
    df_res = pd.DataFrame()
    for bugId, group in grouped:
        bugCmit = search_bugCmit(bugId, dataset)
        CC_data = readCCinfo(bugCmit)
        CC_None = 0
        for index, file in group.iterrows():
            filepaths = {'path_0': file['path_0'],
                         'path_1': file['path_1'],
                         'path_2': file['path_2']}
            df_CC = searchCCdata(CC_data, filepaths)
            if index == 2:
                df_CC = None
            if df_CC is None:
                CC_None +=1
            else:
                df_CC['index'] = index
                df_CC['bugId'] = bugId
                df_CC['bugFixDependencies'] = df_CC['fanin']+df_CC['fanout']
                df_res = pd.concat([df_res,df_CC])
        if CC_None != 0:
            print(f"=====BugId:{bugId},CodeCorpus_None:{CC_None}")
    df_bfFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_bfFeature_result = updateFeatures(df_bfFeatures,df_res)
    df_bfFeature_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)

if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====six features: {dataset}=====") #CC/fix denpendency
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            getCodeComplexity(readRecList(dataset,i))
            # exit()

