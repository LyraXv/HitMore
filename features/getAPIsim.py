import pandas as pd
from configx.configx import ConfigX
from features.utils_features import readRecList, searchLRdata, updateFeatures


def readLRinfo(filePath):
    res_list =[]
    with open(filePath, 'r', encoding='utf-8') as file:
        for line in file:
             columns = line.strip().split(':')
             file = columns[3]
             value = float(columns[5])
             res_list.append([file,value])
    df_lr = pd.DataFrame(res_list,columns=['file','values'])
    return df_lr


def getLRdata(rec_lists):
    grouped = rec_lists.groupby('bugId')
    res_list = []  # output
    for bugId, group in grouped:
        filePath = f"../../LR/{dataset}/{bugId}_6features.res"
        LR_data = readLRinfo(filePath)
        Contributors_None = 0
        for index, file in group.iterrows():
            res = []
            filepaths = {'path_0': file['path_0'],
                         'path_1': file['path_1'],
                         'path_2': file['path_2']}
            value = searchLRdata(LR_data, filepaths)
            if value is None:
                apiEnrichedLexicalSimilarity = 0
            else:
                apiEnrichedLexicalSimilarity = value

            res.append(index)
            res.append(bugId)
            res.append(apiEnrichedLexicalSimilarity)
            res_list.append(res)
        if Contributors_None != 0:
            print(f"=====BugId:{bugId},CodeCorpus_None:{Contributors_None}")
    df_api = pd.DataFrame(res_list,columns=['index','bugId',"apiEnrichedLexicalSimilarity"])
    df_relationFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv")
    df_realtionFeature_result = updateFeatures(df_relationFeatures,df_api)
    df_realtionFeature_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv",index=False)

if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====API similarity: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            getLRdata(readRecList(dataset,i))

