# merge the DAF and MAF data
import math

import pandas as pd

from configx.configx import ConfigX
from features.utils_features import readRecList, updateFeatures, search_bugCmit, openDAF_MAF, searchDAF_MAF


def mergeDAF_MAF(rec_lists):
    # 根据文件地址去匹配
    grouped = rec_lists.groupby('bugId')
    res_list = [] # output
    for bugId,group in grouped:
        # print("BUGID: ",bugId)
        DAF_MAF_None=0
        bugCmit = search_bugCmit(bugId,dataset)
        # print("bugCmit", bugCmit)
        DAF_MAF_all = openDAF_MAF(dataset,bugCmit)
        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            daf, maf = searchDAF_MAF(DAF_MAF_all,filepaths)
            if (daf is None) and (maf is None):
                daf,maf = math.nan,math.nan
                DAF_MAF_None +=1
                # print(f"未匹配到CodeCorpus文件：BugId:{bugId},bugCmit:{bugCmit}\nfilepath:{filepaths}")
            res.append(index)
            res.append(bugId)
            res.append(daf)
            res.append(maf)
            res_list.append(res)
        if DAF_MAF_None != 0:
            print(f"=====BugId:{bugId},DAF_MAF_None:{DAF_MAF_None}")
    df_DAF_MAF = pd.DataFrame(res_list,columns=['index','bugId','DAF','MAF'])
    # merge DAF_MAF with BuggyFileFeatures
    df_buggyFileFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_result = updateFeatures(df_buggyFileFeatures,df_DAF_MAF)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)

if __name__ == '__main__':
    configx = ConfigX()

    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====DAF and MAF: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            mergeDAF_MAF(readRecList(dataset,i))
