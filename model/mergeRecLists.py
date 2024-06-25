import math
import pickle

import numpy as np
import pandas as pd
import re
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from configx.configx import ConfigX
from utils import utils

def getBugFileInfo(filepath,approach):
    df = pd.read_csv(filepath)
    df['Approach'] = approach
    return df

def compute_union_with_all_fields(bugid, df1, df2, df3):
    group1 = df1[df1['BugId'] == bugid]
    group2 = df2[df2['BugId'] == bugid]
    group3 = df3[df3['BugId'] == bugid]

    union_filenames = pd.concat([group1, group2, group3]).drop_duplicates(subset=['SourceFile'])
    return union_filenames.to_dict('records')

def getBugFileLists(df_amalgam,df_bugLocator,df_blizzard):
    # get all bug ids
    all_bug_ids = set(df_amalgam['BugId']).union(set(df_bugLocator['BugId']), set(df_blizzard['BugId']))
    # calculate the file union
    results = []
    for bugid in all_bug_ids:
        union = compute_union_with_all_fields(bugid, df_amalgam, df_bugLocator, df_blizzard)
        results.extend(union)
    union_df = pd.DataFrame(results)
    return union_df
    # # data view
    # grouped = union_df.groupby('BugId')
    # for name,group in grouped:
    #     print("Group Id:",name,"total:",group.shape[0])
    #     print(group)

def read_txt_to_df(filepath,approach):
    try:
        if approach == 'blizzard':
            df = pd.read_csv(filepath, sep=",", names=['BugId', 'Rank', 'SourceFile'])
        elif approach == 'bugLocator':
            df = pd.read_csv(filepath, sep=",", names=['BugId', 'Rank', 'SourceFile','Score'])
        elif approach == 'amalgam':
            df = pd.read_csv(filepath, sep=",", names=['BugId','Rank', 'Score','SourceFile'])
        return df
    except FileNotFoundError as e:
        missFileInfo = approach + filepath
        if(missFileInfo not in configx.missRecFile):
            configx.missRecFile.add(missFileInfo)
            with open("../data/get_info/missTxtInfo.txt", 'a') as f:
                f.write(f"{missFileInfo}\n")
        df = pd.DataFrame(columns=['BugId', 'Rank', 'SourceFile', 'Score'])
        return df

def searchFileInList(df,filename):
    res = df[df['SourceFile'] == filename]
    # print(f"{filename}")
    if res.empty:
        # print(f"NotSearchInList: filename:{filename},df:{not df.empty}")
        return None
    else:
        return res.copy()

# search the recommended info from Approach Lists
def searchOtherApproachInfo(approach,dataset,bugId,filename):
    df_res = pd.DataFrame(columns=['BugId', 'Rank', 'SourceFile', 'Score','Approach'])
    if approach != configx.approach[0]:
        filepath = "../data/initial_recommendation_data/amalgam_new/" + dataset + "/" + str(
            bugId) + ".txt"
        df_list = read_txt_to_df(filepath, 'amalgam')
        fileInfo = searchFileInList(df_list,filename)
        if fileInfo is None:
            # print(filepath)
            none_info = {'BugId':bugId,'Rank':np.NaN,'Score':np.NaN,'Approach':configx.approach[0]}
            none_file = pd.DataFrame([none_info])
            df_res = pd.concat([df_res, none_file], ignore_index=True)
        else:
            fileInfo.loc[:,'Approach'] = configx.approach[0]
            df_res = pd.concat([df_res,fileInfo],ignore_index=True)

    if approach != configx.approach[1]:
        filepath = "../data/initial_recommendation_data/bugLocator_new/" + dataset + "/" + str(
            bugId) + ".txt"
        df_list = read_txt_to_df(filepath, 'bugLocator')
        fileInfo = searchFileInList(df_list,filename)
        # print(filename,",dataset",dataset,",approach:",configx.approach[1])
        if fileInfo is None:
            # print(filepath)
            none_info = {'BugId':bugId,'Rank':np.NaN,'Score':np.NaN,'Approach':configx.approach[1]}
            none_file = pd.DataFrame([none_info])
            df_res = pd.concat([df_res, none_file], ignore_index=True)
        else:
            fileInfo.loc[:,'Approach'] = configx.approach[1]
            df_res = pd.concat([df_res,fileInfo],ignore_index=True)

    if approach != configx.approach[2]:
        filepath = "../data/initial_recommendation_data/blizzard_new/" + dataset + "/" + str(
            bugId) + ".txt"
        df_list = read_txt_to_df(filepath, 'blizzard')
        fileInfo = searchFileInList(df_list,filename)
        if fileInfo is None:
            # print(filepath)
            none_info = {'BugId':bugId,'Rank':np.NaN,'Score':np.NaN,'Approach':configx.approach[2]}
            none_file = pd.DataFrame([none_info])
            df_res = pd.concat([df_res, none_file], ignore_index=True)
        else:
            fileInfo.loc[:,'Approach'] = configx.approach[2]
            df_res = pd.concat([df_res,fileInfo],ignore_index=True)

    return df_res

def updateRecInfo(bugRecInfo , row):
    if row['Approach'] == configx.approach[0]:
        bugRecInfo['Rank_0'] = row['Rank']
        bugRecInfo['Score_0'] = row['Score']
    elif row['Approach'] == configx.approach[1]:
        bugRecInfo['Rank_1'] = row['Rank']
        bugRecInfo['Score_1'] = row['Score']
    elif row['Approach'] == configx.approach[2]:
        bugRecInfo['Rank_2'] = row['Rank']
    return bugRecInfo

def check_for_nan(data):
    ranks = ['Rank_0', 'Rank_1', 'Rank_2']
    for rank in ranks:
        if math.isnan(data[rank]):
            return True
    return False

def mergeBugFileInfo(filelists,dataset):
    print(">>>>>Merge Bug Info:","dataset:",dataset,"<<<<<")
    exsit_nan_row = 0
    df_merge = pd.DataFrame(columns=['BugId','SourceFile','Rank_0','Score_0','Rank_1','Score_1','Rank_2','label'])
    grouped = filelists.groupby('BugId')
    # print(filelists) # union file lists
    for bugId,group in grouped:
        # load txt
        for index,row in group.iterrows():
            # print(f"Present Info{row.values}")
            bugRecInfo = {'BugId':bugId,'label':row['label'],'SourceFile':row['SourceFile']}
            bugRecInfo = updateRecInfo(bugRecInfo, row)
            otherApproachInfo = searchOtherApproachInfo(row['Approach'], dataset, bugId, row['SourceFile'])
            for i, rowx in otherApproachInfo.iterrows():
                bugRecInfo = updateRecInfo(bugRecInfo,rowx)
            if check_for_nan(bugRecInfo):
                exsit_nan_row += 1
            df_merge = pd.concat([df_merge,pd.DataFrame([bugRecInfo])],ignore_index=True)
    df_merge = df_merge.dropna()
    df_merge.to_csv("../data/get_info/"+dataset+'/recommendedList2.csv')
    with open("../data/get_info/note_RecLists.txt", 'a') as f:
        f.write(f"Dataset:{dataset}  ExistNan Rows: {exsit_nan_row} Total Rows: {filelists.shape[0]} missRatio: {exsit_nan_row/filelists.shape[0]}\n")


if __name__ == "__main__":
    configx = ConfigX()

    # setView
    pd.set_option('display.expand_frame_repr', False)

    initial_path = "../data/get_info/"
    # dataset
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        df_amalgam = getBugFileInfo(initial_path + dataset + "/" + configx.approach[0]+"List_top20.csv",configx.approach[0])
        df_bugLocator = getBugFileInfo(initial_path + dataset + "/" + configx.approach[1] + "List_top20.csv",configx.approach[1])
        df_blizzard = getBugFileInfo(initial_path + dataset + "/" + configx.approach[2] + "List_top20.csv",configx.approach[2])

        recommendedLists = getBugFileLists(df_amalgam,df_bugLocator,df_blizzard)
        mergeBugFileInfo(recommendedLists,dataset)

