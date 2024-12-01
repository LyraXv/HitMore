import math

import pandas as pd
from configx.configx import ConfigX
from features.utils_features import simplify_string


def readGroundTruth(filepath,approach):
    res_pred = []
    res = open(filepath,'r')

    if approach == configx.approach[0]:
        for line in res:
            split_line = list(line.strip('\n').split('	'))
            split_line[1] = split_line[1].rstrip('.')
            res_pred.append(split_line)
        res.close()
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'path','rank','score'])
        res_df = res_df.drop(columns=['rank','score'])
    elif approach == configx.approach[1]:
        for line in res:
            split_line = list(line.strip('\n').split(','))
            if not split_line[0].isdigit():
                continue
            res_pred.append(split_line)
        res.close()
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'path', 'rank', 'score'])
        res_df = res_df.drop(columns=['rank', 'score'])
    else:
        for line in res:
            split_line = list(line.strip('\n').split(','))
            if not split_line[0].isdigit():
                continue
            res_pred.append(split_line)
        res.close()
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'rank', 'path'])
        res_df = res_df.drop(columns=['rank'])
    res_df['bugId'] = pd.to_numeric(res_df['bugId'])
    return res_df



def readRecLists_5(dataset): # five fold lists
    path = f"../data/splited_and_boosted_data/{dataset}/"
    file_list = [f"{path}{i}.csv" for i in range(5)]
    rec_lists = pd.concat([pd.read_csv(file,index_col=0) for file in file_list])
    return rec_lists

def readMergedLists(dataset):
    file_path= f"../data/get_info/{dataset}/mergedRecList.csv"
    df_mergedLists = pd.read_csv(file_path,index_col=0)
    return df_mergedLists

def mergeGTfile(dataset):
    file = configx.filepath_dict[dataset]
    # gt_amalgam_df = readGroundTruth(file['bugPredict'][0],configx.approach[0])
    gt_bugLocator_df = readGroundTruth(file['bugPredict'][1],configx.approach[1])
    gt_blizzard_df = readGroundTruth(file['bugPredict'][2],configx.approach[2])

    print("bugLocator truly buggy files:",gt_bugLocator_df.shape[0])
    print("blizzard truly buggy files:",gt_blizzard_df.shape[0])

    gt_bugLocator_df['simplified_path'] = gt_bugLocator_df['path'].apply(simplify_string)
    gt_blizzard_df['simplified_path'] = gt_blizzard_df['path'].apply(simplify_string)


    gt_bugLocator_df = gt_bugLocator_df.rename(columns={'path':'path_1'})
    gt_blizzard_df = gt_blizzard_df.rename(columns={'path':'path_2'})


    merge_rows = []
    gt_bugLocator_df['matched'] = False
    gt_blizzard_df['matched'] = False
    for _, row_2 in gt_blizzard_df.iterrows():
        simplified_path_2 = row_2['simplified_path']
        bugId_2 = row_2['bugId']
        # 在 df_1 中找到以 df_1['simplified_path'] 结尾的 df_0['simplified_path']
        for idx, row_1 in gt_bugLocator_df[gt_bugLocator_df['matched']==False].iterrows():
            simplified_path_1 = row_1['simplified_path']
            bugId_1 = row_1['bugId']
            if bugId_1 == bugId_2 and simplified_path_2.endswith(simplified_path_1):
                # 如果匹配成功，将这两行合并
                merged_row = {**row_2, **row_1}  # 合并两个字典
                merge_rows.append(merged_row)  # 添加到结果列表
                gt_bugLocator_df.at[idx, 'matched'] = True
                gt_blizzard_df.at[idx, 'matched'] = True
                break
    # 将合并的结果转为 DataFrame
    df_merged = pd.DataFrame(merge_rows)
    df_supplement = gt_bugLocator_df[gt_bugLocator_df['matched'] == False].drop(columns='matched')
    df_merged = pd.concat([df_merged,df_supplement])
    # df_supplement_2= gt_blizzard_df[gt_blizzard_df['matched'] == False].drop(columns='matched')
    # print(df_supplement_2)
    # exit()

    df_merged = df_merged.drop(columns='matched')
    df_merged = df_merged.reindex(columns=['bugId', 'rank_0', 'score_0', 'path_0', 'rank_1', 'path_1', 'score_1',
       'rank_2', 'path_2'], fill_value=math.nan)
    df_merged['label'] = 1
    return df_merged

def getOtherTrueFiles(rec_lists,mergedList,dataset):
    mergedList_filtered = mergedList[mergedList['label']==1]

    df_otherTrueFiles = mergedList_filtered.merge(rec_lists, on=['bugId', 'rank_0', 'score_0', 'path_0', 'rank_1', 'path_1', 'score_1',
       'rank_2', 'path_2', 'label'], how='left', indicator=True)
    df_otherTrueFiles = df_otherTrueFiles[df_otherTrueFiles['_merge'] == 'left_only'].drop(columns=['_merge'])

    return df_otherTrueFiles


def getOtherGTFiles(GT_lists,mergedList,dataset):

    mergedList_filtered = mergedList[mergedList['label']==1]
    # df_otherTrueFiles = GT_lists.merge(mergedList_filtered, on=['bugId', 'path_1','path_2'], how='left', indicator=True)
    # df_res = df_otherTrueFiles[df_otherTrueFiles['_merge'] == 'left_only'].drop(columns=['_merge'])
    matched_columns = ['bugId', 'path_1']
    # matched_columns = ['bugId', 'path_2']
    matched_rows = GT_lists.merge(mergedList_filtered, on=matched_columns, how='inner')
    # 从 df_A 中去除与 df_B 匹配的行
    df_res = GT_lists[~GT_lists.set_index(matched_columns).index.isin(
        matched_rows.set_index(matched_columns).index)]
    return df_res


if __name__ == "__main__":
    configx = ConfigX()
    dataset = 'zookeeper' #lucene(4003/3995) aspectj(980/975) ///hibernate(1344/1010),hibernate需要调整gt_blizzard

    rec_lists = readRecLists_5(dataset)
    buggy_rec_lists = rec_lists[rec_lists['label']==1]
    print(rec_lists.shape)
    len_recLists = rec_lists[rec_lists['label']==1].shape[0]
    print("recLists:",len_recLists)
    mergedLists = readMergedLists(dataset)
    len_mergedLists = mergedLists[mergedLists['label']==1].shape[0]
    print("margedLists :",len_mergedLists)
    GT_lists = mergeGTfile(dataset)
    print("GTlists:",GT_lists.shape[0])

    other_merged_files = getOtherTrueFiles(rec_lists,mergedLists,dataset)
    print(f"other_merged_files:{other_merged_files.shape},maybe:{len_mergedLists-len_recLists}")
    other_GT_files = getOtherGTFiles(GT_lists,mergedLists,dataset)
    other_GT_files = other_GT_files.drop_duplicates()
    print((f"other_GT_files: {other_GT_files.shape},maybe:{GT_lists.shape[0]-len_mergedLists}"))
    # exit()
    otherTrueFiles = pd.concat([other_merged_files,other_GT_files])
    otherTrueFiles = otherTrueFiles.drop_duplicates()
    otherTrueFiles = otherTrueFiles.reset_index(drop=True)
    all_truly_buggy_files = pd.concat([otherTrueFiles,buggy_rec_lists])
    duplicated_rows = all_truly_buggy_files[all_truly_buggy_files.duplicated(subset=['bugId', 'path_1'], keep=False)]
    print(otherTrueFiles.shape)
    print(duplicated_rows)
    # exit()
    otherTrueFiles.to_csv(f"../data/splited_and_boosted_data/{dataset}/otherTrulyBuggyFiles.csv")
    # all_truly_buggy_files.to_csv(f"../data/{dataset}-allTrulyBuggyFiles.csv")

