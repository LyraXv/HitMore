import math
import os

import pandas as pd

from configx.configx import ConfigX
from model.readMergedRecLists import calculate_hit_k


def average_precision(relevance_list):
    """计算一个查询的平均精度 (AP)"""
    precisions = []
    num_relevant = 0
    for i, rel in enumerate(relevance_list):
        if rel:  # 如果文档相关
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def mean_average_precision(relevance_lists):
    """计算多个查询的平均精度 (MAP)"""
    return sum(average_precision(r) for r in relevance_lists) / len(relevance_lists)


def reciprocal_rank(relevance_list):
    """计算一个查询的倒数排名 (RR)"""
    for i, rel in enumerate(relevance_list):
        if rel:  # 找到第一个相关文档
            return 1 / (i + 1)
    return 0.0


def mean_reciprocal_rank(relevance_lists):
    """计算多个查询的平均倒数排名 (MRR)"""
    Hit_1_sum = 0
    for r in relevance_lists:
        res = reciprocal_rank(r)
        if (res ==1):
            Hit_1_sum +=1
    print("Hit",Hit_1_sum)
    print(len(relevance_lists))
    return sum(reciprocal_rank(r) for r in relevance_lists) / len(relevance_lists)

def getHitMoreData(dataset):
    # 读取五折数据，pd.concat，然后按bug_id分成label list
    rec_list_data_path =[f"../data/hitmore_initial_rec_list/{dataset}/{i}.csv" for i in range(5)]
    # rec_list_data_path =[f"../data/splited_and_boosted_data/{dataset}/{i}.csv" for i in range(5)]
    rec_list_data = pd.concat([pd.read_csv(file) for file in rec_list_data_path])
    return rec_list_data

def getHitMoreRecListLabel(rec_list_data):
    rec_list_label = []
    grouped = rec_list_data.groupby('bugId')
    for bugid,group in grouped:
        group_label_list = list(group['label'])
        rec_list_label.append(group_label_list)
    return rec_list_label


def getOtherRecommendList(approach,dataset,file):
    filepath = file['bugFixedFileRanks'][approach]
    files = os.listdir(filepath)
    res_recommend = []

    for fi in files:
        # merge filepath and filename
        fi_d = os.path.join(filepath, fi)
        res = open(fi_d, 'r',encoding='utf-8')
        bugId = fi.split('.')[0]
        for line in res:
            if approach == 0:
                split_line = list(line.strip('\n').split('	'))
                split_line[2] = split_line[2].rstrip('.')
                split_line = [bugId] + split_line
            else:  # bugLocator
                split_line = list(line.strip('\n').split(','))
            if split_line[1].isdigit():
                res_recommend.append(split_line)
        res.close()

    gt = getOtherApproachBuggyData(file,approach)
    gt['label'] = 1

    if(approach == 2):
        # bilizard without Score
        dataframe = pd.DataFrame(res_recommend, columns=['bugId', 'rank', 'path'])
        dataframe['rank'] = dataframe['rank'].astype(int)-1
        dataframe['bugId'] = dataframe['bugId'].astype('str')
        dataframe['rank'] = dataframe['rank'].astype('int64')

        gt['bugId'] = gt['bugId'].astype('str')
        gt['rank'] = gt['rank'].astype('int64')

        df = pd.merge(dataframe, gt, on=['bugId', 'rank','path'], how='left')
    elif(approach == 0):
        dataframe = pd.DataFrame(res_recommend, columns=['bugId', 'rank', 'score', 'path'])
        dataframe = dataframe.reindex(columns =['bugId', 'rank', 'path', 'score'])
        df = pd.merge(dataframe, gt, on=['bugId', 'rank','path'], how='left')
    else:
        dataframe = pd.DataFrame(res_recommend, columns=['bugId', 'rank', 'path', 'score'])
        df = pd.merge(dataframe, gt, on=['bugId', 'rank','path'], how='left')
    df.fillna(0, inplace=True)
    df = df.astype({"label": int})
    return df

def getOtherApproachBuggyData(file,i): # i = {0:amalgam,1:bugLocator,2:blizzard}
    print(f"=====approaches: {i}======")
    res_pred = []
    res = open(file['bugPredict'][i],'r')
    for line in res:
        if i == 0:
            split_line = list(line.strip('\n').split('	'))
            split_line[1] = split_line[1].rstrip('.')
        else: # bugLocator
            split_line = list(line.strip('\n').split(','))
        if split_line[0].isdigit():
            res_pred.append(split_line)
    res.close()
    if i ==2:
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'rank', 'path'])
        res_df['rank']= res_df['rank'].astype(int)
        res_df['rank'] = res_df['rank'] -1
    else:
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'path', 'rank', 'score'])
        res_df = res_df.drop(columns=['score'])
    return res_df

# def getOtherRecListLabel(buggydata):
#     rec_list_label = []
#     grouped = buggydata.groupby('bugId')
#     for bugId,group in grouped:
#         group_label_list = [0]*20
#         for index,row in group.iterrows():
#             rank = int(row['rank'])
#             if rank<20:
#                 group_label_list[rank] = 1
#         rec_list_label.append(group_label_list)
#     rec_list_len = len(rec_list_label)
#     if rec_list_len < bugid_num: # bug_id数量
#         print(f"real-rec-len:{rec_list_len}")
#         group_label_list = [0] * 20
#         for _ in range(bugid_num-rec_list_len):
#             rec_list_label.append(group_label_list)
#     return rec_list_label

def HitCount(data,df_fixed_files):
    HitCount_1 = 0
    HitCount_2 = 0
    HitCount_all = 0
    HitCount_mult = 0

    grouped = data.groupby('bugId')
    for bugid, group in grouped:
        fixed_files = df_fixed_files[df_fixed_files['bugId']==int(bugid)]
        # print(fixed_files)
        top_20_group = group.iloc[0:20,:]
        buggy_file_count = (top_20_group['label']==1).sum()
        if buggy_file_count >= 1:
            HitCount_1 +=1
        if buggy_file_count >= 2:
            HitCount_2 +=1
        # print(fixed_files['file_count'])
        if buggy_file_count == fixed_files['file_count'].values[0]:
            # print(f"bugId:{bugid}filecount{buggy_file_count},fixed_file:{fixed_files['file_count'].values[0]}" )
            HitCount_all +=1
            if buggy_file_count>1:
                HitCount_mult +=1
        # print("bugId",bugid,"buggy_file_count", buggy_file_count,"fixed_files",fixed_files['file_count'].values)
        if buggy_file_count>fixed_files['file_count'].values[0]:
            print("Attention!!!",top_20_group[top_20_group['label']==1])
            print(fixed_files['files'])
    print(f"HitCount@1: {HitCount_1}")
    print(f"HitCount@2: {HitCount_2}")
    print(f"HitCount@All: {HitCount_all}")
    print(f"HitCount@Mult: {HitCount_mult}")

    return HitCount_1,HitCount_2,HitCount_all,HitCount_mult


def TotalTrulyBuggyFilesNum(dataset):
    path = f"../data/get_info/{dataset}/bug_report_fixedfiles.csv"
    df_fixedFiles = pd.read_csv(path,encoding="utf-8-sig")
    df_fixedFiles = df_fixedFiles[['bug_id','files']]
    df_fixedFiles = df_fixedFiles.rename(columns={'bug_id':'bugId'})
    df_fixedFiles['bugId']=df_fixedFiles['bugId'].str.extract('(\d+)').astype(int)

    df_fixedFiles['file_count'] = df_fixedFiles['files'].str.split(r'\.java').apply(lambda x: len(x) - 1)
    # print(df_fixedFiles['files'].str.split(r'\.java')[0])
    return df_fixedFiles

def mutiple_files_bug(df_fixed_files,bug_id):
    print("bug_list_len: ",len(bug_id))
    filterd_df = df_fixed_files[df_fixed_files['bugId'].isin(bug_id)]
    filterd_df = filterd_df[filterd_df['file_count']>1]
    print(filterd_df.shape)


if __name__ == '__main__':
    configx = ConfigX()

    pd.set_option('display.max_rows', None)  # 显示全部行
    pd.set_option('display.max_columns', None)  # 显示全部列
    res_locating_performance = []
    for dataset, file in configx.filepath_dict.items():
        # if dataset not in ['zookeeper']:
        #     continue
        print(f"============dataset:{dataset}===========")
        # 获取完整的TrulyBuggyFilesNum
        df_fixed_files = TotalTrulyBuggyFilesNum(dataset)
        # # df_fixed_files['bugId'] = df_fixed_files['bugId'].str.extract('(\d+)').astype(int)
        # mult_num = df_fixed_files[df_fixed_files['file_count']>1].shape[0]
        # print("MultipleBuggyFiles_bugNum:",mult_num)
        # # 计算 Initial_Rec_List_Label
        # HitMore_data = getHitMoreData(dataset)
        # rec_list_label = getHitMoreRecListLabel(HitMore_data)
        # # print(rec_list_label)
        # print("HitMore_len",rec_list_label.__len__())
        # bugid_num = rec_list_label.__len__()
        # print(rec_list_label)
        # # exit()
        # map_score = mean_average_precision(rec_list_label)
        # mrr_score = mean_reciprocal_rank(rec_list_label)
        # print(f"=====approaches: HitMore======")
        # print(f"MAP: {map_score}")
        # print(f"MRR: {mrr_score}")
        # HitCount(HitMore_data,df_fixed_files)
        #
        # HitMore_data['rank'] = HitMore_data.groupby('bugId').cumcount()
        # Hit_buggy_data = HitMore_data[HitMore_data['label']==1]
        # # print(HitMore_data.head())
        # # exit()
        # for k in [1, 5, 10, 15, 20]:
        #     calculate_hit_k(Hit_buggy_data,k,dataset)

        # other three techniques Performance
        fold = 4
        for i in range(3):
            data = getOtherRecommendList(i,dataset,file)

            # time
            bug_id_info = pd.read_csv(f"../data/ordered_bugCmit/ordered_bugCmit_{dataset}_time", header=None)
            bug_id_info.columns = ['bugId', 'commit', 'time']
            bug_id = bug_id_info['bugId'].tolist()

            # fold
            # bug_id_info = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/{fold}.csv")
            # bug_id = set(bug_id_info['bugId'].tolist())

            # mutiple_files_bug(df_fixed_files,bug_id)
            # break
            bug_list = [str(i) for i in bug_id]
            data = data[data['bugId'].isin(bug_list)]
            # print(data.shape)
            # print(data.columns)
            # print(type(data))
            # print(len(data.groupby('bugId')))
            rec_list_label = getHitMoreRecListLabel(data)
            print(i)
            # print(rec_list_label)
            # print(rec_list_label)
            print(rec_list_label.__len__())
            map_score = mean_average_precision(rec_list_label)
            mrr_score = mean_reciprocal_rank(rec_list_label)
            print(f"MAP: {map_score}")
            print(f"MRR: {mrr_score}")
            HitCount_1,HitCount_2,HitCount_all,HitCount_mult = HitCount(data,df_fixed_files)
            res_locating_performance.append([
                dataset,
                HitCount_1,
                HitCount_2,
                HitCount_all,
                HitCount_mult,
                '{:.2f}'.format(map_score),
                '{:.2f}'.format(mrr_score)
            ])
    # df_locating_performace = pd.DataFrame(data=res_locating_performance, index=None,
    #                                       columns=['dataset', 'HitCount@1','HitCount_2','HitCount_all','HitCount_mult', 'MAP', 'MRR'])
    # df_locating_performace.to_csv(f"../data/results/three_technique_HitCount_{fold}.csv")
