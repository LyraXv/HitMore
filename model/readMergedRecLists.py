'''
    From data/initial_recommendation_data/merge_data_withoutGT
    Label the data
'''
import pandas as pd

from configx.configx import ConfigX


def check_paths(row,df0,df1,df2):
    bugid = row['bugId']
    path_0 = row['path_0']
    path_1 = row['path_1']
    path_2 = row['path_2']

    if ((df0[(df0['bugId'] == bugid) & (df0['path'] == path_0)].shape[0] > 0) or
            (df1[(df1['bugId'] == bugid) & (df1['path'] == path_1)].shape[0] > 0) or
            (df2[(df2['bugId'] == bugid) & (df2['path'] == path_2)].shape[0] > 0)):
        return 1
    else:
        return 0

    # if (row['path_0'] in df0['path'].values) or (row['path_1'] in df1['path'].values) or (
    #         row['path_2'] in df2['path'].values):
    #     return 1
    # else:
    #     return 0

def readMergedTxtWithoutGT(dataset):
    filepath = f"..\data\initial_recommendation_data\merge_data_without_GT\{dataset}.merge.txt"
    df = pd.read_csv(filepath,sep=',',names=['bugId','rank_0','score_0','path_0','id_1','rank_1','path_1','score_1','id_2','rank_2','path_2'])
    df = df.drop(columns=['id_1','id_2'])
    df['rank_2'] = pd.to_numeric(df['rank_2'])-1
    return df

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


def readMergedRecLists(dataset,df0,df1,df2):
    without_GT_df = readMergedTxtWithoutGT(dataset)
    without_GT_df['label'] = without_GT_df.apply(check_paths,axis=1,df0=df0,df1=df1,df2=df2)
    return without_GT_df


def calculate_hit_k(df,k,dataset):
    # df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    grouped = df.groupby('bugId')
    hit = 0
    for bugid,group in grouped:
        if (group['rank']<k).any():
            hit +=1

    data_len = {'zookeeper':470,
           'openjpa':533,
           'Tomcat':992,
           'aspectj':563,
           'hibernate':1285,
           'lucene':1454
           }
    print(f"hit: {hit} length:{data_len}")
    print(f"hit@{k}:{hit/data_len[dataset]}")
    # exit()


if __name__ == '__main__':
    configx =ConfigX()

    # setView
    pd.set_option('display.expand_frame_repr', False)

    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f">>>>>readMergedRecLists: {dataset} <<<<<<")
        gt_amalgam_df = readGroundTruth(file['bugPredict'][0],configx.approach[0])
        gt_bugLocator_df = readGroundTruth(file['bugPredict'][1],configx.approach[1])
        gt_blizzard_df = readGroundTruth(file['bugPredict'][2],configx.approach[2])

        # # temp : top20_other_approaches
        for i in range(3):
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
            for k in [1,5,10,15,20]:
                calculate_hit_k(res_df,k)
        # print(gt_bugLocator_df)
        # merged_df = readMergedRecLists(dataset,gt_amalgam_df,gt_bugLocator_df,gt_blizzard_df)
        # # print(merged_df)
        #
        # df_merged = gt_bugLocator_df.merge(merged_df, left_on=['bugId', 'path'], right_on=['bugId', 'path_1'], how='left',
        #                        indicator=True)
        #
        # # 筛选出没有匹配的数据
        # df_not_in_B = df_merged[df_merged['_merge'] == 'left_only']
        #
        # # 只保留 df_A 的列
        # df_not_in_B = df_not_in_B[gt_bugLocator_df.columns]
        # print(df_not_in_B)
        # exit()
        # df_A_filtered = merged_df[merged_df['label'] == 0]
        #
        # # 查找 df_A_filtered 中 'path_1' 是否在 df_B 的 'path' 中
        # df_matched = gt_bugLocator_df.merge(merged_df, left_on=['bugId','path'], right_on=['bugId','path_1'], how='inner')
        # # df_matched = df_matched.drop_duplicates()
        # print(df_matched)
        #
        # exit()
        # merged_df.to_csv(f"../data/get_info/{dataset}/mergedRecList.csv")
        # print(f"MergedRecList of {dataset} is finished!")
