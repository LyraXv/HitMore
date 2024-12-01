import os.path
import re

import pandas as pd

from model.MAP_MRR import TotalTrulyBuggyFilesNum, HitCount, getHitMoreData, getHitMoreRecListLabel, \
    mean_average_precision, mean_reciprocal_rank


def read_reranked_list():
    df_bug_info = pd.read_csv(f"../data/get_info/{dataset}/bug_report_fixedfiles.csv",encoding="utf-8-sig")
    bug_id = df_bug_info['bug_id'].tolist()
    df_ensemble_list = getHitMoreData(dataset)
    df_ensemble_list = df_ensemble_list[['bugId','label']]
    df_ensemble_list['rank'] = df_ensemble_list.groupby('bugId').cumcount()
    df_res = pd.DataFrame(columns=['bugId','label','rank'])
    for id in bug_id:
        [bugId] = re.findall(r'\d+',id)
        bugId = int(bugId)
        rec_file_path = f"D:/HitMore/{ranked_res_filepath}/{dataset}/{bugId}.csv"
        if os.path.exists(rec_file_path):
            df_rec_list = pd.read_csv(rec_file_path)
            ##
            # df_rec_list['score'] = df_rec_list['rank_score']*df_rec_list['normalized_call_num']*df_rec_list['occ_num']
            # df_rec_list['score'] = df_rec_list['rank_score']*a+df_rec_list['normalized_call_num']+df_rec_list['occ_num']*b
            # df_rec_list['score'] = df_rec_list['rank_score']*df_rec_list['occ_num']*10+df_rec_list['normalized_call_num']/(4-df_rec_list['occ_num']*10)
            df_rec_list['score'] = df_rec_list['rank_score'] * df_rec_list['occ_num']
            # df_rec_list['score'] = df_rec_list['rank_score']
            df_rec_list = df_rec_list.sort_values(by="score", ascending=False)
            df_rec_list = df_rec_list.reset_index(drop=True)
            ##
            df_rec_list = df_rec_list.loc[0:19,['bugId','label']]
            df_rec_list['rank'] = df_rec_list.groupby('bugId').cumcount()
        else:
            df_rec_list = df_ensemble_list[df_ensemble_list['bugId']==bugId].iloc[0:20,:]
        df_res = pd.concat([df_res,df_rec_list])
    # df_res.to_csv(f"D:/HitMore/{ranked_res_filepath}/{dataset}_rec_lists.csv")
    df_res['bugId'] = df_res['bugId'].astype(int)
    return df_res

def read_only_reranked_list():
    bug_id_info = pd.read_csv(f"../data/ordered_bugCmit/ordered_bugCmit_{dataset}_time",header=None)
    bug_id_info.columns = ['bugId', 'commit', 'time']
    # bug_id_info = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/{fold}.csv")
    bug_id = set(bug_id_info['bugId'].tolist())
    print("bug_id_num",len(bug_id))
    df_ensemble_list = getHitMoreData(dataset)
    df_ensemble_list = df_ensemble_list[['bugId', 'label']]
    df_ensemble_list['rank'] = df_ensemble_list.groupby('bugId').cumcount()
    df_res = pd.DataFrame(columns=['bugId','label','rank'])
    for id in bug_id:
        bugId = int(id)
        rec_file_path = f"D:/HitMore/{ranked_res_filepath}/{dataset}/{bugId}.csv"
        if os.path.exists(rec_file_path):
            df_rec_list = pd.read_csv(rec_file_path)
            ##
            # df_rec_list['score'] = df_rec_list['rank_score']*df_rec_list['normalized_call_num']*df_rec_list['occ_num']
            # df_rec_list['score'] = df_rec_list['rank_score']*a+df_rec_list['normalized_call_num']+df_rec_list['occ_num']*b
            # df_rec_list['score'] = df_rec_list['rank_score']*df_rec_list['occ_num']*10+df_rec_list['normalized_call_num']/(4-df_rec_list['occ_num']*10)
            df_rec_list['score'] = df_rec_list['rank_score'] * df_rec_list['occ_num']
            # df_rec_list['score'] = df_rec_list['rank_score']
            df_rec_list = df_rec_list.sort_values(by="score", ascending=False)
            df_rec_list = df_rec_list.reset_index(drop=True)
            ##

            df_rec_list = df_rec_list.loc[0:19, ['bugId', 'label']]
            df_rec_list['rank'] = df_rec_list.groupby('bugId').cumcount()
        else:
            df_rec_list = df_ensemble_list[df_ensemble_list['bugId'] == bugId].iloc[0:20, :]
        df_res = pd.concat([df_res, df_rec_list])
    # df_res.to_csv(f"D:/HitMore/{ranked_res_filepath}/{dataset}_rec_lists.csv")
    df_res['bugId'] = df_res['bugId'].astype(int)
    return df_res

# complete lists
def read_all_reranked_list():
    df_bug_info = pd.read_csv(f"../data/get_info/{dataset}/bug_report_fixedfiles.csv",encoding="utf-8-sig")
    bug_id = df_bug_info['bug_id'].tolist()
    df_ensemble_list = getHitMoreData(dataset)
    df_ensemble_list = df_ensemble_list[['bugId','label']]
    df_ensemble_list['rank'] = df_ensemble_list.groupby('bugId').cumcount()
    df_res = pd.DataFrame(columns=['bugId','label','rank'])
    for id in bug_id:
        [bugId] = re.findall(r'\d+',id)
        bugId = int(bugId)
        rec_file_path = f"D:/HitMore/{ranked_res_filepath}/{dataset}/{bugId}.csv"
        if os.path.exists(rec_file_path):
            df_rec_list = pd.read_csv(rec_file_path)
            ##
            # df_rec_list['score'] = df_rec_list['rank_score']
            # df_rec_list['score'] = df_rec_list['rank_score']*a+df_rec_list['normalized_call_num']+df_rec_list['occ_num']*b
            df_rec_list['score'] = df_rec_list['rank_score'] * df_rec_list['occ_num']
            df_rec_list = df_rec_list.sort_values(by="score", ascending=False)
            df_rec_list = df_rec_list.reset_index(drop=True)
            ##
            df_rec_list = df_rec_list[['bugId','label']]
            df_rec_list['rank'] = df_rec_list.groupby('bugId').cumcount()
        else:
            df_rec_list = df_ensemble_list[df_ensemble_list['bugId']==bugId]
        df_res = pd.concat([df_res,df_rec_list])
    df_res['bugId'] = df_res['bugId'].astype(int)
    return df_res

def read_complete_reranked_list_partly():
    bug_id_info = pd.read_csv(f"../data/ordered_bugCmit/ordered_bugCmit_{dataset}_time",header=None)
    bug_id_info.columns = ['bugId', 'commit', 'time']
    bug_id = set(bug_id_info['bugId'].tolist())
    # bug_id_info = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/{fold}.csv")
    # bug_id = set(bug_id_info['bugId'].tolist())

    df_ensemble_list = getHitMoreData(dataset)
    df_ensemble_list = df_ensemble_list[['bugId','label']]
    df_ensemble_list['rank'] = df_ensemble_list.groupby('bugId').cumcount()
    df_res = pd.DataFrame(columns=['bugId','label','rank'])
    for id in bug_id:
        bugId = id
        bugId = int(bugId)
        rec_file_path = f"D:/HitMore/{ranked_res_filepath}/{dataset}/{bugId}.csv"
        if os.path.exists(rec_file_path):
            df_rec_list = pd.read_csv(rec_file_path)
            ##
            # df_rec_list['score'] = df_rec_list['rank_score']
            # df_rec_list['score'] = df_rec_list['rank_score']*a+df_rec_list['normalized_call_num']+df_rec_list['occ_num']*b
            df_rec_list['score'] = df_rec_list['rank_score'] * df_rec_list['occ_num']
            df_rec_list = df_rec_list.sort_values(by="score", ascending=False)
            df_rec_list = df_rec_list.reset_index(drop=True)
            ##
            df_rec_list = df_rec_list[['bugId','label']]
            df_rec_list['rank'] = df_rec_list.groupby('bugId').cumcount()
        else:
            df_rec_list = df_ensemble_list[df_ensemble_list['bugId']==bugId]
        df_res = pd.concat([df_res,df_rec_list])
    df_res['bugId'] = df_res['bugId'].astype(int)
    return df_res

def calculate_Accuracy_k(data,k):
    print(f">>>>>>k::{k}<<<<<<<<<")
    hit_list = []

    grouped = data.groupby('bugId')
    for bugid, group in grouped:
        sorted_group = group.sort_values(by='rank', ascending=True)
        if len(sorted_group) < k:
            top_k = sorted_group.copy()
        else:
            top_k = sorted_group.head(k).copy()

        if (((top_k['label'] == 1)).any()):
            hit = 1
        else:
            hit = 0
        hit_list.append(hit)
    print(f"sum:{sum(hit_list)},len:{len(hit_list)}")
    return sum(hit_list) / len(hit_list)



if __name__ == "__main__":
    res_locating_performance = []
    res_effectiveness = []
    ranked_res_filepath = "ranked_res_byTime" # ranked_res_0 ranked_res_byTime

    datasets = ['zookeeper','openjpa','Tomcat','aspectj','hibernate','lucene']
    # fold = 4
    for dataset in datasets:
        print("Current dataset: ",dataset)
        df_fixed_files = TotalTrulyBuggyFilesNum(dataset)
        # print(df_fixed_files[df_fixed_files['file_count']>1].shape) # multiple bugs

        # data = read_reranked_list()
        data = read_only_reranked_list() #time

        HitCount_1,HitCount_2,HitCount_all,HitCount_mult = HitCount(data, df_fixed_files)

        hits = {}
        for k in [1,5,10,15,20]:
            hits[f"Hit@{k}"] =calculate_Accuracy_k(data,k)
        print(hits)

        # all_reranked_list = read_all_reranked_list()
        all_reranked_list = read_complete_reranked_list_partly()# time
        all_reranked_label = getHitMoreRecListLabel(all_reranked_list)
        print(all_reranked_label.__len__())
        map_score = mean_average_precision(all_reranked_label)
        mrr_score = mean_reciprocal_rank(all_reranked_label)
        print(f"MAP: {map_score}")
        print(f"MRR: {mrr_score}")

        res_locating_performance.append([
            dataset,
            '{:.2f}%'.format(hits['Hit@1'] * 100),
            '{:.2f}%'.format(hits['Hit@5'] * 100),
            '{:.2f}%'.format(hits['Hit@10'] * 100),
            '{:.2f}%'.format(hits['Hit@20'] * 100),
            '{:.2f}'.format(map_score),
            '{:.2f}'.format(mrr_score)
        ])
        res_effectiveness.append([dataset,HitCount_1,HitCount_2,HitCount_all,HitCount_mult])


    df_locating_performace = pd.DataFrame(data=res_locating_performance, index=None, columns=['dataset','Hit@1','Hit@5','Hit@10','Hit@20','MAP','MRR'])
    df_locating_performace.to_csv(f"res_locating_performance_fold_time_temp.csv")
    df_effectiveness =  pd.DataFrame(data=res_effectiveness,index=None,columns=['dataset','HitCount@1','HitCount_2','HitCount_all','HitCount_mult'])
    df_effectiveness.to_csv(f"res_effectiveness_time_fold_time_temp.csv")