
import pandas as pd
from configx.configx import ConfigX
from model.MAP_MRR import TotalTrulyBuggyFilesNum



def get_amalgam_hit_k(k,bug_list,file_path):
    columns = ['id', 'file', 'rank', 'score']
    data = pd.read_csv(file_path, sep='\t', names=columns)

    hc_all =0

    filtered_data = data[data['rank'] <= 4] #19
    groups = filtered_data.groupby('id')
    hit_num= 0
    for id,group in groups:
        if str(id) in bug_list:
            if (((group['rank'] < k)).any()):
                hit_num +=1
            fixed_files = df_fixed_files[df_fixed_files['bugId'] == int(id)]
            if group.shape[0] == fixed_files['file_count'].values[0] & group.shape[0] > 1:
                hc_all+=1
    hit_k = hit_num/len(bug_list)
    hit_k = '{:.2f}%'.format(hit_k * 100)
    top_5_all[dataset]+=hc_all
    return hit_k

def get_bugLocator_hit_k(k,bug_list,file_path):
    hc_all = 0
    data = pd.read_csv(file_path, sep=',', header=0)
    # 去除 `rank` 大于 19 的数据
    filtered_data = data[data['rank'] <= 4] #19
    groups = filtered_data.groupby('bugId')
    hit_num= 0
    for id,group in groups:
        if str(id) in bug_list:
            if (((group['rank'] < k)).any()):
                hit_num +=1
            fixed_files = df_fixed_files[df_fixed_files['bugId'] == int(id)]
            if group.shape[0] == fixed_files['file_count'].values[0] & group.shape[0] > 1:
                hc_all += 1
    hit_k = hit_num/len(bug_list)
    hit_k = '{:.2f}%'.format(hit_k * 100)
    top_5_all[dataset]+=hc_all
    return hit_k

def get_blizzard_hit_k(k,bug_list,file_path):
    hc_all =0
    data = pd.read_csv(file_path, sep=',', header=0)
    # 去除 `rank` 大于 19 的数据
    filtered_data = data[data['rank'] <= 5] #20
    groups = filtered_data.groupby('bugId')
    hit_num= 0
    for id,group in groups:
        if str(id) in bug_list:
            if (((group['rank'] <= k)).any()):
                hit_num +=1
            fixed_files = df_fixed_files[df_fixed_files['bugId'] == int(id)]
            if group.shape[0] == fixed_files['file_count'].values[0] & group.shape[0]>1:
                hc_all += 1
    hit_k = hit_num/len(bug_list)
    hit_k = '{:.2f}%'.format(hit_k * 100)
    top_5_all[dataset]+=hc_all
    return hit_k

if __name__ == '__main__':
    configx = ConfigX()
    Preprocess = configx.preprocess

    # setView
    pd.set_option('display.expand_frame_repr', False)
    res_list = []
    # fold = 4
    top_5_all ={
        'zookeeper':0,
        'openjpa':0,
        'Tomcat':0,
        'aspectj':0,
        'hibernate':0,
        'lucene':0
    }

    for dataset,content in configx.filepath_dict.items():
        # bug_list(time)
        # bug_id_info = pd.read_csv(f"../data/ordered_bugCmit/ordered_bugCmit_{dataset}_time", header=None)
        # bug_id_info.columns = ['bugId', 'commit', 'time']
        # bug_id = bug_id_info['bugId'].tolist()

        # fold
        # bug_id_info = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/{fold}.csv")
        # bug_id = set(bug_id_info['bugId'].tolist())


        # fiexed_files_list
        df_fixed_files = TotalTrulyBuggyFilesNum(dataset)

        # all bugs
        bug_id = df_fixed_files['bugId'].tolist()

        bug_list = [str(i) for i in bug_id]
        # predict_file_path
        path_list = content['bugPredict']

        amalgam_list = []
        bugLocator_list = []
        blizzard_list = []
        amalgam_list.append(dataset)
        bugLocator_list.append(dataset)
        blizzard_list.append(dataset)
        for k in [1,5,10,20]:
            if k!=5:
                continue
            # amalgam_list.append(get_amalgam_hit_k(k,bug_list,path_list[0]))
            # bugLocator_list.append(get_bugLocator_hit_k(k,bug_list,path_list[1]))
            blizzard_list.append(get_blizzard_hit_k(k,bug_list,path_list[2]))
        # res_list.append(amalgam_list)
        # res_list.append(bugLocator_list)
        res_list.append(blizzard_list)

    print(top_5_all)
        # df_res = pd.DataFrame(data=res_list,columns=['dataset','hit@1','hit@5','hit@10','hit@20'])
        # df_res.to_csv(f"../data/results/three_techniques_hit_k.csv")

