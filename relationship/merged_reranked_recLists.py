import ast
import os.path
import re
import numpy as np
import pandas as pd

from configx.configx import ConfigX



def read_df_lists(bugId):
    df_file = f'D:\\HitMore\\DataFlow\\{dataset}_df_list.csv'
    df_df_lists = pd.read_csv(df_file,encoding='utf-8')
    df_df_lists = df_df_lists[['index','bugID','df_filePath']]
    df_df_lists = df_df_lists[df_df_lists['bugID']==bugId]

    pre_fix = f"D:\\Dataset\\SourceFiles\\SourcesFilesNew\\{dataset_name}\\"
    df_df_lists['df_filePath'] = df_df_lists['df_filePath'].str.replace(pre_fix,'',regex=False)
    if dataset == "Tomcat":
        pre_fix = f"D:\\Dataset\\SourceFiles\\SourcesFilesNew\\tomcat\\"
        df_df_lists['df_filePath'] = df_df_lists['df_filePath'].str.replace(pre_fix, '', regex=False)
    df_df_lists = df_df_lists.rename(columns={'df_filePath':'path','bugID':'bugId'})

    df_df_lists['df_occ'] = 1
    return df_df_lists

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_cf_lists(bugId):
    cf_file = f'D:\\HitMore\\ControlFlow\\{dataset}_cf_list.csv'
    cf_lists = pd.read_csv(cf_file,encoding='utf-8',sep=',')
    cf_lists = cf_lists[['index','bugId','cf_file']]
    if dataset == 'aspectj':
        for i in range(1,len(cf_lists)):
            if not is_numeric(cf_lists.loc[i,'bugId']) and len(cf_lists.loc[i,'index'])>6:
                str = cf_lists.loc[i,'index']
                print("a:",str)
                if str[-1].isdigit():
                    str = str +','+cf_lists.loc[i,'bugId']
                else:
                    str +=cf_lists.loc[i,'bugId']
                print("b:",str)
                if cf_lists.loc[i - 1, 'cf_file'][-1].isdigit():
                    cf_lists.loc[i - 1, 'cf_file'] += ","+str
                else:
                    cf_lists.loc[i - 1, 'cf_file'] += str
                print("c:",cf_lists.loc[i - 1, 'cf_file'])

                if cf_lists.loc[i - 1, 'cf_file'].endswith('"'):
                    cf_lists.loc[i - 1, 'cf_file'] = cf_lists.loc[i - 1, 'cf_file'].rstrip('"')

                print(cf_lists.loc[i - 1, 'cf_file'])
                cf_lists.drop(i,inplace=True)
    if not pd.api.types.is_numeric_dtype(cf_lists['bugId']):
        cf_lists['bugId'] = cf_lists['bugId'].astype(int)
    cf_lists = cf_lists[cf_lists['bugId']==bugId]
    new_rows = []
    cf_lists['cf_file'] = cf_lists['cf_file'].apply(lambda x: ast.literal_eval(x))

    for idx, row in cf_lists.iterrows():
        address_dict = row['cf_file']
        for path, num in address_dict.items():
            new_rows.append({'index':row['index'],'bugId': int(row['bugId']), 'cf_filePath': path, 'call_num': num})
    df_cf_lists = pd.DataFrame(new_rows)
    if df_cf_lists.empty:
        return df_cf_lists
    pre_fix = f"D:\\Dataset\\SourceFiles\\SourcesFilesNew\\{dataset_name}\\"
    df_cf_lists['cf_filePath'] = df_cf_lists['cf_filePath'].str.replace(pre_fix,'',regex=False)
    if dataset == "Tomcat":
        pre_fix = f"D:\\Dataset\\SourceFiles\\SourcesFilesNew\\tomcat\\"
        df_cf_lists['cf_filePath'] = df_cf_lists['cf_filePath'].str.replace(pre_fix, '', regex=False)
    df_cf_lists = df_cf_lists.rename(columns={'cf_filePath':'path'})
    df_cf_lists['cf_occ'] = 1
    return df_cf_lists


def read_oc_lists(bugId):
    co_file = f'D:\\HitMore\\Cooccurence\\{dataset}_co_list.csv'
    co_lists = pd.read_csv(co_file,encoding='utf-8')
    co_lists = co_lists[co_lists['bugId']==bugId]
    co_lists = co_lists[['index','bugId','co_file']]

    # str to list
    co_lists['co_file'] = co_lists['co_file'].apply(convert_to_list)
    # Decomposition list
    df_exploded = co_lists.explode('co_file')
    # rename filepath
    df_exploded = df_exploded.rename(columns = {'co_file':'path'})
    df_exploded['path'] = df_exploded['path'].str.replace("/","\\")

    df_exploded['co_occ']=1

    return df_exploded

def convert_to_list(str_path):
    # 去掉开头的 '{' 和结尾的 '}'，然后以逗号分割并去掉空格
    string_cleaned = str_path[1:-1]
    # 分割字符串并去掉单引号和多余的空格
    return [s.strip().strip("'") for s in string_cleaned.split(',')]

def read_rec_lists():
    csv_file = f'D:\\HitMore\\HitMore-main\\data\\rec_lists\\{dataset}_truly_buggy_file_result.csv'
    df_rec_lists = pd.read_csv(csv_file,encoding='utf-8')
    df_rec_lists = df_rec_lists[['index','bugID','filePath']]
    return df_rec_lists

def read_amalgam_lists(bugId):
    list_path = configx.filepath_dict[dataset]['bugFixedFileRanks'][0] +"/"+ str(bugId) + ".txt"
    df = pd.read_csv(list_path, sep='\t', header=None)
    df.columns = ['rank','score','path']
    df = df.drop(columns=['score'])
    df['rank'] = df['rank']+1
    df['path'] = df['path'].apply(process_amalgam_path)
    dict_ama = dict(zip(df['path'],df['rank']))
    # print(dict_ama)
    return dict_ama

def process_amalgam_path(path):
    path = path.rstrip('.java.')
    path =path.replace('.','\\')+'.java'
    return  path

def read_bugLocator_lists(bugId):
    list_path = configx.filepath_dict[dataset]['bugFixedFileRanks'][1] +"/"+ str(bugId) + ".txt"
    if not os.path.exists(list_path):
        return {}
    else:
        df = pd.read_csv(list_path, sep=',', header=None)
        df.columns = ['bugId','rank','path','score']
        df = df.drop(columns=['bugId','score'])
        df['rank'] = df['rank']+1
        df['path'] = df['path'].apply(process_bugLocator_path)
        dict_bugl = dict(zip(df['path'],df['rank']))
        return dict_bugl

def process_bugLocator_path(path):
    path = path.rstrip('.java')
    path =path.replace('.','\\')+'.java'
    return  path

def read_blizzard_lists(bugId):
    list_path = configx.filepath_dict[dataset]['bugFixedFileRanks'][2] +"/"+ str(bugId) + ".txt"
    df = pd.read_csv(list_path, sep=',', header=None)
    df.columns = ['bugId','rank','path']
    df = df.drop(columns=['bugId'])
    df['path'] = df['path'].apply(process_blizzaed_path)
    dict_bli = dict(zip(df['path'],df['rank']))
    return dict_bli

def process_blizzaed_path(path):
    path =path.replace('/','\\')
    return path

def find_rank(path,dict):
    for partial_path, num in dict.items():
        if partial_path in path:
            return num
    return None

def get_three_ranks(bugId,df):
    dict_amalgam = read_amalgam_lists(bugId)
    dict_bugLocator = read_bugLocator_lists(bugId)
    dict_blizzard = read_blizzard_lists(bugId)
    df['rank_0'],df['rank_1'],df['rank_2'] = [np.nan,np.nan,np.nan]
    df['rank_0'] = df['path'].apply(lambda  x:find_rank(x,dict_amalgam))
    df['rank_1'] = df['path'].apply(lambda  x:find_rank(x,dict_bugLocator))
    df['rank_2'] = df['path'].apply(lambda  x:find_rank(x,dict_blizzard))
    return  df

def calculate_buggy_score(df):
    df[['df_occ','cf_occ','co_occ','call_num']] = (df[['df_occ','cf_occ','co_occ','call_num']].fillna(0))
    df['rank_score'] = df.apply(lambda row: calculate_rank_score(row['rank_0'],row['rank_1'],row['rank_2']),axis=1)

    min_val = df['call_num'].min()
    max_val = df['call_num'].max()
    if max_val != min_val:
        df['normalized_call_num'] = (df['call_num'] - min_val) / (max_val - min_val)
    else:
        df['normalized_call_num'] = 0

    df['occ_num'] = df['df_occ']+ df['cf_occ'] + df['co_occ']
    # df['normalized_occ_num'] = (df['occ_num'] - 1) / (3 - 1)
    df['processed_occ_num'] = df['occ_num']/10

    df['score'] = df['rank_score']+df['processed_occ_num']+df['normalized_call_num']
    return df


def calculate_rank_score(a,b,c):
    n,sum = 0,0
    if not pd.isna(a):
        sum += 1/a
        n +=1
    if not pd.isna(b):
        sum += 1/b
        n +=1
    if not pd.isna(c):
        sum += 1/c
        n +=1
    # return sum*n
    return sum  # without multiply

def merge_three_predict_lists(df_ama,df_bugl,df_bli):
    res_list = list(df_bli)
    ama_list = list(df_ama)
    bugl_list = list(df_bugl)

    bugl_path = [path for path in bugl_list if not any (path in res for res in res_list)]
    res_list.extend(bugl_path)
    ama_path = [path for path in ama_list if not any (path in res for res in res_list)]
    res_list.extend(ama_path)
    return res_list

def readGroundTruth(bugId,df):
    # amalgam
    amalgam_path = configx.filepath_dict[dataset]['bugPredict'][0]
    df_ama = pd.read_csv(amalgam_path, sep='\t', header=None)
    df_ama.columns = ['bugId','path','rank','score']
    df_ama = df_ama[df_ama['bugId']==bugId]['path']
    df_ama = df_ama.apply(process_amalgam_path)

    # bugLocator
    bugLocator_path = configx.filepath_dict[dataset]['bugPredict'][1]
    df_bugl = pd.read_csv(bugLocator_path, sep=',')
    df_bugl.columns = ['bugId','path','rank','score']
    df_bugl = df_bugl[df_bugl['bugId']==bugId]['path']
    df_bugl = df_bugl.apply(process_bugLocator_path)

    # blizzard
    blizzard_path = configx.filepath_dict[dataset]['bugPredict'][2]
    df_bli = pd.read_csv(blizzard_path,sep=',')
    df_bli.columns = ['bugId','rank','path']
    df_bli = df_bli[df_bli['bugId']==bugId]['path']
    df_bli = df_bli.apply(process_blizzaed_path)

    truly_path_list = merge_three_predict_lists(df_ama,df_bugl,df_bli)

    df['label'] = 0
    # 判断list是否是df['']的子串
    df['label'] = df['path'].apply(lambda x:1 if any(path in x for path in truly_path_list) else 0)
    # for path in truly_path_list:
    #     df['label'] = df['path'].apply(lambda x:1 if path in x else 0)
    df = df.sort_values(by="score",ascending=False)
    return len(truly_path_list),df['label'].sum(),truly_path_list
    # return df

def read_fixed_file_list(bugId,df):
    path = f"../data/get_info/{dataset}/bug_report_fixedfiles.csv"
    df_fixedFiles = pd.read_csv(path, encoding="utf-8-sig")
    df_fixedFiles = df_fixedFiles.rename(columns={'bug_id':'bugId'})
    df_fixedFiles['bugId'] = df_fixedFiles['bugId'].str.extract('(\d+)').astype(int)
    path = df_fixedFiles[df_fixedFiles['bugId']==bugId]['files'].values[0]
    split_list = re.split(r'(\.java)', path)
    path_list = [split_list[i] + split_list[i + 1] for i in range(0, len(split_list) - 1, 2)]
    path_list = [path.replace('/', '\\') for path in path_list]
    df['label'] = 0
    df['label'] = df['path'].apply(lambda x: 1 if any(path in x for path in path_list) else 0)
    df = df.sort_values(by="score", ascending=False)
    return df


def origin_list_info(df,df_list,cf_list,co_list):

   df['df_occ'] = df['index'].isin(df_list['index']).astype(int)
   df['co_occ'] = df['index'].isin(co_list['index']).astype(int)
   if cf_list.empty:
       df[['cf_occ','call_num']] = 0,0
   else:
       df['cf_occ'] = df['index'].isin(cf_list['index']).astype(int)
       df_call_num = cf_list.groupby('index').size().reset_index(name = 'call_num')
       df = pd.merge(df,df_call_num,on='index',how='left')
   return df


def main():
    df_rec_lists = read_rec_lists()
    # final_lists = pd.DataFrame(columns=['bugId','fileIndex','filepath','df_occ','cf_occ','co_occ','call_num','rank_0','rank_1','rank_2'])
    # 按bugId分组
    grouped = df_rec_lists.groupby('bugID')
    flag = True
    for bugId,group in grouped:

        # if bugId != 4925 and flag:
        #     continue
        # else:
        #     flag = False
        print("bugId",bugId)
        df_origin_list = group.rename(columns={'filePath':'path','bugID':'bugId'})

        # three lists
        df_list = read_df_lists(bugId)
        cf_list = read_cf_lists(bugId)
        co_list = read_oc_lists(bugId)

        df_origin_list = origin_list_info(df_origin_list,df_list,cf_list,co_list)
        print(df_origin_list.shape,df_list.shape,cf_list.shape,co_list.shape)
        if not cf_list.empty:
            merged_df = df_list.merge(cf_list, on=['index', 'bugId', 'path'], suffixes=('_df1', '_df2'), how='outer')
        else:
            df_list['cf_occ'] =None
            df_list['call_num'] = None
            merged_df = df_list
        # print(merged_df.shape)
        merged_df = merged_df.merge(co_list, on=['index', 'bugId', 'path'], suffixes=('', '_df3'), how='outer')
        # print(merged_df.shape)
        merged_df = pd.concat([merged_df,df_origin_list])
        # print(merged_df.shape)

        # rank_lists
        df_res = get_three_ranks(bugId,merged_df)
        df_res = calculate_buggy_score(df_res)

        # truly_bugg_file
        df_res= read_fixed_file_list(bugId,df_res)
        # df_res= readGroundTruth(bugId,df_res)

        # save_only_file
        df_res = df_res.groupby('path',as_index=False).first()
        df_res = df_res.sort_values(by="score",ascending=False)
        if len(df_res) > 3000:
            df_to_save = df_res.head(3000)  # 取前 1000 行
        else:
            df_to_save = df_res  # 保留所有行
        df_to_save.to_csv(f"D:/HitMore/{ranked_res_filepath}/{dataset}/{bugId}.csv")

if __name__ == "__main__":
    configx = ConfigX()
    ranked_res_filepath = "ranked_res_0"
    dataset = "hibernate"
    dataset_name = 'hibernate-orm'
    main()