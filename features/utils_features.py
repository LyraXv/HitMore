import math
import re

import pandas as pd


def readRecList(dataset,i):
    return pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/{i}.csv")

# merge features with three dimensions
def updateFeatures(df_Features,df_new):
    if 'index' not in df_Features.columns:
        df_Features = df_Features.rename(columns ={'Unnamed: 0':'index'})
    df_merge = pd.merge(df_Features,df_new,how="left",on=['index','bugId'],suffixes=('','_new'))
    if 'Unnamed: 0' in df_merge:
        df_merge = df_merge.drop(columns='Unnamed: 0')
    common_columns = [col for col in df_new.columns if col in df_Features.columns and col not in ['index','bugId']]
    if common_columns:
        for col in common_columns:
            df_merge[col] = df_merge[col + '_new']
            df_merge.drop(columns=[col + '_new'], inplace=True)
    return df_merge

def search_bugCmit(bugId,dataset):
    path = f"../data/ordered_bugCmit/{dataset}"
    with open(path,'r')as f:
        commits = [line.strip().split(',') for line in f.readlines()]
        df_commits = pd.DataFrame(commits,columns=['bugId','cmit','date'])
    return df_commits[df_commits['bugId']==str(bugId)]['cmit'].values[0]

def openCodeCorpus(dataset,bugCmit):
    path = f"../../CodeCorpus/{dataset}/{bugCmit}.csv"
    return pd.read_csv(path)

def simplify_string(s):
    # 只保留字母和数字
    return re.sub(r'[^a-zA-Z0-9_]', '', s)

def is_substring(sub, df):
    sub_simplified = simplify_string(sub)
    filtered_df = df[df['File Path'].apply(lambda x: sub_simplified in simplify_string(x))]
    if len(filtered_df) > 1:
        print("Warning: Multiple matches found, this may indicate an error.")
    return filtered_df

def searchCodeAndComments(CodeCorpus,filepaths): #df dict
    # path_i存在且CodeCorpus能够查询
    CodeCorpus['simplified_path'] = CodeCorpus['File Path'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = CodeCorpus[CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = CodeCorpus[CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = CodeCorpus[CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None,None
    if len(filtered_df)>1:
        print("Warning: Multiple matches found, this may indicate an error.")
        print(filepaths)
        return filtered_df[0:1]['Code'].values[0], filtered_df[0:1]['Comments'].values[0]
    else:
        return filtered_df['Code'].values[0],filtered_df['Comments'].values[0]

def openNDEV(dataset,bugCmit):
    path = f"../../NDEV/{dataset}/{bugCmit}.csv"
    return pd.read_csv(path)

def searchContributorsNum(NDEV,filepaths,bugCmit):#df dict
    NDEV['simplified_path'] = NDEV['file'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and NDEV['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = NDEV[NDEV['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and NDEV['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = NDEV[NDEV['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and NDEV['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = NDEV[NDEV['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None
    if len(filtered_df)>1:
        with open("../data/splited_and_boosted_data/MultipleFilesPaths.txt", 'a') as f:
            f.write(f"{bugCmit},{filepaths}\n")
        f.close()
        print("Warning: Multiple matches found, this may indicate an error.")
        return filtered_df[0:1]['contributors'].values[0]
    else:
        return filtered_df['contributors'].values[0]

# overlap
# def openContributors(dataset,bugCmit):
#     path = f"../data/Contributors/{dataset}/{bugCmit}.csv"
#     return pd.read_csv(path)

def searchContributors(Contributors, filepaths):
    Contributors['simplified_path'] = Contributors['file'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and Contributors['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = Contributors[Contributors['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and Contributors['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = Contributors[Contributors['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and Contributors['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = Contributors[Contributors['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None
    if len(filtered_df)>1:
        print("Warning: Multiple matches found, this may indicate an error.")
        print(filepaths)
        return filtered_df[0:1]['contributors'].values[0]
    else:
        return filtered_df['contributors'].values[0]

def openDevelopers(dataset,bugCmit):
    path = f"../../Developers/{dataset}/{bugCmit}.csv"
    return pd.read_csv(path)

def searchDeveloper(Developer_files,filepaths):
    Developer_files['simplified_path'] = Developer_files['file'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and Developer_files['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = Developer_files[Developer_files['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and Developer_files['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = Developer_files[Developer_files['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and Developer_files['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = Developer_files[Developer_files['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None
    return filtered_df[0:]['developer_name'].values

def openDAF_MAF(dataset,bugCmit):
    path = f"../../DAF_MAF/{dataset}/{bugCmit}.csv"
    return pd.read_csv(path)

def searchDAF_MAF(DAF_MAF_all,filepaths):
    DAF_MAF_all['simplified_path'] = DAF_MAF_all['file'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and DAF_MAF_all['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = DAF_MAF_all[DAF_MAF_all['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and DAF_MAF_all['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = DAF_MAF_all[DAF_MAF_all['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and DAF_MAF_all['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = DAF_MAF_all[DAF_MAF_all['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None,None
    if len(filtered_df)>1:
        sum_daf = 0
        sum_maf = 0
        for index,(daf,maf) in filtered_df[['DAF','MAF']].iterrows():
            sum_daf += daf
            sum_maf += maf
        len_filtered_df = len(filtered_df)
        return sum_daf/len_filtered_df,sum_maf/len_filtered_df
    else:
        return filtered_df['DAF'].values[0],filtered_df['MAF'].values[0]

def searchLRdata(LR_data,filepaths):
    LR_data['simplified_path'] = LR_data['file'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and LR_data['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = LR_data[LR_data['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and LR_data['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = LR_data[LR_data['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and LR_data['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = LR_data[LR_data['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None
    if len(filtered_df)>1:
        print("Warning: Multiple matches found, this may indicate an error.")
        return filtered_df[0:1]['values'].values[0]
    else:
        return filtered_df['values'].values[0]


def searchCCdata(CC_data, filepaths):
    CC_data['simplified_path'] = CC_data['file'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and CC_data['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = CC_data[CC_data['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and CC_data['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = CC_data[CC_data['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and CC_data['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = CC_data[CC_data['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None
    if len(filtered_df)>1:
        if (filtered_df['type'] == 'class').any():
            filtered_df = filtered_df[filtered_df['type']=='class']
        # if len(filtered_df) > 1:
        #     print("filtered", filtered_df['file'].values,filtered_df['type'].values)
        return filtered_df.iloc[0:1,3:-1]
    else:
        return filtered_df.iloc[:,3:-1]

if __name__ == "__main__":
    # test
    dataset = 'zookeeper'
    bugCmit = search_bugCmit('2',dataset)
    print(bugCmit)
    CodeCorpus = openCodeCorpus(dataset,bugCmit)
    path = {'path_0':'zookeeper.server.quorum.FastLeaderElection.java',
            'path_1':'org.apache.zookeeper.server.quorum.FastLeaderElection.java',
            'path_2':'src/java/main/org/apache/zookeeper/server/quorum/FastLeaderElection.java'}
    print(searchCodeAndComments(CodeCorpus,path))  # wait to write