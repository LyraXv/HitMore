import re

import pandas as pd


def readRecList(dataset,i):
    path = "../data/splited_and_boosted_data/"+dataset+"/"+str(i)+".csv"
    return pd.read_csv(path)

def search_bugCmit(bugId,dataset):
    path = "../data/ordered_bugCmit/"+dataset
    with open(path,'r')as f:
        commits = [line.strip().split(',') for line in f.readlines()]
        df_commits = pd.DataFrame(commits,columns=['bugId','cmit','date'])
    return df_commits[df_commits['bugId']==str(bugId)]['cmit'].values[0]

def openCodeCorpus(dataset,bugCmit):
    path = "../../CodeCorpus/"+dataset+"/"+bugCmit+".csv"
    return pd.read_csv(path)

def simplify_string(s):
    return re.sub(r'[^a-zA-Z0-9_]', '', s)

def is_substring(sub, df):
    sub_simplified = simplify_string(sub)
    filtered_df = df[df['File Path'].apply(lambda x: sub_simplified in simplify_string(x))]
    if len(filtered_df) > 1:
        print("Warning: Multiple matches found, this may indicate an error.")
    return filtered_df

def searchAllContent(CodeCorpus,filepaths): #df dict
    CodeCorpus['simplified_path'] = CodeCorpus['File Path'].apply(simplify_string)
    if pd.notna(filepaths['path_2']) and CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_2'])).any():
        filtered_df = CodeCorpus[CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_2']))]
    elif pd.notna(filepaths['path_1']) and CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_1'])).any():
        filtered_df = CodeCorpus[CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_1']))]
    elif pd.notna(filepaths['path_0']) and CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_0'])).any():
        filtered_df = CodeCorpus[CodeCorpus['simplified_path'].str.contains(simplify_string(filepaths['path_0']))]
    else:
        return None
    if len(filtered_df)>1:
        print("Warning: Multiple matches found, this may indicate an error.")
        print(filepaths)
        return filtered_df[0:1]['All Content'].values[0]
    else:
        return filtered_df['All Content'].values[0]