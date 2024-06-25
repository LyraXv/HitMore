# wait to delete

import math

import pandas as pd

from configx.configx import ConfigX
from utils import utils


def updateFilePath(df_git,sourcefile):
    matching_rows = df_git[df_git['class'].str.contains(sourcefile)]
    if matching_rows.empty:
        return None
    else:
        return matching_rows['class'].values[0].split('$')[0]

def updateBlizzardFilePath(df_git,sourcefile):
    matching_row = df_git.loc[df_git['file'].apply(lambda x: sourcefile in x), 'class']
    # print("sourcefile",sourcefile)
    # print("matching_row",matching_row.values[0])
    if not matching_row.empty:
        return matching_row.values[0].split('$')[0]
    else:
        return math.nan

# load recommended List; extract filename u
def getBugFileInfo(datapath,approach,dataset):
    df = pd.read_csv(datapath)

    filepaths = []
    git_commit_cache = {}
    df_git_cache = {}

    for index, row in df.iterrows():
        bug_id = row['BugId']
        if bug_id not in git_commit_cache:
            git_commit = utils.searchGitCommit(bug_id, dataset)
            git_commit_cache[bug_id] = git_commit
        else:
            git_commit = git_commit_cache[bug_id]

        if git_commit not in df_git_cache:
            df_git = utils.read_commit_to_df(git_commit, dataset)
            df_git_cache[git_commit] = df_git
        else:
            df_git = df_git_cache[git_commit]

        if approach != configx.approach[2]:
            filepaths.append(updateFilePath(df_git, row['SourceFile'].rstrip(".java")))
        else:
            filepaths.append(updateBlizzardFilePath(df_git, row['SourceFile']))

    df['Filepath'] = filepaths
    df['Approach'] = approach
    if approach == configx.approach[0]:
        df = df.dropna(subset=['Filepath'])
        df['Rank'] = df.groupby('BugId')['score'].rank(method='zeor',ascending=False)
    return df

if __name__ == "__main__":
    configx = ConfigX()

    # setView
    pd.set_option('display.expand_frame_repr', False)

    # dataset
    for dataset,file in configx.filepath_dict.items():
        df_amalgam = getBugFileInfo('../data/get_info/'+dataset+'/'+ configx.approach[0]+'List.csv', configx.approach[0], dataset)
        df_amalgam .to_csv('../data/get_info/'+dataset+'/analgamList_2.csv')
        print(dataset+":amalgamList is updated")

        # df_bugLocator = getBugFileInfo('../data/get_info/'+dataset+'/'+ configx.approach[1]+'List.csv', configx.approach[1], dataset)
        # df_bugLocator.to_csv('../data/get_info/'+dataset+'/bugLocatorList_2.csv')
        # print(dataset+":bugLoctorList is updated")
        #
        # df_blizzard = getBugFileInfo('../data/get_info/'+dataset+'/'+ configx.approach[2]+'List.csv', configx.approach[2], dataset)
        # df_blizzard.to_csv('../data/get_info/'+dataset+'/blizzardList_2.csv')
        # print(dataset+":blizzardList is updated")

        exit()

