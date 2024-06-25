#读 amalgam的Recmened，
import math
import os

import pandas as pd

from configx.configx import ConfigX
from utils import utils
from utils.readBugInfo import readGroundTruth


def updateFilePath(df_git,sourcefile):
    matching_rows = df_git[df_git['class'].str.contains(sourcefile)]
    if matching_rows.empty:
        return None
    else:
        return matching_rows['class'].values[0].split('$')[0]


def updateAmalgamRecList(filepaths,dataset,groundTruthPath):
    files = os.listdir(filepaths)
    res_recommend = []
    for fi in files:

        # merge filepath and filename
        fi_d = os.path.join(filepaths, fi)
        res = open(fi_d, 'r')
        # bugid = int(fi.strip(".txt"))
        bugid = fi.strip(".txt")

        git_commit = utils.searchGitCommit(bugid,dataset)
        if git_commit is None:
            print(f"No git commit found for BugId: {bugid}")
            continue
        df_git = utils.read_commit_to_df(git_commit, dataset)

        top_20_index = 0
        for rr in res:
            # res_list = []
            res_list = list(rr.strip('\n').split('	'))
            filepath = updateFilePath(df_git,res_list[2].strip(".java"))
            if filepath is not None:
                res_list.insert(0,fi.split(".txt")[0]) # insert bugId
                res_list.insert(4,filepath)
                res_list[1] = top_20_index #rank
                top_20_index +=1
                res_recommend.append(res_list)
            if top_20_index == 20:
                break

        # rank_index =0
        # for file in res_recommend:
        #     file[1] = rank_index
        #     amalgam_recList.append(file)
        #     rank_index +=1
        #     if rank_index==20:
        #         break
        # print(amalgam_recList)
        res.close()

    gt = readGroundTruth(groundTruthPath,configx.approach[0],dataset)
    gt['label'] = 1

    dataframe = pd.DataFrame(res_recommend, columns=['BugId', 'Rank', 'Score', 'SourceFile','FilePath'])
    dataframe = dataframe.reindex(columns=['BugId', 'Rank', 'SourceFile', 'Score','FilePath'])
    df = pd.merge(dataframe, gt, on=['BugId', 'SourceFile'], how='left')
    df = df.drop(columns=['Rank_y', 'Score_y'])
    df = df.rename(columns={'Rank_x': 'Rank', 'Score_x': 'Score'})
    df['SourceFile'] = df['SourceFile'].str.rstrip('.')
    df['Approach'] = 'amalgam'
    df.fillna(0,inplace=True)
    df = df.astype({"label": int})
    return df


if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset in('zookeeper','openjpa','Tomcat','aspecj'):
            continue
        df = updateAmalgamRecList(file['bugFixedFileRanks'][0],dataset,file['bugPredict'][0])
        df.to_csv('../data/get_info/'+dataset+'/amalgamList.csv')
        print(dataset+"AmalgamList is saved!")