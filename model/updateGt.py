# update initial_recommendation_data
# update sourcefile
# update Amalgam Rank

import math
import os

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
    if not matching_row.empty:
        return matching_row.values[0].split('$')[0]
    else:
        return None

def updateRecLists(filepaths,dataset,approach):
    files = os.listdir(filepaths)
    FileNotFound = []
    for fi in files:
        res_recommend = []
        # merge filepath and filename
        fi_d = os.path.join(filepaths, fi)
        res = open(fi_d, 'r',encoding='utf-8')
        bugid = fi.strip(".txt")

        git_commit = utils.searchGitCommit(bugid,dataset)
        if git_commit is None:
            print(f"No git commit found for BugId: {bugid}")
            continue
        df_git = utils.read_commit_to_df(git_commit, dataset)

        index = 0
        if approach == configx.approach[0]:
            for rr in res:
                res_list = list(rr.strip('\n').split('	'))
                filepath = updateFilePath(df_git,res_list[2].strip(".java"))
                if filepath is not None:
                    res_list.insert(0,fi.split(".txt")[0]) # insert bugId
                    res_list[3] = filepath
                    res_list[1] = index #rank
                    index +=1
                    res_recommend.append(res_list)
                else:
                    # print(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
                    FileNotFound.append(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
        elif approach == configx.approach[1]:
            for rr in res:
                res_list = list(rr.strip('\n').split(','))
                filepath = updateFilePath(df_git, res_list[2].strip(".java"))
                if filepath is not None:
                    res_list[2] = filepath
                    res_recommend.append(res_list)
                else:
                    # print(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
                    FileNotFound.append(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
        elif approach == configx.approach[2]:
            for rr in res:
                res_list = list(rr.strip('\n').split(','))
                res_list[2] = res_list[2].replace('/', '\\')
                filepath = updateBlizzardFilePath(df_git, res_list[2])
                if filepath is not None:
                    res_list[1] = int(res_list[1]) - 1
                    res_list[2] = filepath
                    res_recommend.append(res_list)
                else:
                    # print(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
                    FileNotFound.append(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")

        res.close()
        # save as txt
        save_path = "../data/initial_recommendation_data/" + approach + '_new/' + dataset +'/' + bugid + '.txt'
        with open(save_path,'w',encoding='utf-8') as file:
            for row in res_recommend:
                line = ','.join(map(str,row))
                file.write(line + '\n')
        # Not Found File Info
        fileNotFoundInfoPath = "../data/initial_recommendation_data/" + approach + '_new/'+ approach + '_' +dataset +'_FileNotFound.txt'
        with open(fileNotFoundInfoPath,'w',encoding='utf-8') as file:
            for row in FileNotFound:
                file.write(row + '\n')

def updateGroundTruthFile(filepaths,approach,dataset):

    res = open(filepaths,'r')
    groundTruthList = []
    FileNotFound = []
    if approach == configx.approach[0]: # amalgam
        for line in res:
            res_list = list(line.strip('\n').split('	'))
            bugid = res_list[0]
            git_commit = utils.searchGitCommit(bugid, dataset)
            if git_commit is None:
                print(f"No git commit found for BugId: {bugid}")
                continue
            df_git = utils.read_commit_to_df(git_commit, dataset)
            if df_git.empty:
                print(f"FileNotFoundError:{bugid}")
                continue
            sourcefile = updateFilePath(df_git,res_list[1].strip(".java."))
            if sourcefile is not None:
                res_list[1] = sourcefile
                res_list = res_list[:2]
                groundTruthList.append(res_list)
            else:
                # print(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
                FileNotFound.append(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
    elif approach == configx.approach[1]:
        for line in res:
            res_list = list(line.strip('\n').split(','))
            # print(f"resList:{res_list}")
            if not res_list[0].isdigit():
                continue
            bugid = res_list[0]
            git_commit = utils.searchGitCommit(bugid, dataset)
            if git_commit is None:
                print(f"No git commit found for BugId: {bugid}")
                continue
            df_git = utils.read_commit_to_df(git_commit, dataset)
            sourcefile = updateFilePath(df_git, res_list[1].strip(".java"))
            if sourcefile is not None:
                res_list[1] = sourcefile
                res_list = res_list[:2]
                # print(f"new_res_list:{res_list}")
                groundTruthList.append(res_list)
            else:
                # print(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
                FileNotFound.append(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
    elif approach == configx.approach[2]:
        for line in res:
            res_list = list(line.strip('\n').split(','))
            if not res_list[0].isdigit():
                continue
            bugid = res_list[0]
            git_commit = utils.searchGitCommit(bugid, dataset)
            if git_commit is None:
                print(f"No git commit found for BugId: {bugid}")
                continue
            df_git = utils.read_commit_to_df(git_commit, dataset)
            res_list[2] = res_list[2].replace('/', '\\')
            sourcefile = updateBlizzardFilePath(df_git, res_list[2].strip(".java"))
            if sourcefile is not None:
                temp_list = []
                temp_list.append(bugid)
                temp_list.append(sourcefile)
                groundTruthList.append(temp_list)
            else:
                # print(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")
                FileNotFound.append(f"FileNotFound: BugId:{bugid} Git: {git_commit} FilePath: {res_list}")

    res.close()
    # save as txt
    save_path = "../data/initial_recommendation_data/" + approach +'_new/'+dataset + '_bugPredict.txt'
    with open(save_path, 'w', encoding='utf-8') as file:
        for row in groundTruthList:
            line = ','.join(map(str, row))
            file.write(line + '\n')

    gtNotFoundInfoPath = "../data/initial_recommendation_data/" + approach + '_new/' + approach + '_' + dataset + '_GTFileNotFound.txt'
    with open(gtNotFoundInfoPath, 'w', encoding='utf-8'):
        for row in FileNotFound:
            file.write(row + '\n')


if __name__ == '__main__':
    configx = ConfigX()
    for index in range(3):

        for dataset,file in configx.filepath_dict.items():
            if dataset not in ('hibernate'):
                continue
            # update gt
            print(f">>>>>{configx.approach[index]} of {dataset} GroundTruth begin updating<<<<<")
            updateGroundTruthFile(file['bugPredict'][index], configx.approach[index], dataset)
            print(f">>>>>{configx.approach[index]} GroundTruth of {dataset} is updated!<<<<<")