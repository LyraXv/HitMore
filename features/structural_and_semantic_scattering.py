# developers having a high scattering are more likely to introduce bugs during code change activities
import os
import csv
import itertools
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

import similarity as CS
from configx.configx import ConfigX
from features.utils_features import search_bugCmit, openDevelopers, searchDeveloper, readRecList, updateFeatures


def read_csv(dataset,bugCmit):
    developer_files = defaultdict(list)
    file_path = f"../../Developers/{dataset}/{bugCmit}.csv"
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            developer = row['developer_name']
            file = row['file']
            file = file.replace('\\', '/')
            developer_files[developer].append(file)
    return developer_files


def read_code_corpus(file_path):
    file_content = {}
    with open(file_path, 'r',errors='ignore') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file = row['File Path'].replace('\\', '/')
            content = row['All Content']
            file_content[file] = content
    return file_content


def calculate_path_distance(file1, file2):
    # 分割路径并计算不同部分的数量
    parts1 = file1.split('/')
    parts2 = file2.split('/')

    compath = os.path.commonpath([file1, file2]).replace('\\', '/')
#     print(compath)
    common_length = len(compath.split('/'))
#     print(common_length)
    distance = (len(parts1) - common_length) + (len(parts2) - common_length) - 1
#     print(distance)
    return distance


# 求和平均（每两个文件之间的最短距离）
def calculate_shortest_path_average(file_list):
    if len(file_list) < 2:
        return 0  # developer only modified one file

    path_lengths = []
    for file1, file2 in itertools.combinations(file_list, 2):
        path_length = calculate_path_distance(file1, file2)
        path_lengths.append(path_length)

    if path_lengths:
        return sum(path_lengths) / len(path_lengths)
    else:
        return 0


# 结构散射：开发人员d在时间段α中修改的类的数量 × 求和平均—每两个类之间的包的距离（最短路径算法）
def structural_scattering(all_files, file_count):
    if file_count > 1:
        avg_distance = calculate_shortest_path_average(all_files)
    else:
        avg_distance = 0

    StrScat = file_count * avg_distance
    return avg_distance, StrScat


# 求和平均（每两个文件之间的文本相似性）
def calculate_similarity_average(file_list, file_contents):
    file_sim = []
    # print(file_contents)
    for file1, file2 in itertools.combinations(file_list, 2):
        try:
            f1 = file_contents[file1]
            f2 = file_contents[file2]
        except KeyError:
            continue
        
        # f1 = CS.preprocess(file1_content)
        # f2 = CS.preprocess(file2_content)
        Str_f1 = " ".join(f1)
        Str_f2 = " ".join(f2)
        
        sim = CS.cosine_sim([Str_f1, Str_f2])

        file_sim.append(sim)

    if len(file_sim)!=0:
        return sum(file_sim) / len(file_sim)
    else:
        return 0
    

# 语义散射：开发人员d在时间段α中修改的类的数量 × （ 1 / 平均值—每两个类之间的文本相似度（VSM））
def semantic_scattering(all_files, file_count,file_contents):
    if file_count > 1:
        if file_count > 150:
            all_files = random.sample(all_files,150)
        avg_sim = calculate_similarity_average(all_files, file_contents)
        SemScat = file_count * (1 / avg_sim)
    else:
        avg_sim = 0
        SemScat = 0
    return avg_sim, SemScat


# 根据 commit 查找对应的 CSV 文件
def find_developer_files(commit, csv_folder='Developers'):
    csv_file = os.path.join(csv_folder, f'{commit}.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} not found.")
    
    developer_files = read_csv(csv_file)

    # 返回字典 {developer: 修改的文件列表}
    return developer_files


# find files edited in three months by the developer
def get_developer_files(fileGroup):
    fileLists = []
    fileGroup['date'] = pd.to_datetime(fileGroup['date'])
    newest_date = fileGroup['date'].max()
    # print(newest_date)
    fileGroup = fileGroup[(fileGroup['date']<=newest_date) & (fileGroup['date']>=(newest_date-pd.DateOffset(months=3)))]
    # print(fileGroup['date'])
    for i,row in fileGroup.iterrows():
        file = row['file'].replace('\\','/')
        fileLists.append(file)
    return fileLists

def preprocess_contents(file_contents):
    for key in file_contents:
        file_contents[key] = CS.preprocess(file_contents[key])
    return file_contents


def save_to_csv(data, filepath):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists: # head
            writer.writerow(['index','bugId','developersStructuralScattering','developersSemanticScattering'])
        for row in data:
            writer.writerow(row)
    csvfile.close()

def search_done_id(filepath):
    file_exists = os.path.isfile(filepath)
    if file_exists:
        temp = pd.read_csv(filepath)
        return set(temp['bugId'])
    return ()


def structrual_and_semantic_scattering(rec_lists):
    grouped = rec_lists.groupby('bugId')
    savePath = f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/temp/{i}.csv"
    done_id_set = search_done_id(savePath)
    for bugId,group in grouped:
        res_list = []  # output (by bugId)
        if bugId in done_id_set:
            continue
        developer_None=0
        bugCmit = search_bugCmit(bugId,dataset)
        print(f"bugid: {bugId} bugCmit: {bugCmit}")
        Developer_files = openDevelopers(dataset,bugCmit)
        file_contents = read_code_corpus(f"../../CodeCorpus/{dataset}/{bugCmit}.csv")
        file_contents = preprocess_contents(file_contents)
        developer_modified_dict = {}

        # 先根据 develop.csv的developer名单计算；按developer分组；然后根据时间取前三个月
        Developer_files['developer_name'] = Developer_files['developer_name'].fillna('NaN')
        developer_grouped = Developer_files.groupby('developer_name')
        for developer_name,fileGroup in developer_grouped:
            print(f"==============developer_name: {developer_name}")
            all_files = get_developer_files(fileGroup)
            filecount = len(all_files)
            print(f"developer:{developer_name},filecount:{filecount}")
            avg_distance, structuralScat = structural_scattering(all_files, filecount)
            print("structuralScat", format(structuralScat, '.4f'), "avg", format(avg_distance, '.4f'))
            avg_sim, semanticScat = semantic_scattering(all_files, filecount, file_contents)
            print("semanticScat", format(semanticScat, '.4f'), "avg", format(avg_sim, '.4f'))
            developer_modified_dict[developer_name] = [structuralScat, semanticScat]

        for index, file in group.iterrows():
            res = []
            filepaths ={'path_0':file['path_0'],
                        'path_1':file['path_1'],
                        'path_2':file['path_2']}
            developers = searchDeveloper(Developer_files,filepaths) #list
            if developers is None:
                StrScat = np.NAN
                SemScat = np.NAN
                developer_None +=1
                print(f"developer：BugId:{bugId},bugCmit:{bugCmit}\nfilepath:{filepaths}")
            else:
                developers = set(developers)
                sum_strScat, sum_semScat = 0,0
                for developer in developers:
                    if pd.isna(developer):
                        developer = ""
                    [structuralScat,semanticScat] = developer_modified_dict[developer]
                    sum_strScat += structuralScat
                    sum_semScat += semanticScat
                StrScat = sum_strScat/len(developers)
                SemScat = sum_semScat/len(developers)
                # print(f"Str: {StrScat} Sem: {SemScat}")
            # print(f">>>index:{index},developer:{developers},StrScat:{StrScat},SemScat:{SemScat}")
            res.append(index)
            res.append(bugId)
            res.append(StrScat)
            res.append(SemScat)
            res_list.append(res)
        save_to_csv(res_list,savePath)
        if developer_None != 0:
            print(f"=====BugId:{bugId},developer_None:{developer_None}")

    # df_scatter = pd.DataFrame(res_list,columns=['index','bugId','developersStructuralScattering','developersSemanticScattering'])
    df_scatter = pd.read_csv(savePath)
    # 与BuggyFileFeatures合并
    df_buggyFileFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_result = updateFeatures(df_buggyFileFeatures,df_scatter)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)


if __name__ == "__main__":
    configx = ConfigX()
    csv.field_size_limit(10000000)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    for dataset, file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: #zoo openjpa /aspectj /tomcat
            continue
        print(f"============Structural and Semantic Scattering: {dataset}============")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            structrual_and_semantic_scattering(readRecList(dataset,i))


