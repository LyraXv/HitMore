#从bug报告的CCList和Comments中提取报告人员和开发人员，以及那些对代码文件做出贡献的人员(通过挖掘Git提交历史)，以检查他们之间有多少人重叠。
import csv
import math
import os
import re
import xml.etree.ElementTree as ET

# 分割contributor字符串为姓名和邮箱，再通过邮箱提取用户名，如Benjamin Reed <breed@apache.org>
import pandas as pd

from configx.configx import ConfigX
from features.utils_features import readRecList, search_bugCmit, searchContributors, updateFeatures


def parse_contributor(contributor):
    match = re.match(r'([^<]+) <([^>]+)>', contributor)
    # print("      ",match)
    if match:
        name = match.group(1).strip()
        email = match.group(2).strip()
        # 如果name都是小写字母且无空格
        if(bool(re.match(r'^[a-z]+$', name)) == True):
            username = name
        elif dataset == 'Tomcat':
            parts = name.split()
            if len(parts) > 2:
                username = parts[0] + ' ' + parts[-1]
            else:
                username = name
        elif dataset == 'hibernate':
            username = name
        else:
            username = email.split('@')[0]
        # return name, email, username

        return username   #对于zookeeper等返回username
        # return name  # 对于tomcat和aspectj返回name
    return contributor.strip(), None


def get_contributorss(commit, file):
    # 构建CSV文件的名称
    filename = f"{commit}.csv"
    filepath = os.path.join('Contributors', filename)

    # 检查文件是否存在
    if not os.path.isfile(filepath):
        print(f"文件 {filepath} 不存在.")
        return ""

    contributors = ""

    # 读取Contributors文件下的CSV文件
    with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)

        # 跳过CSV文件的标题行
        next(csv_reader)

        # 遍历每一行
        for row in csv_reader:
            if file in row[0]:
                # 如果文件路径匹配，提取贡献者名字
                # 合并后续所有列并分割贡献者名字
                contributors = ','.join(row[1:]).split(',')
                break

    # 解析贡献者名字和邮箱，提取name/username
    parsed_contributors = [parse_contributor(contributor) for contributor in contributors]
    return parsed_contributors
    # return contributors


def getHibernateDict():
    xmlPath = configx.filepath_dict['hibernate']['bugReport']
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    hibernate_dict = {}

    for br in root.findall('item'):
        reporter = br.find('reporter')
        username = reporter.find('name').text
        account_id = reporter.find('account_id').text
        if account_id not in hibernate_dict:
            hibernate_dict[account_id] = username
    return hibernate_dict

def getAspectjDict():
    xmlPath = configx.filepath_dict['aspectj']['bugReport']
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    aspectj_dict = {}

    for br in root.findall('item'):
        reporter = br.find('reporter')
        name = reporter.find('name').text
        username = reporter.find('username').text
        if username not in aspectj_dict:
            aspectj_dict[name] = username
    return aspectj_dict

def get_reporters_and_developers(bug_id):
    xmlPath = configx.filepath_dict[dataset]['bugReport']
    tree = ET.parse(xmlPath)   # load xml data
    root = tree.getroot()   # get the root node

    for br in root.findall('item'):
        id_text = ''.join(re.findall(r'\d+',br.find("id").text))
        id = int(id_text)

        if id == bug_id:
            # Storing a list of reporter and comments authors
            rep_dev = []

            # extract reporter
            reporter = br.find('reporter')

            if dataset in ['zookeeper','openjpa','aspectj']:
                username = reporter.find('username').text
            elif dataset in ['Tomcat','hibernate']:
                username = reporter.find('name').text
            elif dataset == 'lucene':
                username = reporter.find('username').text
                if '@' in username:
                   username = username.split('@')[0]

            rep_dev.append(username)

            comments = br.find('comments')
            for comment in comments.findall('value'):
                if dataset in ['zookeeper','openjpa','hibernate']:
                    developer = comment.find('author').text
                elif dataset == 'Tomcat':
                    developer = comment.find('developer').text
                elif dataset == 'lucene':
                    developer = comment.find('author').text
                    if '@' in developer:
                        developer= developer.split('@')[0]
                else: #aspectj
                    name = comment.find('commenter').text
                    try:
                        developer = aspectj_dict[name]
                    except KeyError:
                        developer = name
                if dataset == 'hibernate':
                    try:
                        developer = hibernate_dict[developer]
                    except KeyError:
                        developer = ""

                rep_dev.append(developer)
            return rep_dev
    return None


def calculate_overlap(contributors, rep_dev):
    # Calculate the overlap between contributors and reporters/developers
    overlap = set(contributors) & set(rep_dev)
    overlap_count = len(overlap)
    if(min(len(contributors), len(rep_dev))) == 0:
        return 0
    return overlap_count / min(len(contributors), len(rep_dev))

def  load_contributors(dataset,bugCmit):
    csv_data = []
    filepath = f"../../Contributors/{dataset}/{bugCmit}.csv"
    with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip title line
        next(csv_reader)
        # Read each row of data and add it to the csv_data list
        for row in csv_reader:
            file = row[0]
            contributors = ','.join(row[1:]).split(',')
            csv_data.append([file,contributors])
    return pd.DataFrame(csv_data,columns=['file','contributors'])

def calculate_developersAndReporters(rec_lists):
    grouped = rec_lists.groupby('bugId')
    res_list = []  # output
    for bugId, group in grouped:
        rep_dev = list(set(get_reporters_and_developers(bugId)))
        print("BUGID: ",bugId)
        Contributors_None = 0
        bugCmit = search_bugCmit(bugId, dataset)
        Contributors = load_contributors(dataset,bugCmit)
        for index, file in group.iterrows():
            res = []
            filepaths = {'path_0': file['path_0'],
                         'path_1': file['path_1'],
                         'path_2': file['path_2']}
            unparsed_contributors = searchContributors(Contributors, filepaths)
            if unparsed_contributors is None:
                overlap = math.nan
                Contributors_None += 1
            else:
                contributors = [parse_contributor(contributor) for contributor in unparsed_contributors]
                overlap = calculate_overlap(contributors, rep_dev)
                # print(f"index:{index}, contributors:{contributors} overlap:{overlap}")

            res.append(index)
            res.append(bugId)
            # res.append(contributors)
            # res.append(rep_dev)
            res.append(overlap)
            res_list.append(res)
        if Contributors_None != 0:
            print(f"=====BugId:{bugId},CodeCorpus_None:{Contributors_None}")
    # df_example = pd.DataFrame(res_list,columns=['index','bugId','contributors','rep_dev','developersAndReporters'])
    # df_example.to_csv("../data/test_overlap.csv")
    df_overlap = pd.DataFrame(res_list,columns=['index','bugId','developersAndReporters'])
    df_relationFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv")
    df_result = updateFeatures(df_relationFeatures,df_overlap)
    df_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv",index=False)

if __name__ == "__main__":
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: #zoo openjpa tomcat///aspectj hibernate lucene???
            continue
        if dataset == 'hibernate':
            hibernate_dict = getHibernateDict()
        if dataset == 'aspectj':
            aspectj_dict = getAspectjDict()
        print(f"=====Develope and Reporters(Overlap): {dataset}=====")
        for i in [0,1,2,3,4,'otherTrulyBuggyFiles']:
            if i !='otherTrulyBuggyFiles':
                continue
            rec_lists = readRecList(dataset,i)
            calculate_developersAndReporters(rec_lists)

