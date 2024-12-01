import re
import string
import pandas as pd
import numpy as np
import datetime
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import time
from xml.etree import ElementTree as ET


# 去茎化 / 去除停止词
from configx.configx import ConfigX


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    removed_stopwords = [
        stemmer.stem(item) for item in tokens if item not in stopwords.words("english")
    ]

    return removed_stopwords


# 去掉一般连词 / 全部小写 / 删除标点符号 / 标记化 / 词干
def normalize(text):
    remove_punc_map = dict((ord(char), None) for char in string.punctuation)
    removed_punc = text.lower().translate(remove_punc_map)
    tokenized = word_tokenize(removed_punc)
    stemmed_tokens = stem_tokens(tokenized)
    return stemmed_tokens


def hasStackTraces(text):
    isStackTraces = 0
    # print(text)
    # result = re.search(r'at(.*)\.(.*)\(([^:] *):?([\d]*)\)', text)
    result = re.search(r'at\s(.*)\.java\:\d', text)
    # return result.group()
    if(result != None):
        isStackTraces = 1
    return isStackTraces


def hasItemizations(text):
    isItemizations = 0
    for i in text.split("\n"):
        if i.startswith("1.") or i.startswith("1)") or i.startswith("+") or i.startswith("*"):
            isItemizations = 1
    return isItemizations


def hasStack(text):
    isStackTrace = 0
    if(re.search(r'at\s(.*)\.java\:\d', text) != None):
        isStackTrace = 1

    if(re.search(r'^\!SUBENTRY .*', text) != None):
        isStackTrace = 1
    if(re.search(r'^\!ENTRY .*', text) != None):
        isStackTrace = 1
    if(re.search(r'^\!MESSAGE .*', text) != None):
        isStackTrace = 1
    if(re.search(r'^\!STACK .*', text) != None):
        isStackTrace = 1
    if(re.search(r'^[\s]*at[\s]+.*[\n]?\([\w]+\.java(:[\d]+)?\)', text) != None):
        isStackTrace = 1
    if(re.search(r'^[\s]*([\w]+\.)+[\w]+(Exception|Error)(:[\s]+(.*\n)*.*)?', text) != None):
        isStackTrace = 1
    if "trace" in text:
        isStackTrace = 1
    return isStackTrace


def hasCode(text):
    isCode = 0
    code_class = re.search(r'^[\s]*(public|private|protected).*class[\s]+[\w]+[\s]', text)
    code_variable = re.search(r'^[\s]*(public|private|protected).*\(.*\)[\n]?', text)
    code_if = re.search(r'^[\s]*(if|for|while)[\s]*\(.*\)', text)
    codeSamples = re.search(r'\{(.*\n)*.*\}', text)
    code_import = re.search(r'import[\s]+.*;', text)
    if(code_class != None):
        isCode = 1
    elif(code_variable != None):
        isCode = 1
    elif(code_if != None):
        isCode = 1
    elif(codeSamples != None):
        isCode = 1
    elif(code_import != None):
        isCode = 1
    return isCode


def hasPatch(text):
    isPatch = 0
    patch = re.search(r'[-]{3}[\s].*\n[\+]{3}[\s].*\n[@]{2}', text)
    if(patch != None):
        # print(patch)
        isPatch = 1
    if "patch" in text or "fix" in text:
        isPatch = 1
    return isPatch


def hasScreenshot(text):
    isScreenshot = 0
    words = ['window', 'view', 'picture', 'screenshot', 'visible', 'image', 'png', 'bmp', 'jpg', 'jpeg', 'where to', 'screen shot', 'yellow', 'rectangle']
    for word in words:
        if word in text:
            # print(word)
            isScreenshot = 1
    return isScreenshot


# 计算每个词在bug报告中出现的频率，返回频率>=1%的关键词
def word_frequency(text):
    keywords = []
    cnt = Counter()
    for word in text:
        cnt[word] += 1
    cnt = dict(cnt)
    words_num = cnt.__len__()
    for key, value in cnt.items():
        if value >= words_num*0.01:
            keywords.append(key)
            # print(key + ":" + str(value))
    return keywords

#
# # zookeeper的关键词
# keywords_zookeeper = ['caus', 'zookeep', 'client', 'configur', 'creat', 'use', 'zk', 'tri', 'server', 'case', 'node', 'session', 'issu',
#      'check', 'problem', 'code', 'chang', 'snapshot', 'time', 'set', 'like', 'thread', 'data', 'leader', 'also',
#      'need', 'class', 'path', 'follow', 'zxid', 'file', 'get', 'return', 'observ', 'fail', 'test', 'except',
#      'packet', 'start', 'state', 'fix', 'socket', 'call', 'request', 'messag', 'close', 'connect', 'look', 'log', 'one',
#      'debug', 'error', 'null', 'method', 'new', 'id', 'info', 'elect', 'quorum', 'warn', 'cluster', 'junit',
#      'myid', 'run', 'javac', 'see']
#     # action_words = ['caus', ]
#
# # tomcat的关键词
# keywords_tomcat = ['work', 'follow', 'file', 'tomcat', 'fail', 'except', 'line', 'contain', 'method', 'use',
#                       'applic', 'like', 'new', 'request', 'exampl', 'call', 'server', 'configur', 'creat', 'return',
#                       'one', 'null', 'page', 'test', 'public', 'class', 'respons', 'throw', 'name', 'problem', 'time',
#                       'set', 'sourc', 'would', 'valu', 'fix', 'servlet', 'case', 'context', 'attach', 'process', 'start',
#                       'run', 'string', 'messag', 'tri', 'log', 'thread', 'issu', 'object', 'result', 'bug', 'error',
#                       'sever', 'code', 'see', 'webapp', 'session', 'chang', 'patch', 'jsp', 'get', 'caus', 'systemerr']
#
# # openjpa的关键词
# keywords_openjpa = ['string', 'version', 'test', 'fail', 'run', 'except', 'call', 'tabl', 'creat', 'openjpa', 'problem',
#                     'issu', 'caus', 'use', 'set', 'follow', 'error', 'type', 'method', 'like', 'entiti', 'public',
#                     'class', 'privat', 'id', 'queri', 'new', 'execut', 'null', 'persist', 'map', 'valu', 'select', 'gt',
#                     'see', 'result', 'column', 'return', 'case', 'code', 'work', 'object', '91main93', 'field', 'name',
#                     'sql', 'key', 'load', 'get', 'order', 'gener', 'sourc', 'trace', '91java93']

# keywords_aspectj = []

# text = "Hi, if I use the OpenJPA shipped with Spring 2.0.3, I got the following error when start application: " \
#         "org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'entityManagerFactory' defined in ServletContext resource &amp;#91;/WEB-INF/applicationContext.xml&amp;#93;: Invocation of init method failed; nested exception is java.lang.VerifyError: class loading constraint violated (class: org/apache/openjpa/kernel/BrokerImpl method: newQueryImpl(Ljava/lang/String;Lorg/apache/openjpa/kernel/StoreQuery;)Lorg/apache/openjpa/kernel/QueryImpl at pc: 0" \
#         "Caused by: java.lang.VerifyError: class loading constraint violated (class: org/apache/openjpa/kernel/BrokerImpl method: newQueryImpl(Ljava/lang/String;Lorg/apache/openjpa/kernel/StoreQuery;)Lorg/apache/openjpa/kernel/QueryImpl at pc: 0\n" \
#         "If I change to the latest openjpa 0.96 or 0.97 jar files downloaded from openjpa, I got the following error:\n"\
#         "1. at java.lang.J9VMInternals.verify(J9VMInternals.java:59)" \
#         "2. at java.lang.J9VMInternals.initialize(J9VMInternals.java:120)" \
#         "at java.lang.Class.forNameImpl(Native Method)" \

# print(stem_tokens(normalize(text)))
# word_frequency(stem_tokens(normalize(text)))


# filepath = 'data/tomcat_dat/bugs/tomcat_merged4tools'
# filepath = 'data/hibernate_dat/hibernate-orm_merged4tools.merged4tools'
# filepath = 'data/lucene_dat/lucene_merged4tools.merged4tools'
# filepath = 'data/openjpa_dat/openjpa_merged4tools.merged4tools.txt'
# filepath = 'data/zookeeper_dat/zookeeper_merged4tools.merged4tools'
# filepath = 'data/aspectj_dat/org.aspectj_merged4tools'
def read_bugId_from_csv(dataset):
    filepath = f"../data/get_info/{dataset}/bug_info.csv"
    df = pd.read_csv(filepath)
    return df['bugId'].tolist()

def read_xml(dataset,xmlPath):
    tree = ET.parse(xmlPath)  # load xml as ElementTree

    root = tree.getroot()   # root node
    # print(root.tag)

    all_brs = []
    bug_corpus = ""
    bug_ids = read_bugId_from_csv(dataset)
    for br in root.findall('item'):
        br_info = []

        bug_id = br.find("id").text
        try:
            id = int(bug_id)
            if id not in bug_ids:
                continue
        except ValueError:
            print(f"Skipping non-integer id:{bug_id}")
            continue

        # if id in bug_ids_list:
        # print(id)

        if dataset not in ['Tomcat']:
            summary = br.find("summary").text
        else:
            summary = br.find("short_desc").text   # tomcat


        description = br.find("description").text
        if description is None:
            bug_text = (summary + "")
        else:
            bug_text = (summary + description)

        # 经过预处理的bug报告文本
        pre_bug_text = normalize(bug_text)

        # 把所有bug报告文本集合成一个bug报告语料库
        bug_corpus += bug_text

        # bug报告文本中是否出现明细信息
        is_itemization = hasItemizations(bug_text)
        # print(is_itemization)

        # bug报告文本中是否出现堆栈跟踪信息
        # is_stacktrace = hasStackTraces(bug_text)
        # print(is_stacktrace)
        isStack = hasStack(bug_text)

        # bug报告文本中是否出现堆栈跟踪信息
        isCode = hasCode(bug_text)

        # bug报告是否含有patch
        patch = br.find("patch").text
        isPatch = hasPatch(bug_text)
        if patch == 1 or isPatch == 1:
            isPatch == 1

        # bug报告是否含有screenshot
        screenshots = br.find("screenshots").text
        isScreenshot = hasScreenshot(bug_text)
        if screenshots == 1 or isScreenshot == 1:
            isScreenshot == 1

        # bug报告文本中是否出现keywords
        is_keyword = 0
        for i in pre_bug_text:
            # if i in keywords_zookeeper:
            if i in configx.keywords[dataset]:
                # if i in keywords_openjpa:
                # if i in keywords_aspectj:
                # print("keywords:" + i)
                is_keyword = 1
                break

        br_info.append(id)
        br_info.append(summary)
        br_info.append(description)
        br_info.append(pre_bug_text)
        br_info.append(isPatch)
        br_info.append(isScreenshot)
        br_info.append(is_itemization)
        br_info.append(isStack)
        br_info.append(isCode)
        br_info.append(is_keyword)
        all_brs.append(br_info)
    return all_brs


# extract keyWords
def extractKeywords(dataset,xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()   # 获取根节点
    bug_corpus = ""

    for br in root.findall('item'):
        if dataset not in ['Tomcat']:
            summary = br.find("summary").text  # zookeeper / openjpa / aspectj
        else:
            summary = br.find("short_desc").text   # tomcat

        description = br.find("description").text
        if description is None:
            bug_text = (summary + "")
        else:
            bug_text = (summary + description)

        # 把所有bug报告文本集合成一个bug报告语料库
        bug_corpus += bug_text
    keywords = word_frequency(normalize(bug_corpus))
    print(keywords)


# <create_time>Fri, 21 Sep 2018 02:12:36 +0000</create_time>   # zookeeper / openjpa
# <resolve_time>Thu, 22 Nov 2018 16:56:24 +0000</resolve_time>
# < creation_ts > 2014-01-07 21:26:07 +0000 < / creation_ts >   # tomcat
# <delta_ts>2014-01-13 13:59:25 +0000</delta_ts>
def deal_time(text,dataset):
    # zookeeper
    if dataset in ['zookeeper','openjpa','lucene','hibernate']:
        text = text.split(", ")   # ['Fri', '21 Sep 2018 02:12:36 +0000']
    # # print(text)
        if dataset == 'hibernate':
            text = text[1].split(" -")  # ['21 Sep 2018 02:12:36', '0000']
        else:
            text = text[1].split(" +")   # ['21 Sep 2018 02:12:36', '0000']
    # # print(text)
        t = datetime.datetime.strptime(text[0], '%d %b %Y %H:%M:%S')   # 2018-09-21 02:12:36
    elif dataset == 'Tomcat':
    # tomcat
        text = text.split(" +")  # ['2014-01-07 21:26:07', '0000']
    # print(text)
        t = datetime.datetime.strptime(text[0], '%Y-%m-%d %H:%M:%S')  # 2014-01-07 21:26:07
    elif dataset == 'aspectj':
    # aspectj 2013-12-04 06:36:24 -0500
        text = text.split(" -")
        # print(text)
        t = datetime.datetime.strptime(text[0], '%Y-%m-%d %H:%M:%S')

    # 转成时间戳 1537467156.0
    t = time.mktime(t.timetuple())
    return t



def reporter_commit_num(dataset,xmlPath):
    tree = ET.parse(xmlPath)

    root = tree.getroot()  # 获取根节点
    # print(root.tag)

    all_brs = []

    for br in root.findall('item'):
        br_info = []

        bug_id = br.find("id").text
        try:
            id = int(bug_id)
        except ValueError:
            print(f"Skipping non-integer id:{bug_id}")
            continue

        # print(id)
        br_info.append(id)

        if dataset == 'hibernate':
            reporter = br.find('reporter').find('name').text
        else:
            reporter = br.find('reporter').find('username').text
        # print(reporter)
        br_info.append(reporter)

        if dataset in ['zookeeper','openjpa','lucene']:
            assignee = br.find('assignee').find('username').text
        elif dataset == 'hibernate':
            assignee = br.find('assignee').find('name').text
        elif dataset in ['Tomcat','aspectj']:
            assignee = br.find('assigned_to').find('username').text   # tomcat / aspectj
        else:
            print('Dataset Error:assignee')
        # print(assignee)
        br_info.append(assignee)

        if dataset in ['zookeeper','openjpa','aspectj','hibernate','lucene']:
            create_time = br.find('create_time').text   # zookeeper / openjpa / aspectj
        elif dataset in ['Tomcat']:
            create_time = br.find('creation_ts').text   # tomcat
        else:
            print('Dataset Error:create_time')
        # print(create_time)
        br_info.append(create_time)

        if dataset in ['zookeeper','openjpa','hibernate','lucene']:
            resolve_time = br.find('resolve_time').text   # zookeeper / openjpa
        elif dataset in ['Tomcat']:
            resolve_time = br.find('delta_ts').text   # tomcat
        elif dataset in ['aspectj']:
            resolve_time = br.find('delta_time').text   # aspectj
        else:
            print('Dataset Error:resolve_time')
        # print(resolve_time)
        br_info.append(resolve_time)

        # 遍历整个bug reports，在其它br小于create time的情况下，输出同一个reporter name出现的次数 commit_mnum
        # 输出这个reporter在其它br中出现为assignee的次数 assign_num
        # <resolution> Fixed </resolution> zookeeper
        # <resolution> FIXED </resolution> tomcat
        commit_num = 0
        assign_num = 0
        valid_commit_num = 0
        for i in root.findall('item'):
            # commit_num
            if dataset == 'Tomcat':
                if i.find('reporter').find('username').text == reporter and deal_time(
                        i.find('creation_ts').text,dataset) < deal_time(create_time,dataset):  # tomcat
                    commit_num += 1
            elif dataset == 'hibernate':
                if i.find('reporter').find('name').text == reporter and deal_time(
                        i.find('create_time').text, dataset) < deal_time(create_time, dataset):  # tomcat
                    commit_num += 1
            else:
                if i.find('reporter').find('username').text == reporter and deal_time(
                        i.find('create_time').text, dataset) < deal_time(create_time, dataset):
                    commit_num += 1
            # assign_num
            if dataset == 'Tomcat':
                if i.find('assigned_to').find('username').text == reporter and i.find('resolution').text == "FIXED" \
                        and deal_time(i.find('delta_ts').text,dataset) < deal_time(create_time,dataset):  # tomcat
                    assign_num += 1
            elif dataset == 'aspectj':
                if i.find('assigned_to').find('username').text == reporter and i.find('resolution').text == "FIXED" \
                        and deal_time(i.find('delta_time').text,dataset) < deal_time(create_time,dataset):  # aspectj
                    assign_num += 1
            elif dataset == 'hibernate':
                if i.find('assignee').find('name').text == reporter and i.find('resolution').text == "Fixed" \
                        and deal_time(i.find('resolve_time').text,dataset) < deal_time(create_time,dataset):
                    assign_num += 1
            else:
                if i.find('assignee').find('username').text == reporter and i.find('resolution').text == "Fixed" \
                        and deal_time(i.find('resolve_time').text,dataset) < deal_time(create_time,dataset):
                    assign_num += 1

            # valid_commit_num
            if dataset == 'Tomcat':
                if (i.find('resolution').text == "FIXED" and i.find('reporter').find('username').text == reporter
                        and deal_time(i.find('creation_ts').text,dataset) < deal_time(create_time,dataset)):  # tomcat
                    valid_commit_num += 1
            elif dataset == 'hibernate':
                if(i.find('resolution').text == "Fixed" and i.find('reporter').find('name').text == reporter
                        and deal_time(i.find('create_time').text,dataset) < deal_time(create_time,dataset)):
                    valid_commit_num += 1
            elif dataset == 'aspectj':
                if(i.find('resolution').text == "FIXED" and i.find('reporter').find('username').text == reporter
                        and deal_time(i.find('create_time').text,dataset) < deal_time(create_time,dataset)):
                    valid_commit_num += 1
            else:
                if(i.find('resolution').text == "Fixed" and i.find('reporter').find('username').text == reporter
                        and deal_time(i.find('create_time').text,dataset) < deal_time(create_time,dataset)):
                    valid_commit_num += 1


        # print(commit_num)
        # print(assign_num)
        # print(valid_commit_num)
        br_info.append(commit_num)
        br_info.append(assign_num)
        br_info.append(valid_commit_num)

        all_brs.append(br_info)
    # print(all_brs)
    return all_brs


if __name__ == "__main__":
    configx = ConfigX()
    print("Begin to Extract Bug Report Features")
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['aspectj']:
            continue
        print("=====Dataset:", dataset, "=====")

        # if you don't have keywords, use it
        # extractKeywords(dataset,file['bugReport'])

        # BugReportFeatures-part
        # br_features_list = read_xml(dataset,file['bugReport'])
        # br_features = pd.DataFrame(br_features_list, columns=['BugId', 'Summary', 'Description', 'Pre_bug_text',
        #                                                       'Patch', 'Screenshots', 'Itemization', 'StackTrace',
        #                                                       'CodeSample','Keywords'])
        # br_features.to_csv(f'../data/get_info/{dataset}/bugReportFeatures.csv',index=False)
        # print(f"BugReportFeatures of {dataset} is extracted")

        # reporters_features
        print(f"Begin to extract reporter's features.")
        reporter_features_list = reporter_commit_num(dataset,file['bugReport'])
        reporter_features = pd.DataFrame(reporter_features_list,
                                         columns=['BugId', 'Reporter', 'Assignee', 'create_time',
                                                  'resolve_time', 'commit_num', 'assign_num',
                                                  'valid_commit_num'])
        reporter_features.to_csv(f"../data/get_info/{dataset}/ReportersFeatures.csv",index=False)
        print(f"ReportersFeatures of {dataset} is extracted!")