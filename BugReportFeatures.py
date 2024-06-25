import re
import pandas as pd
import numpy as np
import datetime
import time
from xml.etree import ElementTree as ET


def hasStackTraces(text):
    isStackTraces = 0
    # print(text)
    # result = re.search(r'at(.*)\.(.*)\(([^:] *):?([\d]*)\)', text)
    result = re.search(r'at\s(.*)\.java\:\d', text)
    # return result.group()
    if(result != None):
        isStackTraces = 1
    return isStackTraces


def hasStack(text):
    isStackTrace = 0
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


# text = "Hi, if I use the OpenJPA shipped with Spring 2.0.3, I got the following error when start application: " \
#         "org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'entityManagerFactory' defined in ServletContext resource &amp;#91;/WEB-INF/applicationContext.xml&amp;#93;: Invocation of init method failed; nested exception is java.lang.VerifyError: class loading constraint violated (class: org/apache/openjpa/kernel/BrokerImpl method: newQueryImpl(Ljava/lang/String;Lorg/apache/openjpa/kernel/StoreQuery;)Lorg/apache/openjpa/kernel/QueryImpl at pc: 0" \
#         "Caused by: java.lang.VerifyError: class loading constraint violated (class: org/apache/openjpa/kernel/BrokerImpl method: newQueryImpl(Ljava/lang/String;Lorg/apache/openjpa/kernel/StoreQuery;)Lorg/apache/openjpa/kernel/QueryImpl at pc: 0" \
#         "at java.lang.J9VMInternals.verify(J9VMInternals.java:59)" \
#         "at java.lang.J9VMInternals.initialize(J9VMInternals.java:120)" \
#         "at java.lang.Class.forNameImpl(Native Method)" \
#         "If I change to the latest openjpa 0.96 or 0.97 jar files downloaded from openjpa, I got the following error:"
#
# print(hasStackTraces(text))


# filepath = 'data/tomcat_dat/bugs/tomcat_merged4tools'
# filepath = 'data/hibernate_dat/hibernate-orm_merged4tools.merged4tools'
# filepath = 'data/lucene_dat/lucene_merged4tools.merged4tools'
# filepath = 'data/openjpa_dat/openjpa_merged4tools.merged4tools'
filepath = 'data/zookeeper_dat/zookeeper_merged4tools.merged4tools'
# filepath = 'data/aspectj_dat/org.aspectj_merged4tools'


# 从merged4tools文件中提取实际使用的bug report的id集合
def readBugId(filepath):
    res_pred = []
    res = open(filepath,'r', encoding='utf-8')
    for line in res:
        res_pred.append(list(line.strip('\n').split('邹')))
    res.close()
    res_df = pd.DataFrame(res_pred, columns=['bugId', 'fixedCmit', 'cmitUnixTime', 'allFixedFiles', 'addFiles', 'addPackClass', 'delFiles',
                                             'delPackClass', 'modiFiles', 'modiPackClass', 'opendate', 'opendateUnixTime', 'fixdate',
                                             'fixdateUnixtTime', 'reporterName', 'reporterEmail', 'summary', 'description'])
    # print(res_df)
    res_df['bugId'] = res_df['bugId'].astype('int')
    return res_df['bugId']

# bug_ids = readBugId(filepath)
# print(bug_ids)
# bug_ids_arr = np.array(bug_ids) #先将数据框转换为数组
# bug_ids_list = bug_ids_arr.tolist()  #其次转换为列表
# print(bug_ids_list)


def read_xml():
    tree = ET.parse("data/zookeeper_dat/zookeeper.xml")   # 载入xml数据,使用ElementTree代表整个XML文档，Element代表这个文档树中的单个节点
    # print(tree)
    root = tree.getroot()   # 获取根节点
    # print(root.tag)

    all_brs = []

    for br in root.findall('item'):
        # br_info = {}

        # print(br.tag, br.attrib)

        id = int(br.find("id").text)

        if id not in bug_ids_list:
            print(id)
            root.remove(br)

        tree.write("data/zookeeper_dat/output.xml")

    #         # br_info[child.tag] = child.text
    #     all_brs.append(br_info)
    # print(all_brs)

# read_xml()


def reporter_commit_num():
    tree = ET.parse("data/zookeeper_dat/zookeeper.xml")
    root = tree.getroot()  # 获取根节点
    # print(root.tag)

    all_brs = []
    for br in root.findall('item'):
        id = int(br.find("id").text)
        reporter = br.find('reporter').find('username').text
        print(reporter)
        assignee = br.find('assignee').find('username').text
        print(assignee)
        create_time = br.find('create_time').text
        print(create_time)
        resolve_time = br.find('resolve_time').text
        print(resolve_time)

        # <status>Resolved</status>

        # 遍历整个bug reports，在其它br小于create time的情况下，输出同一个reporter name出现的次数 commit_mnum
        # 输出这个reporter在其它br中出现为assignee的次数 assign_num
        commit_num = 0
        assign_num = 0
        valid_commit_num = 0
        for i in root.findall('item'):
            if(i.find('reporter').find('username').text == reporter and deal_time(i.find('create_time').text) < deal_time(create_time)):
                commit_num += 1
            if(i.find('assignee').find('username').text == reporter and deal_time(i.find('resolve_time').text) < deal_time(create_time)):
                assign_num += 1
            if(i.find('status').text == "Resolved" and i.find('reporter').find('username').text == reporter
                    and deal_time(i.find('create_time').text) < deal_time(create_time)):
                valid_commit_num += 1

        print(commit_num)
        print(assign_num)
        print(valid_commit_num)


# reporter_commit_num()

# <create_time>Fri, 21 Sep 2018 02:12:36 +0000</create_time>
# <resolve_time>Thu, 22 Nov 2018 16:56:24 +0000</resolve_time>
def deal_time(text):
    text = text.split(", ")   # ['Fri', '21 Sep 2018 02:12:36 +0000']
    text = text[1].split(" +")   # ['21 Sep 2018 02:12:36', '0000']
    # print(text)
    t = datetime.datetime.strptime(text[0], '%d %b %Y %H:%M:%S')   # 2018-09-21 02:12:36
    t = time.mktime(t.timetuple())   # 转成时间戳 1537467156.0
    return t

# text = "Fri, 21 Sep 2018 02:12:36 +0000"
# print(deal_time(text))


def BRFeatures():
    tree = ET.parse("data/zookeeper_dat/zookeeper.xml")
    root = tree.getroot()  # 获取根节点

    all_brs = []
    for br in root.findall('item'):
        id = int(br.find("id").text)
        # print("id: ", id)
        summary = br.find("summary").text
        description = br.find("description").text
        if description is None:
            text = summary + ""
        else:
            text = summary + description
        # print(text)

        is_stacktrace = hasStackTraces(text)
        isStack = hasStack(text)
        isCode = hasCode(text)
        patch = br.find("patch").text
        isPatch = hasPatch(text)
        if patch == 1 or isPatch == 1:
            isPatch == 1
        screenshots = br.find("screenshots").text
        isScreenshots = hasScreenshot(text)
        if screenshots == 1 or isScreenshots == 1:
            isScreenshots == 1

        if is_stacktrace != isStack:
            print("id: ", id)
            print(is_stacktrace)
            print(isStack)
        # print(isStack)
        # print(isCode)
        # print(isPatch)
        # print(isScreenshots)

BRFeatures()


