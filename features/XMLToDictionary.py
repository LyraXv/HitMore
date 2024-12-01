import re

import pandas as pd
from xml.etree import ElementTree as ET
import csv
import string

from configx.configx import ConfigX


def xml_parser(dataset,xmlPath):
    tree = ET.parse(xmlPath)   # 载入xml数据,使用ElementTree代表整个XML文档，Element代表这个文档树中的单个节点
    # tree = ET.parse("data/tomcat_dat/tomcat.xml")
    # tree = ET.parse("data/openjpa_dat/openjpa.xml")

    root = tree.getroot()  # 获取根节点

    all_brs = []

    for br in root.findall('bug'):
        br_info = []
        attrib = br.attrib
        id = attrib.get('id')
        report_time = attrib.get('opendate')
        commit_time = attrib.get('fixdate')
        buginfo = br.find('buginformation')
        summary = buginfo.find('summary').text
        description = buginfo.find('description').text

        fixedFiles = br.find('fixedFiles')
        # fixedFiles = br.find('fixedFiles').getchildren()
        files = ""
        for file in fixedFiles:
            files += file.text
        # print(files)

        br_info.append(id)
        br_info.append(summary)
        br_info.append(description)
        br_info.append(report_time)
        br_info.append(commit_time)
        br_info.append(files)
        all_brs.append(br_info)

    return all_brs


# br_list = xml_parser()
# br_df = pd.DataFrame(br_list, columns=['bug_id', 'summary', 'description', 'report_time', 'commit_time', 'files'])
# print(br_df)
# br_df.to_csv('data/zookeeper_dat/bug_report.csv', index=False)


'''
Function to create a combined corpus out of a bug report
@param a bug report
'''
def getCombinedCorpus(report):
    report["summary"] = cleanAndSplit(report["summary"])
    report["description"] = cleanAndSplit(report["description"])
    combinedCorpus = report["summary"] + report["description"]
    return combinedCorpus


'''
Function to remove all punctuation and split text strings into lists of words
'''
def cleanAndSplit(text):
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    returnText = text.translate(replace_punctuation)
    returnText = [s.strip() for s in returnText.split(" ") if s];
    return returnText


'''
A method to open a CSV file and read in data.  Converts a space delimited set of file names into a list
'''
def CSVToDictionary(dataset):
    reader = csv.DictReader(open(f'../data/get_info/{dataset}/bug_report_fixedfiles.csv', 'rt', encoding='utf-8'), delimiter=',')
    dict_list = []
    for line in reader:
        if not line['bug_id'].isdigit():
            line['bug_id'] = re.sub(r'\D','',line['bug_id'])
        # Strip files string & split by .java (spaces in filenames). Discard empty strings & reappend .java to filenames
        line["files"] = [s.strip() + ".java" for s in line["files"].split(".java") if s]

        line["rawCorpus"] = line["summary"] +". "+line["description"]
        # print(line['rawCorpus'])

        # Change summary & description string into list of words
        # combinedCorpus = getCombinedCorpus(line)
        '''
        # Create a dictionary with a term frequency for each term
        d = dict.fromkeys(combinedCorpus, 0)
        for i in range(len(combinedCorpus)):
            if combinedCorpus[i] in d:
                d[combinedCorpus[i]] = d[combinedCorpus[i]] + 1
            else:
                print "error for index " + i
        line["termCounts"] = d
        '''
        dict_list.append(line)
    return dict_list


if __name__ == '__main__':
    configx = ConfigX()
    for dataset, file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']: #zookeeper
            continue
        # XMLToDictionary.py(contain fiexdFiles)
        print(f"=====Begin to save Bug Reports info:{dataset}=====")
        br_df = pd.DataFrame(xml_parser(dataset, file['bugReports']),
                             columns=['bug_id', 'summary', 'description', 'report_time', 'commit_time', 'files'])
        br_df.to_csv(f"../data/get_info/{dataset}/bug_report_fixedfiles.csv", index=False)
        print(f"{dataset}'s bug reports with fixedfiles is saved")

