import os
import re
import xml.dom.minidom
import pandas as pd

def readBugId(filepath):
    res_pred = []
    res = open(filepath,'r', encoding='utf-8')
    for line in res:
        bugInfo = line.split('邹')
        if not bugInfo[0].isdigit():
            continue
        bugInfo[-1] = bugInfo[-1].replace('傻','\n').replace('\xa0','')
        res_pred.append(bugInfo)
        # if bugInfo[0] == '51340':
        #     res_pred.append(bugInfo)

    res.close()
    res_df = pd.DataFrame(res_pred, columns=['bugId', 'fixedCmit', 'cmitUnixTime', 'allFixedFiles', 'addFiles', 'addPackClass', 'delFiles',
                                             'delPackClass', 'modiFiles', 'modiPackClass', 'opendate', 'opendateUnixTime', 'fixdate',
                                             'fixdateUnixtTime', 'reporterName', 'reporterEmail', 'summary', 'description'])

    return res_df,res_df['bugId']

# 预处理bug报告文本 分割成单词
def split_and_clear(text):
    import re
    rstr = r"[\#\!\.\{\}\;\_\-\[\]\=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"
    clear_text = re.sub(rstr, " ", text)
    # tokens = clear_text.split()
    # return tokens
    return clear_text


# BugReport
def readBugReport(dataset,filepath,bug_ids):
    DOMTree = xml.dom.minidom.parse(filepath)
    items = DOMTree.documentElement

    # 获取所有bug报告
    bug_reports = items.getElementsByTagName("item")
    br_list = []

    for bug_id in bug_ids:

        for bug_report in bug_reports:
            br = []

            id = bug_report.getElementsByTagName('id')[0]
            if (id.childNodes[0].data == bug_id):
                br.append(id.childNodes[0].data)


                patch = bug_report.getElementsByTagName('patch')[0]
                # print("patch:" + patch.childNodes[0].data)
                br.append(patch.childNodes[0].data)

                screenshots = bug_report.getElementsByTagName('screenshots')[0]
                # print("screenshots:" + screenshots.childNodes[0].data)
                br.append(screenshots.childNodes[0].data)

                # openjpa没有keywords
                # zookeeper 无 keywords
                if(dataset in ('zookeeper','openjpa')):
                    kw = 0
                # keywords = bug_report.getElementsByTagName('keywords')[0]
                # if (keywords.childNodes[0].data == 'None'):
                #     kw = 0
                # else:
                #     kw = 1
                br.append(kw)

                # tomcat
                # summary = bug_report.getElementsByTagName('short_desc')[0]
                # aspectj
                if(dataset in ('zookeeper','openjpa')):
                    summary = bug_report.getElementsByTagName('summary')[0]
                else:
                    summary = bug_report.getElementsByTagName('short_desc')[0]
                br.append(summary.childNodes[0].data)
                br.append(split_and_clear(summary.childNodes[0].data))

                try:
                    description = bug_report.getElementsByTagName('description')[0]
                    br.append(description.childNodes[0].data)
                    br.append(split_and_clear(description.childNodes[0].data))
                except IndexError:
                    br.append("NULL")
                    br.append("NULL")
                    print("des" + bug_id)

                reporters = bug_report.getElementsByTagName('reporter')
                for reporter in reporters:
                    reporterName = reporter.getElementsByTagName('name')[0]
                    br.append(reporterName.childNodes[0].data)
                    # print(reporterName.childNodes[0].data)

                create_time = bug_report.getElementsByTagNameNS('*', 'create_time')[0]
                br.append(create_time.childNodes[0].data)

                commenters = []
                try:
                    comments = bug_report.getElementsByTagName('comments')
                    for comment in comments:
                        # commenter = comment.getElementsByTagName('commenter')[0]
                        # comment_time = comment.getElementsByTagName('time')[0]
                        # openjpa
                        commenter = comment.getElementsByTagName('author')[0]
                        commenters.append(commenter.childNodes[0].data)
                        # comment_time = comment.getElementsByTagName('create_time')[0]
                        # br.append(commenter.childNodes[0].data)
                        # br.append(comment_time.childNodes[0].data)
                        # print(reporterName.childNodes[0].data)
                except IndexError:
                    commenters = ""
                    # print("comment" + bug_id)
                # print(commenters)
                br.append(commenters)

                # reviewer = bug_report.getElementsByTagName('assigned_to')[0]
                # openjpa
                reviewers = bug_report.getElementsByTagName('assignee')
                for reviewer in reviewers:
                    reviewerName = reviewer.getElementsByTagName('name')[0]
                    br.append(reviewerName.childNodes[0].data)

                br_list.append(br)

    # print(br_list)
    # br_df.to_csv('aspectj_bugid1.csv')
    # br_df = pd.DataFrame(br_list, columns=['BugId', 'Patch', 'Screenshots', 'Keywords', 'Summary', 'Description',
    #                                        'Reporter', 'Create_time', 'Commenter', 'Comment_time', 'Reviewer'])
    br_df = pd.DataFrame(br_list, columns=['BugId', 'Patch', 'Screenshots', 'Keywords','Summary', 'Clear_Summary', 'Description',
                                           'Clear_Description', 'Reporter', 'Create_time', 'Commenter', 'Reviewer'])
    return br_df

# 读取ground truth文件到dataframe
def readGroundTruth(filepath,approach,dataset):
    res_pred = []
    res = open(filepath,'r')
    for line in res:
        split_line = list(line.strip('\n').split(','))
        res_pred.append(split_line)
    res.close()

    # if approach == 'amalgam':
    #     for line in res:
    #         split_line = list(line.strip('\n').split('	'))
    #         res_pred.append(split_line)
    #     res.close()
    #
    # else:
    #     for line in res:
    #         split_line = list(line.strip('\n').split(','))
    #         if approach == 'blizzard'and split_line[0].isdigit() and ('src' in split_line[2]) and dataset =='zookeeper':
    #             split_line[2] = 'src' + split_line[2].split('src',1)[1]
    #         if split_line[0].isdigit():
    #             res_pred.append(split_line)
    #     res.close()

    res_df = pd.DataFrame(res_pred, columns=['BugId','SourceFile'])
    # if(approach == 'blizzard'):
    #     res_df = pd.DataFrame(res_pred, columns=['BugId', 'Rank', 'SourceFile'])
    # else:
    #     res_df = pd.DataFrame(res_pred, columns=['BugId', 'SourceFile', 'Rank', 'Score'])

    return res_df


# RecommendList(Top20)
def readRecommendList(approach,dataset,filepath,groundTruthPath):
    files = os.listdir(filepath)
    res_recommend = []

    for fi in files:
        count = 0
        # merge filepath and filename
        fi_d = os.path.join(filepath, fi)
        res = open(fi_d, 'r')
        # load top20 files
        for rr in res:
            res_list = []
            if (count == 20):
                break
            res_list += list(rr.strip('\n').split(','))
            res_recommend.append(res_list)
            count += 1
        res.close()

    gt = readGroundTruth(groundTruthPath,approach,dataset)
    gt['label'] = 1

    if(approach == 'blizzard'):
        # bilizard without Score
        dataframe = pd.DataFrame(res_recommend, columns=['BugId', 'Rank', 'SourceFile'])
        df = pd.merge(dataframe, gt, on=['BugId', 'SourceFile'], how='left')
    elif(approach == 'amalgam'):
        dataframe = pd.DataFrame(res_recommend, columns=['BugId', 'Rank', 'Score', 'SourceFile'])
        dataframe = dataframe.reindex(columns =['BugId', 'Rank', 'SourceFile', 'Score'])
        df = pd.merge(dataframe, gt, on=['BugId', 'SourceFile'], how='left')
    else:
        dataframe = pd.DataFrame(res_recommend, columns=['BugId', 'Rank', 'SourceFile', 'Score'])
        df = pd.merge(dataframe, gt, on=['BugId', 'SourceFile'], how='left')
    df.fillna(0, inplace=True)
    df = df.astype({"label": int})

    return df