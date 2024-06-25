# coding=utf-8
import xml.dom.minidom
import pandas as pd


# filepath = 'data/tomcat_dat/bugs/tomcat_merged4tools'
# filepath = 'data/hibernate_dat/hibernate-orm_merged4tools.merged4tools'
# filepath = 'data/lucene_dat/lucene_merged4tools.merged4tools'
# filepath = 'data/openjpa_dat/openjpa_merged4tools.merged4tools'
filepath = '../data/zookeeper_dat/zookeeper_merged4tools.merged4tools'
# filepath = 'data/aspectj_dat/org.aspectj_merged4tools'


# 从merged4tools文件中读取对应的bug id集合
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
    return res_df['bugId']

bug_ids = readBugId(filepath)
print(bug_ids)
# bug_ids.to_csv(path_or_buf='aspectj_bugid.csv')


# 预处理bug报告文本 分割成单词
def split_and_clear(text):
    import re
    rstr = r"[\#\!\.\{\}\;\_\-\[\]\=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"
    clear_text = re.sub(rstr, " ", text)
    # tokens = clear_text.split()
    # return tokens
    return clear_text


# 使用minidom解析器打开 XML 文档，读取上述对应bug id的具体项比如summary description
# DOMTree = xml.dom.minidom.parse("data/bug_report/tomcat.xml")
# DOMTree = xml.dom.minidom.parse("data/hibernate_dat/hibernate.xml")
# DOMTree = xml.dom.minidom.parse("data/lucene_dat/lucene.xml")
DOMTree = xml.dom.minidom.parse("data/openjpa_dat/openjpa.xml")
# DOMTree = xml.dom.minidom.parse("data/zookeeper_dat/zookeeper.xml")
# DOMTree = xml.dom.minidom.parse("data/aspectj_dat/aspectj.xml")

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
            print(bug_id)

            patch = bug_report.getElementsByTagName('patch')[0]
            # print("patch:" + patch.childNodes[0].data)
            br.append(patch.childNodes[0].data)

            screenshots = bug_report.getElementsByTagName('screenshots')[0]
            # print("screenshots:" + screenshots.childNodes[0].data)
            br.append(screenshots.childNodes[0].data)

            # openjpa没有keywords
            # keywords = bug_report.getElementsByTagName('keywords')[0]
            # if (keywords.childNodes[0].data == 'None'):
            #     kw = 0
            # else:
            #     kw = 1
            # br.append(kw)

            # tomcat
            # summary = bug_report.getElementsByTagName('short_desc')[0]
            # aspectj
            summary = bug_report.getElementsByTagName('summary')[0]
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
br_df = pd.DataFrame(br_list, columns=['BugId', 'Patch', 'Screenshots', 'Summary', 'Clear_Summary','Description',
                                       'Clear_Description', 'Reporter', 'Create_time', 'Commenter', 'Reviewer'])
print(br_df)
br_df.to_csv('data/openjpa_dat/openjpa.csv')
