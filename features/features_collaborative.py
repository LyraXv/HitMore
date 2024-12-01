import re
from datetime import datetime
from features import XMLToDictionary as XD
import pandas as pd
import similarity as CS
from configx.configx import ConfigX
from features.utils_features import readRecList, updateFeatures


def checkFileName(filename,Fixedfiles): #dict str
    # path_2>1>0
    if pd.notna(filename['path_2']) and filename['path_2'] in Fixedfiles:
        return True
    elif pd.notna(filename['path_1']) and(re.sub(r'\.(?!java)', '/', filename['path_1'])) in Fixedfiles:
        return True
    elif pd.notna(filename['path_0']) and(re.sub(r'\.(?!java)', '/', filename['path_0'])) in Fixedfiles:
        return True
    else:
        return False



'''
Function to return a list of previously filed bug reports that share a file with the current bug report
@params given file name in a Bug Report, the BR's date, and the dictionary of all BRs
'''
def getPreviousReportByFilename(filename, brdate, dictionary):
    return [br for br in dictionary if (checkFileName(filename,br['files']) and convertToDateTime(br["report_time"]) < brdate)]
    # return [br for br in dictionary if (filename in br["files"] and convertToDateTime(br["report_time"]) < brdate)]


'''
Helper function to convert from string to DateTime
@params the Date to be converted
'''
def convertToDateTime(date):
    return datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    # return datetime.strptime(date, "%m/%d/%Y %H:%M")


'''
Helper function to calculate the number of months between two date strings
@param the first date, the second date
'''
def getMonthsBetween(d1, d2):
    date1 = convertToDateTime(d1)
    date2 = convertToDateTime(d2)
    return abs((date1.year - date2.year) * 12 + date1.month - date2.month)


'''
Function to return the most recently submitted previous report that shares a filename with the given bug report
@params The filename in question, the date the current bug report was submitted, the dictionary of all bug reports
'''
def getMostRecentReport(filename, currentDate, dictionary):
    matchingReports = getPreviousReportByFilename(filename, currentDate, dictionary)
    if len(matchingReports) > 0:
        # Custom-define a lambda function to search the dictionary object for the Bug Report's time and sort by that
        return max((br for br in matchingReports), key=lambda x:convertToDateTime(x.get("report_time")))
    else:
        return None

def getMostRecentReport(matchingReports): #preReports
    if len(matchingReports) > 0:
        # Custom-define a lambda function to search the dictionary object for the Bug Report's time and sort by that
        return max((br for br in matchingReports), key=lambda x:convertToDateTime(x.get("report_time")))
    else:
        return None
'''
Calculate the Bug Fixing Recency as defined by Lam et al.
@params current bug report, most recent bug report
'''
def bugFixingRecency(report1, report2):
    if report1 is None or report2 is None:
        return 0
    else:
        return 1/float(getMonthsBetween(report1.get("report_time"), report2.get("report_time")) + 1)


'''
Calculate the Bug Fixing Frequency as defined by Lam et al.
@params filename fixed by BR, date of current BR, dictionary of all Bug Reports
'''
def bugFixingFrequency(filename, date, dictionary):
    return len(getPreviousReportByFilename(filename, date, dictionary))


'''
Calculate the collaborative filter score as defined by Lam et al.
@params The bug report we're calculating metadata for, the filename we're checking previous bug reports for
'''
def collaborativeFilteringScore(report, filename, dictionary):
    matchingReports = getPreviousReportByFilename(filename, convertToDateTime(report.get("report_time")), dictionary)
    # Get combined text of matching reports and do some rVSM stuff with it

def getFileName(filepath):
    if pd.notna(filepath['path_2']):
        filename = filepath['path_2'].split("/")[-1]
    elif pd.notna(filepath['path_1']):
        filename = filepath['path_1']
    elif pd.notna(filepath['path_0']):
        filename = filepath['path_0']
    else:
        print(f"CurrentPath:{filepath} exists ERROR")
        return None
    return filename.split(".")[-2]



def getbugFeatures_four(dataset,recList):
    allBugReports = XD.CSVToDictionary(dataset)
    list_file = []
    list_relation = []
    grouped = recList.groupby('bugId')
    for bugid,group in grouped:
        # print(bugid)
        [report] = list(filter(lambda x: x['bug_id'] == str(bugid), allBugReports))
        date = convertToDateTime(report.get("report_time"))
        rawCorpus = report["rawCorpus"]
        # files = report["files"]
        summary = report["summary"]
        for index,file in group.iterrows():
            res_file = []
            res_relation = []
            filepath = {'path_0':file['path_0'],'path_1':file['path_1'],'path_2':file['path_2']}
            # Collaboratice Filter Score
            prevReports = getPreviousReportByFilename(filepath, date, allBugReports)
            # print(f"index:{index},bugid:{bugid},PrevReports:{prevReports}")
            relatedCorpus = []
            for preReport in prevReports:
                relatedCorpus.append(preReport["summary"])
            relatedString = ' '.join(relatedCorpus)
            # print(rawCorpus)
            corpus = CS.preprocess(rawCorpus)
            preString = CS.preprocess(relatedString)

            str_corpus = " ".join(corpus)
            srt_relatedStr = " ".join(preString)

            # print(str_corpus)
            # print(srt_relatedStr)

            collaborativeFilterScore = CS.cosine_sim([str_corpus,srt_relatedStr])

            # Class Name Similarity
            rawClassNames = getFileName(filepath)
            # print(f"rawNames{rawClassNames}")
            name_len = len(rawClassNames)
            if rawClassNames in summary:
                # print(summary)
                classNameSimilarity = name_len
                # print("class:", rawClassNames, classNameSimilarity)
            else:
                classNameSimilarity = 0
            # print(f"classNameSimilarilty",classNameSimilarity)
            # Bug Fixing Recency
            mrReport = getMostRecentReport(prevReports)
            # print("mrReport", mrReport.get("bug_id"), mrReport.get("report_time"), mrReport.get("files"))
            bFRecency = bugFixingRecency(report, mrReport)
            # print(bFRecency)

            # Bug Fixing Frequency
            bFFrequency = bugFixingFrequency(file, date, allBugReports)
            # print(bFFrequency)

            res_file.append(index)
            res_file.append(bugid)
            res_file.append(bFRecency)
            res_file.append(bFFrequency)

            res_relation.append(index)
            res_relation.append(bugid)
            res_relation.append(collaborativeFilterScore)
            res_relation.append(classNameSimilarity)

            list_file.append(res_file)
            list_relation.append(res_relation)
    df_file = pd.DataFrame(list_file,columns=['index','bugId','bugFixingRecency','bugFixingFrequency'])
    df_relation = pd.DataFrame(list_relation,columns=['index','bugId','collaborativeFilterScore','classNameSimilarity'])

    # merge
    df_buggyFileFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_fileFeature_result = updateFeatures(df_buggyFileFeatures,df_file)
    df_fileFeature_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)

    df_relationFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv")
    df_relationFeature_result = updateFeatures(df_relationFeatures,df_relation)
    df_relationFeature_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/relationFeatures/{i}.csv",index=False)

            # 将计算得到的值存入 recList 的相应行
            # recList.at[index, 'collaborativeFilterScore'] = collaborativeFilterScore
            # recList.at[index, 'classNameSimilarity'] = classNameSimilarity
            # recList.at[index, 'bFRecency'] = bFRecency
            # recList.at[index, 'bFFrequency'] = bFFrequency

if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====four features: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            getbugFeatures_four(dataset,readRecList(dataset,i))
