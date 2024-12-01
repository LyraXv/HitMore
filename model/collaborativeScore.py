from datetime import datetime
from features import XMLToDictionary as XD
import similarity as CS
from configx.configx import ConfigX

'''
Function to return a list of previously filed bug reports that share a file with the current bug report
@params given file name in a Bug Report, the BR's date, and the dictionary of all BRs
'''
def getPreviousReportByFilename(filename, brdate, dictionary):
    return [br for br in dictionary if (filename in br["files"] and convertToDateTime(br["report_time"]) < brdate)]


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


def getTop20Files(bugId, rightFiles, brSummary):
    filepath = 'data/bugLocator/zookeeper/bugFixedFileRanks/' + bugId + '.txt'
    res = open(filepath, 'r')
    # 读取文件中前二十行数据
    count = 0
    res_recommend = []
    for rr in res:
        res_list = []
        if (count == 20):
            break
        res_list += list(rr.strip('\n').split(','))
        res_recommend.append(res_list)
        count += 1
    res.close()

    files = []
    for i in res_recommend:
        files.append(i[2])

    allFiles = []
    for file in files:
        filename = file.split(".")
        rawClassNames = filename[-2]
        name_len = len(rawClassNames)
        if rawClassNames in brSummary:
            print(brSummary)
            cns = name_len
            print("class:", rawClassNames, cns)
        else:
            cns = 0

        fileInfo = [file, cns]
        allFiles.append(fileInfo)

    return allFiles

def getbugFeatures_four(dataset):
    all_br = []
    allBugReports = XD.CSVToDictionary(dataset)
    for report in allBugReports:
        bug_id = report["bug_id"]
        date = convertToDateTime(report.get("report_time"))
        rawCorpus = report["rawCorpus"]
        files = report["files"]
        summary = report["summary"]

        print("===== " + report["bug_id"] + " =====")

        # 根据TOP20去计算（可以先提整个数据集的特征，之后再考虑划分）
        # 根据 bugid
        for file in top20:
            # 这里的file指的是filepath，注意我们的top20是有3条path,可以构成dict
            br = []
            # Collaboratice Filter Score
            prevReports = getPreviousReportByFilename(file, date, allBugReports)
            relatedCorpus = []
            for preReport in prevReports:
                relatedCorpus.append(preReport["summary"])
            relatedString = ' '.join(relatedCorpus)
            collaborativeFilterScore = CS.cosine_sim(rawCorpus, relatedString)
            # print(collaborativeFilterScore)

            # Class Name Similarity
            # 注意不同Path的格式可能不同，0-2存在空
            filename = file.split(".")
            rawClassNames = filename[-2]
            name_len = len(rawClassNames)
            if rawClassNames in summary:
                print(summary)
                classNameSimilarity = name_len
                print("class:", rawClassNames, classNameSimilarity)
            else:
                classNameSimilarity = 0

            # Bug Fixing Recency
            mrReport = getMostRecentReport(file, date, allBugReports)
            # print("mrReport", mrReport.get("bug_id"), mrReport.get("report_time"), mrReport.get("files"))
            bFRecency = bugFixingRecency(report, mrReport)
            # print(bFRecency)

            # Bug Fixing Frequency
            bFFrequency = bugFixingFrequency(file, date, allBugReports)

            br.append(bug_id)
            br.append(file)
            br.append(collaborativeFilterScore)
            br.append(classNameSimilarity)
            br.append(bFRecency)
            br.append(bFFrequency)
            all_br.append(br)



'''
        for file in report["files"]:
            # br = []
            # src = javaFiles[file]

            # rVSM Text Similarity
            # rVSMTextSimilarity = CS.cosine_sim(report["rawCorpus"], src)

            # Collaborative Filter Score
            prevReports = getPreviousReportByFilename(file, date, allBugReports)
            relatedCorpus = []
            for preReport in prevReports:
                # relatedCorpus.append(report["rawCorpus"])
                relatedCorpus.append(preReport["summary"])
            relatedString = ' '.join(relatedCorpus)
            collaborativeFilterScore = CS.cosine_sim(rawCorpus, relatedString)
            # print(collaborativeFilterScore)

            # Class Name Similarity
            filename = file.split(".")
            rawClassNames = filename[-2]
            name_len = len(rawClassNames)
            if rawClassNames in summary:
                print(summary)
                classNameSimilarity = name_len
                print("class:", rawClassNames, classNameSimilarity)
            else:
                classNameSimilarity = 0

            # Bug Fixing Recency
            mrReport = getMostRecentReport(file, date, allBugReports)
            # print("mrReport", mrReport.get("bug_id"), mrReport.get("report_time"), mrReport.get("files"))
            bFRecency = bugFixingRecency(report, mrReport)
            # print(bFRecency)

            # Bug Fixing Frequency
            bFFrequency = bugFixingFrequency(file, date, allBugReports)
            # print(bFFrequency)

            # br.append(bug_id)
            # br.append(file)
            # br.append(collaborativeFilterScore)
            # br.append(classNameSimilarity)
            # br.append(bFRecency)
            # br.append(bFFrequency)
            # all_br.append(br)

            for wr in getTop20Files(bug_id, report["files"], summary):
                br = []
                file = wr[0]
                classNameSimilarity = wr[1]
                # print(wr[0], wr[1])
                br.append(bug_id)
                br.append(file)
                br.append(collaborativeFilterScore)
                br.append(classNameSimilarity)
                br.append(bFRecency)
                br.append(bFFrequency)
                all_br.append(br)
    print(all_br)
    features = pd.DataFrame(all_br, columns=["report_id", "file", "collab_filter", "classname_similarity", "bug_recency", "bug_frequency"])
    features.to_csv('data/zookeeper_dat/features_all.csv', index=False)
'''



if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        getbugFeatures_four(dataset)

# for wr in getTop50WrongFiles(report["files"], report["rawCorpus"]):
#     writer.writerow([report["id"], wr[0], wr[1], collaborativeFilterScore, wr[2], bugFixingRecency, bugFixingFrequency, 0])


# all_br = []
# allBugReports = XD.CSVToDictionary()
# for report in allBugReports:
#     bug_id = report["bug_id"]
#     date = convertToDateTime(report.get("report_time"))
#     rawCorpus = report["rawCorpus"]
#     files = report["files"]
#     summary = report["summary"]
#
#     print("===== " + report["bug_id"] + " =====")
#
#     for file in report["files"]:
#         for wr in getTop20Files(bug_id, report["files"], summary):
#             print(wr[0], wr[1])