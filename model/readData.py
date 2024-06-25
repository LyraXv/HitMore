# read bugInfo,bugReport,Top20 fileLists for 3 approach
import utils.readBugInfo as info
from configx.configx import ConfigX


if __name__ == '__main__':
    # global config
    configx = ConfigX()

    # read data(bug,bugReport,RecommendedList)
    for dataset,file in configx.filepath_dict.items():
        print("=====Dataset:",dataset,"=====")
        if dataset not in ['hibernate']:
            continue

        # # bug
        bug_df, bug_ids = info.readBugId(file['bugInfo'])
        # # sava bug_info
        bug_df.to_csv('../data/get_info/'+dataset+'/bug_info.csv')
        print("BugInfo:"+dataset+" is saved!")
        #

        # save bug_report_info
        # bugReport_df = info.readBugReport(dataset,file['bugReport'],bug_ids)
        # print(file['bugReport'])
        # bugReport_df.to_csv('../data/get_info/'+dataset+'/bug_report.csv')
        # print("BugReport:"+dataset+" is saved!")

        # load recommendedList

        # save top20 csv
        '''
        for dataset, file in configx.filepath_dict.items():
            if dataset not in ('zookeeper'):
                continue
            for index in range(len(configx.approach)):
                approach = configx.approach[index]
                print(f">>>>>{configx.approach[index]} of {dataset} Top20 begin saving<<<<<")
                rec_filepath = '../data/initial_recommendation_data/'+ approach + '_new/' + dataset
                gt_filepath = rec_filepath + '_bugPredict.txt'
                list_df = info.readRecommendList(approach,dataset,rec_filepath,gt_filepath)
                list_df.to_csv('../data/get_info/' + dataset + '/' + approach + 'List_top20.csv')
                print(f">>>>>{configx.approach[index]} of {dataset} Top20 is Saved<<<<<")
'''