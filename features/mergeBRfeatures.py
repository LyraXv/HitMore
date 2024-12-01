'''
Before running this file, make sure the
"bugReportFeatures.csv","ReportersFeatures.csv" and "ReportReadability" are present.
'''
import pandas as pd

from configx.configx import ConfigX
from features.utils_features import readRecList


def mergeBRfeatures(rec_lists):
    df_res = rec_lists.rename(columns={'bugId':'BugId'})
    if 'index' not in df_res.columns:
        df_res = df_res.rename(columns ={'Unnamed: 0':'index'})

    # Read three files
    df_br_part = pd.read_csv(f"../data/get_info/{dataset}/bugReportFeatures.csv")
    df_reporters = pd.read_csv(f"../data/get_info/{dataset}/ReportersFeatures.csv")
    df_rReadability = pd.read_csv(f"../../ReportReadability/{dataset}.csv")

    # for bugId, group in grouped:
    df_res = df_res.merge(df_br_part,on='BugId',how='left')
    df_res = df_res.merge(df_reporters,on='BugId',how='left')
    df_res = df_res.merge(df_rReadability,on='BugId',how='left')

    df_res = df_res.rename(columns={'BugId':'bugId'})

    # Remove redundant columns
    df_drop_columns = ['Reporter', 'Assignee', 'create_time', 'resolve_time', 'Summary', 'Description', 'Pre_bug_text','Unnamed: 0']
    df_res = df_res.drop(columns = df_drop_columns)


    df_res.to_csv(f"../data/splited_and_boosted_data/{dataset}/bugReportsFeatures/{i}.csv", index=False)


if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====mergeBRfeatures: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            mergeBRfeatures(readRecList(dataset,i))
