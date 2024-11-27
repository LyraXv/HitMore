import pandas as pd

from configx.configx import ConfigX
from features.utils_features import readRecList


def mergeCfReadability(rec_lists):
    df_cfReadability = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/CodeReadability/{i}.csv")

    df_cfReadability.rename(columns={'rd.ARI':'cf.ARI', 'rd.FleschReadingEase':'cf.FleschReadingEase',
                                     'rd.FleschKincaidGradeLevel':'cf.FleschKincaidGradeLevel','rd.GunningFogIndex':'cf.GunningFogIndex',
                                    'rd.SMOGIndex':'cf.SMOGIndex', 'rd.ColemanLiauIndex':'cf.ColemanLiauIndex',
                                     'rd.LIX':'cf.LIX','rd.RIX':'cf.RIX'})



    df_res.to_csv(f"../data/splited_and_boosted_data/{dataset}/bugReportsFeatures/{i}.csv", index=False)


if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset in ['zookeeper']:
            continue
        print(f"=====Code and Comments Consisitency: {dataset}=====")
        for i in range(6):
            print(f"Current fold: {i}")
            mergeBRfeatures(readRecList(dataset,i))
