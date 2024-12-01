import pandas as pd
from configx.configx import ConfigX
from features.utils_features import readRecList, updateFeatures


def getCfReadability(rec_lists):
    df_cfReadability = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/CodeReadability/{i}.csv")

    df_cfReadability = df_cfReadability.rename(columns={'rd.ARI':'cf.ARI', 'rd.FleschReadingEase':'cf.FleschReadingEase',
                                     'rd.FleschKincaidGradeLevel':'cf.FleschKincaidGradeLevel','rd.GunningFogIndex':'cf.GunningFogIndex',
                                    'rd.SMOGIndex':'cf.SMOGIndex', 'rd.ColemanLiauIndex':'cf.ColemanLiauIndex',
                                     'rd.LIX':'cf.LIX','rd.RIX':'cf.RIX'})

    df_bfFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_bfFeature_result = updateFeatures(df_bfFeatures,df_cfReadability)
    df_bfFeature_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)



if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====Code Readability: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            getCfReadability(readRecList(dataset,i))

