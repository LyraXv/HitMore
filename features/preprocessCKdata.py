import pandas as pd

from configx.configx import ConfigX
from features.utils_features import updateFeatures


def preprocessCK():
    df_bfFeatures = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    # Specifies the columns to be populated
    # if dataset == 'zookeeper':
    #     columns_to_fill = list(df_bfFeatures.iloc[:,19:69].columns)
    # else:
    #     columns_to_fill = list(df_bfFeatures.iloc[:,17:67].columns)
    columns_to_fill = ['cbo', 'cboModified', 'fanin', 'fanout', 'wmc', 'dit', 'noc', 'rfc', 'lcom', 'lcom*', 'tcc', 'lcc', 'totalMethodsQty', 'staticMethodsQty', 'publicMethodsQty', 'privateMethodsQty', 'protectedMethodsQty', 'defaultMethodsQty', 'visibleMethodsQty', 'abstractMethodsQty', 'finalMethodsQty', 'synchronizedMethodsQty', 'totalFieldsQty', 'staticFieldsQty', 'publicFieldsQty', 'privateFieldsQty', 'protectedFieldsQty', 'defaultFieldsQty', 'finalFieldsQty', 'synchronizedFieldsQty', 'nosi', 'loc', 'returnQty', 'loopQty', 'comparisonsQty', 'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty', 'mathOperationsQty', 'variablesQty', 'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty', 'lambdasQty', 'uniqueWordsQty', 'modifiers', 'logStatementsQty', 'bugFixDependencies']

    # 对每一个指定列，按 bugID 分组，并用组内的中位数填充缺失值
    for col in columns_to_fill:
        df_bfFeatures[col] = df_bfFeatures.groupby('bugId')[col].transform(lambda x: x.fillna(x.median()))

    df_temp = pd.read_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv")
    df_bfFeature_result = updateFeatures(df_temp,df_bfFeatures)
    df_bfFeature_result.to_csv(f"../data/splited_and_boosted_data/{dataset}/buggyFileFeatures/{i}.csv",index=False)

def preprocessScattering(): #hibernate
    df_bfFeatures = pd.read_csv(f"../data/splited_and_boosted_data/hibernate/buggyFileFeatures/{i}.csv")
    columns_to_fill = ['developersStructuralScattering','developersSemanticScattering']

    for col in columns_to_fill:
        df_bfFeatures[col] = df_bfFeatures.groupby('bugId')[col].transform(lambda x: x.fillna(x.mean()))
    df_temp = pd.read_csv(f"../data/splited_and_boosted_data/hibernate/buggyFileFeatures/{i}.csv")
    df_bfFeature_result = updateFeatures(df_temp,df_bfFeatures)
    df_bfFeature_result.to_csv(f"../data/splited_and_boosted_data/hibernate/buggyFileFeatures/{i}.csv",index=False)



if __name__ == '__main__':
    configx = ConfigX()
    for dataset,file in configx.filepath_dict.items():
        if dataset not in ['zookeeper']:
            continue
        print(f"=====preprocess CK: {dataset}=====")
        for i in [0, 1, 2, 3, 4, 'otherTrulyBuggyFiles']:
            if i != 'otherTrulyBuggyFiles':
                continue
            print(f"Current fold: {i}")
            preprocessCK()
            # preprocessScattering()
