#！/usr/bin/env python
import os
import pandas as pd

# 读取ground truth文件到dataframe
def readGroundTruth(filepath):
    res_pred = []
    res = open(filepath,'r')
    for line in res:
        res_pred.append(list(line.strip('\n').split(',')))
    res.close()
    res_df = pd.DataFrame(res_pred, columns=['BugId', 'SourceFile', 'Rank', 'Score'])
    # print(res_df)
    return res_df


# 读取推荐列表
filepath = 'data/bugLocator/openjpa/bugFixedFileRanks'
# files 返回指定的文件夹包含的文件列表
files = os.listdir(filepath)
res_recommend = []
# count 对每个bug报告推荐结果取子集大小

for fi in files:
    count = 0
    # 把目录和文件名合成一个路径
    fi_d = os.path.join(filepath, fi)
    res = open(fi_d,'r')
    # 读取文件中前二十行数据
    for rr in res:
        res_list = []
        if (count == 20):
            break
        # print(fi)
        # res_list.append(fi.strip('.txt'))
        res_list += list(rr.strip('\n').split(','))
        res_list.append(0)
        res_recommend.append(res_list)
        count += 1
    # print(count)
    res.close()

# print(res_recommend)
dataframe = pd.DataFrame(res_recommend, columns=['BugId', 'Rank','SourceFile', 'Score', 'label'])
# dataframe.to_csv('data/res.csv')
# print(dataframe)

gt = readGroundTruth('data/bugLocator/openjpa/openjpa_bugPredict.res')
gt['label'] = 1
# 通过ground truth给res_recommend数据打标签
# 遍历dataframe中每一行数据，若存在与gt中数据相同的，令label=1
df = pd.merge(dataframe, gt, on=['BugId', 'SourceFile', 'Score', 'Rank'], how='left')
df.fillna(0,inplace=True)
df = df.astype({"label_y": int})
df.drop(columns=['label_x'])
print(df)
df.to_csv('data/openjpa_dat/buglocator_openjpa.csv')
