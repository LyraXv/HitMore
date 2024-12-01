import os
import csv
from collections import defaultdict
from datetime import datetime
import re
import pandas as pd


def extract_contributor_names(contributors_row):
    # 将从第二列开始的所有列数据拼接成一个字符串
    contributors_str = ','.join(contributors_row[1:])  
    # 使用正则表达式提取贡献者姓名
    names = re.findall(r'([^<,]+)<[^>]+>', contributors_str)
    return [name.strip() for name in names]

def find_cooccurrence_files(bug_file_path, blame_info_path, contributors_path):
    # 读取包含 Bug ID、Truly Buggy File 和 Commit 的 CSV 文件
    with open(bug_file_path, mode='r', encoding='utf-8') as bug_file:
        reader = csv.DictReader(bug_file)
        bug_data = [{key.strip(): value for key, value in row.items()} for row in reader]
    
    # 存储每个 Truly Buggy File 的相关文件列表
    related_files_data = {}

    # 日期格式
    date_format = '%Y-%m-%d %H:%M:%S'

    head_flag = True
    # 遍历每个 Commit
    for bug_entry in bug_data:
        index = bug_entry['index'].strip()
        print(index)
        bugId = bug_entry['bugID'].strip()
        commit_hash = bug_entry['commit'].strip()
        print(commit_hash)
        buggy_file = bug_entry['filePath'].strip()  # buggy_file需要换成左斜杠/
        buggy_file = buggy_file.replace('\\','/')

        # 构建Developers/CSV 文件路径
        dev_csv_path = os.path.join(blame_info_path, f"{commit_hash}.csv")
        # 构建Contributors/CSV 文件路径
        contrib_csv_path = os.path.join(contributors_path, f"{commit_hash}.csv")

        # 同时提交、修改的文件集合
        same_date_files = set()
        # 同一个贡献者的文件集合
        same_contributor_files = set()

        # 处理开发者数据
        if os.path.exists(dev_csv_path):
            with open(dev_csv_path, mode='r', encoding='utf-8') as dev_file:
                dev_data = list(csv.DictReader(dev_file))

            # 查找 Truly Buggy File 在 dev.csv 中的匹配行
            matching_entries = [entry for entry in dev_data if buggy_file in entry['file']]
            print("buggy_file: ", buggy_file)
            print("matching_entry: ", matching_entries)
            
            # 对每个匹配行，根据日期查找相同日期的所有文件名
            for entry in matching_entries:
                date_value = datetime.strptime(entry['date'].strip(), date_format)
                same_date_files.update({e['file'].strip() for e in dev_data if datetime.strptime(e['date'].strip(), date_format) == date_value})

            same_date_files = {file_path.replace('./', '', 1) for file_path in same_date_files}
        else:
            print(f"Warning: Developer file {dev_csv_path} does not exist.")

        # 处理贡献者数据
        if os.path.exists(contrib_csv_path):
            with open(contrib_csv_path, mode='r', encoding='utf-8') as contrib_file:
                contrib_reader = csv.reader(contrib_file)

                # 跳过标题行
                next(contrib_reader)
                
                contrib_data = [row for row in contrib_reader]

            # 查找 Truly Buggy File 在 contrib.csv 中的匹配行
            matching_entry = next((row for row in contrib_data if row[0].strip() == buggy_file), None)
            print("contrib_matching_entry:",matching_entry)

            # 如果找到匹配项，解析贡献者名称并查找同一贡献者的所有文件
            if matching_entry:
                contributors = extract_contributor_names(matching_entry)
                print("contributors",contributors)
                for contributor in contributors:
                    same_contributor_files.update({row[0].strip() for row in contrib_data if contributor in extract_contributor_names(row)})
        else:
            print(f"Warning: Contributor file {contrib_csv_path} does not exist.")

        print("两个列表：")
        print("date",len(same_date_files))
        print("contributor",len(same_contributor_files))
        # 两个列表取交集
        cooccurrence_files = same_date_files & same_contributor_files
        print("交集",len(cooccurrence_files))

        # save res as csv:
        res=[{
            'index': index,
            'bugId': bugId,
            'filePath': buggy_file,
            'commit': commit_hash,
            'co_file': cooccurrence_files
        }]

        if head_flag:
            output_df = pd.DataFrame(res)
            output_df.to_csv(output_file, index=False, mode='a')
            head_flag = False
        else:
            output_df = pd.DataFrame(res)
            output_df.to_csv(output_file, index=False, mode='a', header=0)

        # 将合并后的文件列表存储到结果字典中
        # related_files_data[buggy_file] = sorted(cooccurrence_files)
    # return related_files_data
    # df_res = pd.DataFrame(res_list)
    # df_res.to_csv(f'D:\\HitMore\\Cooccurence\\test_co_list.csv',index=False)


dataset = 'zookeeper'
bug_file_path = f'D:\\HitMore\\HitMore-main\\data\\rec_lists\\orderedByTime\\{dataset}_truly_buggy_file_result_byTime.csv'
blame_info_path = f'D:\\HitMore\\Developers\\{dataset}'  # Developers 文件夹路径
contributors_path = f'D:\\HitMore\\Contributors\\{dataset}'  # Contributors 文件夹路径
output_file = f'D:\\HitMore\\Cooccurence_time\\{dataset}_co_list.csv'
find_cooccurrence_files(bug_file_path, blame_info_path, contributors_path)

