import os
import csv
import itertools
from collections import defaultdict
import similarity as CS


def read_csv(file_path):
    developer_files = defaultdict(list)
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            developer = row['developer_name']
            file = row['file']
            file = file.replace('\\', '/')  # 将反斜杠替换为正斜杠
            developer_files[developer].append(file)
    return developer_files


def read_code_corpus(file_path):
    file_content = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file = row['File Path']
            content = row['All Content']
            file_content[file] = content
    return file_content


def calculate_path_distance(file1, file2):
    # 分割路径并计算不同部分的数量
    parts1 = file1.split('/')
    parts2 = file2.split('/')

    compath = os.path.commonpath([file1, file2]).replace('\\', '/')
#     print(compath)
    common_length = len(compath.split('/'))
#     print(common_length)
    distance = (len(parts1) - common_length) + (len(parts2) - common_length) - 1
#     print(distance)
    return distance


# 求和平均（每两个文件之间的最短距离）
def calculate_shortest_path_average(file_list):
    if len(file_list) < 2:
        return 0  # 如果开发者只修改了一个文件，返回0

    path_lengths = []
    for file1, file2 in itertools.combinations(file_list, 2):
        path_length = calculate_path_distance(file1, file2)
#         print(path_length)
        path_lengths.append(path_length)

    if path_lengths:
        return sum(path_lengths) / len(path_lengths)
    else:
        return 0


# 结构散射：开发人员d在时间段α中修改的类的数量 × 求和平均—每两个类之间的包的距离（最短路径算法）
def structural_scattering(file_name, commit):
    try:
        # 根据commit查找Developers文件夹下对应csv文件
        developer_files = find_developer_files(commit)
        # 查找与给定文件名对应的developer
        developer, files = get_developer_files(file_name, developer_files)

        if developer:
            # 计算该 developer 修改的所有文件之间的最短路径的平均值
            all_files = developer_files[developer]
            file_count = len(all_files)

            if file_count > 1:
                avg_distance = calculate_shortest_path_average(all_files)
            else:
                avg_distance = 0

            StrScat = file_count * avg_distance
            return developer, file_count, avg_distance, StrScat
        else:
            return None, 0, 0, 0
    
    except FileNotFoundError as e:
        print(e)
        return None, 0, 0, 0


# 求和平均（每两个文件之间的文本相似性）
def calculate_similarity_average(file_list, commit):
    code_corpus_file = os.path.join('CodeCorpus', f'{commit}.csv')
    file_contents = read_code_corpus(code_corpus_file)
    file_sim = []
    for file1, file2 in itertools.combinations(file_list, 2):
        file1_content = file_contents[file1]
        file2_content = file_contents[file2]
        
        f1 = CS.preprocess(file1_content)
        f2 = CS.preprocess(file2_content)
        Str_f1 = " ".join(f1)
        Str_f2 = " ".join(f2)
        
        sim = CS.cosine_sim(Str_f1, Str_f2)
        file_sim.append(sim)

    if path_lengths:
        return sum(file_sim) / len(file_sim)
    else:
        return 0
    

# 语义散射：开发人员d在时间段α中修改的类的数量 × （ 1 / 平均值—每两个类之间的文本相似度（VSM））
def semantic_scattering(file_name, commit):
    try:
        developer_files = find_developer_files(commit)
        developer, files = get_developer_files(file_name, developer_files)

        if developer:
            # 计算该 developer 修改的所有文件之间的相似性的平均值
            all_files = developer_files[developer]
            file_count = len(all_files)

            if file_count > 1:
                avg_sim = calculate_similarity_average(all_files, commit)
            else:
                avg_sim = 0

            SemScat = file_count * (1 / avg_sim)
            return developer, file_count, avg_sim, SemScat
        else:
            return None, 0, 0, 0
    
    except FileNotFoundError as e:
        print(e)
        return None, 0, 0, 0


# 根据 commit 查找对应的 CSV 文件
def find_developer_files(commit, csv_folder='Developers'):
    csv_file = os.path.join(csv_folder, f'{commit}.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} not found.")
    
    developer_files = read_csv(csv_file)

    # 返回字典 {developer: 修改的文件列表}
    return developer_files


# 根据file name查找对应的developer
def get_developer_files(file_name, developer_files):
    for developer, files in developer_files.items():
        if file_name in files:
            return developer, files
    return None, []


def main():
    input_file = 'data.csv'   # 包含三列数据（bug id，file name，commit）

    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bug_id = row['bug_id']
            file_name = row['file_name']
            commit = row['commit']
            
            developer, file_count, avg_distance, StrScat = structural_scattering(file_name, commit)
            avg_sim, SemScat = semantic_scattering(file_name, commit)

            if developer:
                print(f"Bug ID: {bug_id}, File: {file_name}, Developer: {developer}, Modifications: {file_count}, Structural Scattering: {StrScat}, Semantic Scattering: {SemScat}")


if __name__ == "__main__":
    main()
