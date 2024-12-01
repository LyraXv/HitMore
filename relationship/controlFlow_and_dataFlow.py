import sys
sys.path.append("D:\\SciTools\\bin\\pc-win64\\Python")   # Understand 模块位置
import os
os.add_dll_directory("D:\\Scitools\\bin\\pc-win64\\")   # Understand 安装的路径
import understand
import pandas as pd
import re


def extract_data_flow_dependencies(csv_file, und_folder, output_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 用于保存结果
    results = []
    df_head_flag = True
    skip_key = True
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        file_index = row['index']

        print(file_index)
        # if file_index != '3_5143' and skip_key:
        #     continue
        # else:
        #     skip_key = False
        commit = row['commit']

        # 选择指定commit
        if commit not in commits_set:
            continue

        buggy_file = row['filePath']
        und_file_path = os.path.join(und_folder, f"{commit}.und")

        
        # 检查 .und 文件是否存在
        if not os.path.exists(und_file_path):
            print(f"UND file not found: {und_file_path}")
            continue
        
        # 打开 .und 文件
        try:
            db = understand.open(und_file_path)
        except Exception as e:
            print(f"Failed to open UND file: {und_file_path}, Error: {e}")
            continue
        

        # 查找数据库中的目标文件实体
        all_entities = db.ents("File")
        target_file_entity = None
        target_funcs = set()

        # 获取目标文件实体
        for ent in all_entities:
            if ent.longname().endswith(buggy_file):
                target_file_entity = ent
                
                # 获取目标文件中的所有函数或方法
                for func in db.ents("function, method"):
                    parent_entity = func.parent()
                    # 循环查找直到找到文件类型的实体
                    while parent_entity is not None and not parent_entity.kind().check("file"):
                        parent_entity = parent_entity.parent()
                    # 如果找到了文件类型的实体，输出文件名
                    if parent_entity is not None:
                        file_name = parent_entity.longname()
                        if file_name == target_file_entity.longname():
    #                         print(file_name)
    #                         print("func.longname: ", func.longname())
                            target_funcs.add(func)
        
        if target_file_entity is None:
            print(f"Entity not found for file: {buggy_file} in commit: {commit}")
            continue
        
        
        # 提取数据流依赖相关的文件列表
        data_flow_deps = set()

        for func in target_funcs:
            # 获取与数据流相关的引用（定义、使用、读取、赋值等）
            for ref in func.refs():
                referenced_entity = ref.ent()
                referencing_file = ref.file()
                # 检查被引用的实体是否位于目标文件之外的文件中
                if referencing_file != target_file_entity:    
                    data_flow_deps.add(referencing_file.longname())
    #         print(func.longname())

        results = [] # temp
        for dep_file in data_flow_deps:
            results.append({
                'index': row['index'],
                'bugID': row['bugID'],
                'filePath': buggy_file,
                'commit': commit,
                'df_filePath': dep_file
            })



        if df_head_flag:
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_file, index=False, mode='a')
            df_head_flag = False
        else:
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_file, index=False, mode='a', header=0)

        # print(results)
        db.close()
        print(f"数据流依赖的文件: {data_flow_deps}")
        
    
    # # 将结果保存为CSV文件
    # output_df = pd.DataFrame(results)
    # output_df.to_csv(output_file, index=False,mode='a')
    # print(f"Results saved to {output_file}")




def parse_dot_file(dot_file):
    nodes = {}
    edges = []

    with open(dot_file, 'r') as file:
        for line in file:
            # 匹配节点
            node_match = re.match(r'^\s*__(N\d+)\s*\[label="(.+?)"\s*.*\];', line)
            if node_match:
                node_id = node_match.group(1)
                label = re.sub(r'&#[0-9]+;', '', node_match.group(2)).strip()  # 清理 HTML 实体
                label = re.sub(r'\s*&#965\d;\s*', '', label)  # 移除特殊符号
                nodes[node_id] = label

            # 匹配边
            edge_match = re.match(r'^\s*__(N\d+)\s*->\s*__(N\d+)\s*\[label="(\d+)(?:\s*/\s*(\d+))?".*\];', line)
            if edge_match:
                src_node = edge_match.group(1)
                dst_node = edge_match.group(2)
                count = int(edge_match.group(3)) + (int(edge_match.group(4)) if edge_match.group(4) else 0)

                edges.append((src_node, dst_node, count))

    return nodes, edges


def get_file_path(entity_name, udb):
    # 查找与实体名称对应的文件路径
    for ent in udb.ents("file ~unknown ~unresolved"):
        if entity_name in ent.longname():
            return ent.longname()
    return entity_name


def get_n1_edges(edges, nodes, udb):
    n1_edges = []
    for src, dst, count in edges:
        # N1指向（目标文件调用的文件）
        if src == "N1":
            dst_file = get_file_path(nodes[dst], udb)
            n1_edges.append((dst_file, count))
        # 指向N1（调用目标文件的文件）
        elif dst == "N1":
            src_file = get_file_path(nodes[src], udb)
            n1_edges.append((src_file, count))
    return n1_edges


def extract_control_flow_dependencies(csv_file, und_folder, butterfly_folder, output_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    head_flag = True # 注意下个数据集修改
    error_index_flag = True

    for _, row in df.iterrows():
        index = row['index']
        # if index!='2043'and error_index_flag:
        #     continue
        # else:
        #     error_index_flag = False
        print(index)
        commit = row['commit']
        # 选择指定commit
        if commit not in commits_set:
            continue

        und_file_path = os.path.join(und_folder, f"{commit}.und")
        dot_file = os.path.join(butterfly_folder, f"{index}.dot")

        # 检查 .und 文件是否存在
        if not os.path.exists(und_file_path):
            print(f"UND file not found: {und_file_path}")
            continue
        
        # 打开 .und 文件
        try:
            udb = understand.open(und_file_path)
        except Exception as e:
            print(f"Failed to open UND file: {und_file_path}, Error: {e}")
            continue

        # 检查 .dot 文件是否存在
        if not os.path.exists(dot_file):
            print(f"Dot file not found: {dot_file}")
            continue
        
        # 解析 .dot 文件
        nodes, edges = parse_dot_file(dot_file)
        n1_edges = get_n1_edges(edges, nodes, udb)

        # print("N1 Edges (Dependencies):")
        # for src, dst, count in n1_edges:
        #     print(f"{src} -> {dst} : {count} time(s)")

        cf_info = {}
        print("cf_files",len(n1_edges))
        for edge, count in n1_edges:
            cf_info[edge] = count

        results=[{
            'index': index,
            'bugId': row['bugID'],
            'filePath': row['filePath'],
            'commit': row['commit'],
            'cf_file': cf_info
        }]

    # 将结果保存为CSV文件
        if head_flag:
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_file, index=False,mode='a')
            head_flag = False
        else:
            output_df = pd.DataFrame(results)
            output_df.to_csv(output_file, index=False, mode='a',header=0)
        udb.close()
    print(f"Results saved to {output_file}")





dataset = 'zookeeper'
file_range = 'time'

# hibernate
path = f"D:\\SciTools\\bin\\pc-win64\\ordered_bugCmit_{dataset}_{file_range}"
with open(path, 'r') as f:
    commits = [line.strip().split(',') for line in f.readlines()]
    df_commits = pd.DataFrame(commits, columns=['bugId', 'cmit', 'date'])
commits_set = df_commits['cmit'].values.tolist()

csv_file = f'D:\\HitMore\\HitMore-main\\data\\rec_lists\\orderedByTime\\{dataset}_truly_buggy_file_result_byTime.csv'
und_folder = f"D:\\SciTools\\bin\\pc-win64\\{dataset}Db_{file_range}"
data_flow_file = f'D:\\HitMore\\DataFlow_time\\{dataset}_df_list.csv'
# #
# # # 数据流依赖文件列表提取
extract_data_flow_dependencies(csv_file, und_folder, data_flow_file)

# csv_file = f'D:\\HitMore\\HitMore-main\\data\\rec_lists\\{dataset}_truly_buggy_file_result_byTime.csv'
butterfly_folder = f'D:\\HitMore\\Butterfly_time\\{dataset}'
control_flow_file = f'D:\\HitMore\\ControlFlow_time\\{dataset}_cf_list.csv'

# 控制流依赖文件列表提取
extract_control_flow_dependencies(csv_file, und_folder, butterfly_folder, control_flow_file)