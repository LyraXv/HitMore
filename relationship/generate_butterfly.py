import sys
sys.path.append("D:\\SciTools\\bin\\pc-win64\\Python")   # Understand 模块位置
import os
os.add_dll_directory("D:\\Scitools\\bin\\pc-win64\\")   # Understand 安装的路径
import understand
import pandas as pd
# import locale
# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')



def generate_butterfly_dot(csv_file, und_folder, output_folder):
    # 读取CSV文件
    df = pd.read_csv(csv_file,encoding='utf-8')
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        # file_index = row['index']
        # if file_index not in ['4_1701','4_1742','4_1747','4_1762','4_1763','4_1764','4_1765','4_1766','4_1768','4_1771','4_1772','4_1773','4_1803','4_1821','4_1822','4_1823','4_1824','4_1825','4_1827','4_1840','4_1842','4_1843','4_1845','4_1846']:
        #     continue
        commit = row['commit']
        buggy_file = row['filePath']
        buggy_file = buggy_file.replace('/','\\')
        print(buggy_file)
        und_file_path = os.path.join(und_folder, f"{commit}.und")
        print(und_file_path)
        # 检查 .und 文件是否存在
        if not os.path.exists(und_file_path):
            print(f"UND file not found: {und_file_path}")
            continue

        # 打开 .und 文件
        try:
            db = understand.open(und_file_path)
            print("db",db)
        except Exception as e:
            print(f"Failed to open UND file: {und_file_path}, Error: {e}")
            continue
        
        # 获取目标文件实体
        all_entities = db.ents("File")
        target_file_entity = None
        for ent in all_entities:
            # print(f"und name,{ent.longname()}")
            if ent.longname().endswith(buggy_file):
                print(ent.longname())
                target_file_entity = ent
        if target_file_entity is None:
            print(f"Entity not found for file: {buggy_file} in commit: {commit}")
            continue
        
        # 生成 Butterfly 图并保存为 .dot 文件
        output_dot_file = os.path.join(output_folder, f"{row['index']}.dot")
        print(type(target_file_entity))
        print("target_file_entity",target_file_entity)
        try:
            target_file_entity.draw("Butterfly", output_dot_file)
            print(f"Butterfly dot file saved: {output_dot_file}")
        except Exception as e:
            print(f"Failed to generate Butterfly graph for {buggy_file}, Error: {e}")
        
        # 关闭数据库
        db.close()

if __name__ == "__main__":
    csv_file = r'D:\HitMore\HitMore-main\data\rec_lists\zookeeper_truly_buggy_file_result.csv'
    und_folder = r"D:\SciTools\bin\pc-win64\zookeeperDb"
    output_folder = f'D:\\HitMore\\Butterfly\\zookeeper'

    os.makedirs(output_folder, exist_ok=True)

    # 生成 Butterfly 图
    try:
        generate_butterfly_dot(csv_file, und_folder, output_folder)
    except Exception as e:
        print(f"An error occurred: {e}")