import sys
sys.path.append("D:\\SciTools\\bin\\pc-win64\\Python")   # Understand 模块位置
import os
os.add_dll_directory("D:\\Scitools\\bin\\pc-win64\\")   # Understand 安装的路径
import understand
import pandas as pd


def generate_butterfly_dot(csv_file, und_folder, output_folder):

    path = f"D:\\SciTools\\bin\\pc-win64\\ordered_bugCmit_{dataset}"
    with open(path,'r')as f:
        commits = [line.strip().split(',') for line in f.readlines()]
        df_commits = pd.DataFrame(commits,columns=['bugId','cmit','date'])
    commits_set = df_commits['cmit'].values.tolist()

    # read csv file
    df = pd.read_csv(csv_file,encoding='utf-8')

    #
    skip_key = True
    # Iterate through each row of data
    for index, row in df.iterrows():
        file_index = row['index']
        # if str(file_index)!= '5806' and skip_key:
        #     continue
        # else:
        #     skip_key =False
        commit = row['commit']
        if commit not in commits_set:
            continue
        buggy_file = row['filePath']
        buggy_file = buggy_file.replace('/','\\')
        print(buggy_file)
        und_file_path = os.path.join(und_folder, f"{commit}.und")
        print(und_file_path)
        # check for the existence of .und files
        if not os.path.exists(und_file_path):
            print(f"UND file not found: {und_file_path}")
            continue

        # open .und file
        try:
            db = understand.open(und_file_path)
            print("db",db)
        except Exception as e:
            print(f"Failed to open UND file: {und_file_path}, Error: {e}")
            continue
        
        # Get the target file entity
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
        
        # generate a Butterfly diagram and save it as a .dot file
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
    dataset = 'zookeeper'
    # file_range = 'time'
    csv_file = f'D:\\HitMore\\HitMore-main\\data\\rec_lists\\orderedByTime\\{dataset}_truly_buggy_file_result.csv'
    und_folder = f"D:\\SciTools\\bin\\pc-win64\\{dataset}Db"
    output_folder = f'D:\\HitMore\\Butterfly_time\\{dataset}'

    os.makedirs(output_folder, exist_ok=True)

    # generate a Butterfly diagram
    try:
        generate_butterfly_dot(csv_file, und_folder, output_folder)
    except Exception as e:
        print(f"An error occurred: {e}")