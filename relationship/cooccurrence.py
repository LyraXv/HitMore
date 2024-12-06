import os
import csv
from collections import defaultdict
from datetime import datetime
import re
import pandas as pd


def extract_contributor_names(contributors_row):
    # Splices all columns starting from the second column into a string
    contributors_str = ','.join(contributors_row[1:])  
    # Extracting Contributor Names Using Regular Expressions
    names = re.findall(r'([^<,]+)<[^>]+>', contributors_str)
    return [name.strip() for name in names]

def find_cooccurrence_files(bug_file_path, blame_info_path, contributors_path):
    # Read CSV file containing Bug ID, Truly Buggy File and Commit
    with open(bug_file_path, mode='r', encoding='utf-8') as bug_file:
        reader = csv.DictReader(bug_file)
        bug_data = [{key.strip(): value for key, value in row.items()} for row in reader]

    related_files_data = {}

    # Date format
    date_format = '%Y-%m-%d %H:%M:%S'

    head_flag = True
    # Iterate through each Commit
    for bug_entry in bug_data:
        index = bug_entry['index'].strip()
        print(index)
        bugId = bug_entry['bugID'].strip()
        commit_hash = bug_entry['commit'].strip()
        print(commit_hash)
        buggy_file = bug_entry['filePath'].strip()
        buggy_file = buggy_file.replace('\\','/')

        # Build Developers/CSV File Paths
        dev_csv_path = os.path.join(blame_info_path, f"{commit_hash}.csv")
        # Build Contributors/CSV File Paths
        contrib_csv_path = os.path.join(contributors_path, f"{commit_hash}.csv")

        # Collection of files submitted and revised at the same time
        same_date_files = set()
        # Collection of files from the same contributor
        same_contributor_files = set()

        # Processing of developer data
        if os.path.exists(dev_csv_path):
            with open(dev_csv_path, mode='r', encoding='utf-8') as dev_file:
                dev_data = list(csv.DictReader(dev_file))

            # Find Truly Buggy File matching lines in dev.csv
            matching_entries = [entry for entry in dev_data if buggy_file in entry['file']]
            print("buggy_file: ", buggy_file)
            print("matching_entry: ", matching_entries)
            
            # For each matching line, find all filenames with the same date by date
            for entry in matching_entries:
                date_value = datetime.strptime(entry['date'].strip(), date_format)
                same_date_files.update({e['file'].strip() for e in dev_data if datetime.strptime(e['date'].strip(), date_format) == date_value})

            same_date_files = {file_path.replace('./', '', 1) for file_path in same_date_files}
        else:
            print(f"Warning: Developer file {dev_csv_path} does not exist.")

        # Processing of contributor data
        if os.path.exists(contrib_csv_path):
            with open(contrib_csv_path, mode='r', encoding='utf-8') as contrib_file:
                contrib_reader = csv.reader(contrib_file)

                # Skip title line
                next(contrib_reader)
                
                contrib_data = [row for row in contrib_reader]

            # Find Truly Buggy File matching lines in contrib.csv
            matching_entry = next((row for row in contrib_data if row[0].strip() == buggy_file), None)
            print("contrib_matching_entry:",matching_entry)

            # If a match is found, parses the contributor name and looks for all files from the same contributor
            if matching_entry:
                contributors = extract_contributor_names(matching_entry)
                print("contributors",contributors)
                for contributor in contributors:
                    same_contributor_files.update({row[0].strip() for row in contrib_data if contributor in extract_contributor_names(row)})
        else:
            print(f"Warning: Contributor file {contrib_csv_path} does not exist.")

        cooccurrence_files = same_date_files & same_contributor_files

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


dataset = 'zookeeper'
bug_file_path = f'D:\\HitMore\\HitMore-main\\data\\rec_lists\\orderedByTime\\{dataset}_truly_buggy_file_result_byTime.csv'
blame_info_path = f'D:\\HitMore\\Developers\\{dataset}'  # Developers 文件夹路径
contributors_path = f'D:\\HitMore\\Contributors\\{dataset}'  # Contributors 文件夹路径
output_file = f'D:\\HitMore\\Cooccurence_time\\{dataset}_co_list.csv'
find_cooccurrence_files(bug_file_path, blame_info_path, contributors_path)

