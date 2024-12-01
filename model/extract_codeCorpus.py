import os
import sys
import csv


# 提取代码和注释内容
def extract_code_and_comments(file_path):
    code_lines = []
    comment_lines = []
    all_lines = []
    in_multiline_comment = False

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            all_lines.append(line.strip())
            stripped_line = line.strip()
            if stripped_line.startswith('/*'):
                in_multiline_comment = True
            if in_multiline_comment or stripped_line.startswith('//'):
                comment_lines.append(line.strip())
            else:
                code_lines.append(line.strip())
            if stripped_line.endswith('*/'):
                in_multiline_comment = False

    code_content = ' '.join(code_lines)
    comment_content = ' '.join(comment_lines)
    all_content = ' '.join(all_lines)

    return file_path, all_content, code_content, comment_content


# 遍历目录中的所有Java文件
def process_directory(directory):
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                result.append(extract_code_and_comments(file_path))
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_and_analyze.py <commit-hash> <output-dir>")
        sys.exit(1)

    commit_hash = sys.argv[1]
    output_dir = sys.argv[2]
    result = process_directory(".")

    output_file = os.path.join(output_dir, f"{commit_hash}.csv")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["File Path", "All Content", "Code", "Comments"])

        for file_path, all_content, code_content, comment_content in result:
            writer.writerow([file_path, all_content, code_content, comment_content])

    print(f"Results saved to {output_file}")

