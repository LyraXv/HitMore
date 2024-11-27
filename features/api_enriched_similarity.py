import os
import re
import csv
import javalang
import string
import pandas as pd
import similarity as CS


def read_api_descriptions(file_path):
    # 读取API文档
    api_descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        class_name = ""
        description_lines = []
        is_description = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('Class/Interface:'):
                if class_name and description_lines:
                    description = ' '.join(description_lines)
                    # 删除中文字符和"?"和空格
                    class_name = re.sub(r'[^\x00-\x7F]+|[?]', '', class_name)
                    class_name = class_name.replace(' ', '')
                    api_descriptions[class_name] = description
                class_name = line.split(':', 1)[1].strip()
                description_lines = []
                is_description = False
            elif line.startswith('Description:'):
                is_description = True
                description_lines.append(line.split(':', 1)[1].strip())
            elif is_description:
                description_lines.append(line)
                
        if class_name and description_lines:
            description = ' '.join(description_lines)
            class_name = re.sub(r'[^\x00-\x7F]+|[?]', '', class_name)
            class_name = class_name.replace(' ', '')
            api_descriptions[class_name] = description

    return api_descriptions


def tokenize_java_code(code):
    tokens = []
    lexer = javalang.tokenizer.tokenize(code)
    for token in lexer:
        tokens.append(token.value)
    return tokens


def extract_identifiers_from_code(code):
    package_names = set()

    # parse code
    tree = javalang.parse.parse(code)
#     print(tree.children[2][0].body)

    # Extract package name
    for path, node in tree.filter(javalang.tree.PackageDeclaration):
        package_names.add(node.name)

    # Extract import names
    for path, node in tree.filter(javalang.tree.Import):
        package_names.add(node.path)
    
    code = re.sub(r'[\./{}[\]<>;=()]', ' ', code)
    # Extract code tokens
    tokens = set(tokenize_java_code(code))

    return package_names, tokens


def generate_api_from_descriptions(package_names, tokens, api_descriptions):
    api = []
    
    # Add package description
    for package_name in package_names:
        if package_name in api_descriptions:
            api.append(api_descriptions[package_name])
    
    # Add token descriptions
    for token in tokens:
        if token in api_descriptions:
            api.append(api_descriptions[token])
            
    return '\n'.join(api)


def main(csv_file, api_doc_dir, output_file):
    results = []

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_name = row['file']
            commit = row['commit']
            br = row['bug report']
            code = row['code']
            all_content = row['all content']
            
            
            # 读取对应的API描述txt
            api_description_file = os.path.join(api_doc_dir, f"{commit}.txt")
            api = ""
            if os.path.exists(api_description_file):
                api_descriptions = read_api_descriptions(api_description_file)
                package_names, tokens = extract_identifiers_from_code(code)
                # 组合所有API信息
                api = generate_api_from_descriptions(package_names, tokens, api_descriptions)
                
            else:
                print(f"API description file not found for commit: {commit}")
                
            text = api + all_content
            t1 = CS.preprocess(br) 
            t2 = CS.preprocess(text)

            Str_t1 = " ".join(t1)
            Str_t2 = " ".join(t2)
            
            api_enriched_sim = CS.cosine_sim(Str_t1, Str_t2)
            
            results.append({'file': file_name, 'api': api, 'api-enriched sim': api_enriched_sim})
    
    
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['file', 'api', 'api-enriched sim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)


csv_file = 'xxx.csv'
api_doc_dir = 'API_DOC'
output_file = 'api_enriched_sim.csv'
main(csv_file, api_doc_dir, output_file)