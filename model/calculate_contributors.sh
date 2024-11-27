#!/bin/bash

# 获取前一个commit
get_previous_commit() {
    local commit_hash=$1
    previous_commit=$(git log --pretty=format:%P -n 1 $commit_hash 2>/dev/null | awk '{print $1}')
    if [[ -z "$previous_commit" ]]; then
        echo "Error: Unable to find the previous commit for $commit_hash" >&2
        exit 1
    fi
    echo $previous_commit
}

# 获取特定commit的修改记录
get_file_modifications() {
    local commit_hash=$1
    git checkout $commit_hash --quiet 2>/dev/null
    if [[ $? -ne 0 ]]; then
        echo "Error: Unable to checkout commit $commit_hash" >&2
        exit 1
    fi
    git log --pretty=format:"%ae" --name-only
}

# 统计每个文件的开发人员修改数量
calculate_contributors() {
    local logs="$1"
    local output_file="$2"
    declare -A file_contributors
    declare -a current_commit_files
    declare -a current_commit_contributors

    while IFS= read -r line; do
        if [[ -z "$line" ]]; then
            # 更新当前commit的文件的开发人员
            for file in "${current_commit_files[@]}"; do
                file_contributors["$file"]+="${current_commit_contributors[*]} "
            done
            current_commit_files=()
            current_commit_contributors=()
        elif [[ "$line" == *@* ]]; then
            # 添加开发人员
            current_commit_contributors+=("$line")
        else
            # 添加文件
            current_commit_files+=("$line")
        fi
    done <<< "$logs"

    # 最后一次更新
    for file in "${current_commit_files[@]}"; do
        file_contributors["$file"]+="${current_commit_contributors[*]} "
    done

    # 输出每个文件的开发人员数量到CSV文件
    {
        echo "File,Contributors"
        for file in "${!file_contributors[@]}"; do
            unique_contributors=$(echo "${file_contributors[$file]}" | tr ' ' '\n' | sort | uniq | wc -l)
            echo "$file,$unique_contributors"
        done
    } > "$output_file"
}

# 处理每个commit
process_commit() {
    local commit_hash=$1
    local output_file=$2

    if [[ -z "$commit_hash" ]]; then
        echo "Commit hash is empty" >> "$output_file"
        return
    fi

    previous_commit=$(get_previous_commit "$commit_hash")
    if [[ -z "$previous_commit" ]]; then
        echo "Unable to find the previous commit for $commit_hash" >> "$output_file"
        return
    fi

    logs=$(get_file_modifications "$previous_commit")
    calculate_contributors "$logs" "$output_file"

    # 回到原来的分支
    git checkout - --quiet
}

# 主函数
main() {
    local txt_file=$1
    local output_dir="NDEV"
    
    if [[ -z "$txt_file" ]]; then
        echo "请提供文本文件路径" >&2
        exit 1
    fi

    if [[ ! -f "$txt_file" ]]; then
        echo "文本文件不存在: $txt_file" >&2
        exit 1
    fi

    # 创建输出目录
    mkdir -p "$output_dir"

    while IFS=, read -r col1 commit_hash col3; do
        output_file="${output_dir}/${commit_hash}.csv"
        process_commit "$commit_hash" "$output_file"
    done < "$txt_file"
}

# 执行主函数
txt_file="$1"
main "$txt_file"
