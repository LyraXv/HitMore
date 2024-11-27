#!/bin/bash


COMMITS_FILE=$1
OUTPUT_DIR="CodeCorpus"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 处理每个commit
while IFS=, read -r id commit_hash timestamp; do
  echo "Processing commit: $commit_hash"

  # 获取前一个commit哈希值
  PREVIOUS_COMMIT_HASH=$(git log --format="%H" -n 2 $commit_hash | tail -n 1)

  if [ -z "$PREVIOUS_COMMIT_HASH" ]; then
    echo "Unable to find previous commit for $commit_hash"
    continue
  fi

  echo "Previous commit hash: $PREVIOUS_COMMIT_HASH"

  # 检出前一个commit
  git checkout $PREVIOUS_COMMIT_HASH

  # 调用Python脚本处理
  python3 extract_codeCorpus.py $PREVIOUS_COMMIT_HASH $OUTPUT_DIR
done < "$COMMITS_FILE"

# 返回到最新的commit
git checkout main
