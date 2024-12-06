#!/bin/bash

repo="D:\HitMore\Dataset\zookeeper"   # repository path

output_dir="zookeeperDb"
mkdir -p "$output_dir"

newestCmitSha=$(git -C "$repo" reflog | tail -1 | cut -f1 -d " ")

# Loop through the commit file
for line in $(cat ordered_bugCmit_zookeeper); do

  currentCmit=$(echo $line | cut -f2 -d ",")

  git -C "$repo" checkout -f "${currentCmit}~1"
  
  # output file path
  output_file="$output_dir/${currentCmit}.und"
  echo $output_file
  
  und create -db $output_file -languages java add $repo analyze -all

done

git -C "$repo" checkout -f "$newestCmitSha"
