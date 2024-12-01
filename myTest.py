def getOtherApproachBuggyData(file,i): # i = {0:amalgam,1:bugLocator,2:blizzard}
    print(f"=====approaches: {i}======")
    res_pred = []
    res = open(file['bugPredict'][i],'r')
    for line in res:
        if i == 0:
            split_line = list(line.strip('\n').split('	'))
            split_line[1] = split_line[1].rstrip('.')
        else: # bugLocator
            split_line = list(line.strip('\n').split(','))
        if split_line[0].isdigit():
            res_pred.append(split_line)
    res.close()
    if i ==2:
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'rank', 'path'])
        res_df['rank']= res_df['rank'].astype(int)
        res_df['rank'] = res_df['rank'] -1
    else:
        res_df = pd.DataFrame(res_pred, columns=['bugId', 'path', 'rank', 'score'])
    return res_df