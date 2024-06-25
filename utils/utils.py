import numpy
import pandas as pd


def readBugId(dataset):
    df = pd.read_csv("../data/get_info/zookeeper/bug_info.csv",encoding="utf-8")
    return df['bugId']

def searchGitCommit(bugid,dataset):
    df = pd.read_csv('../data/get_info/'+dataset+'/bug_info.csv', encoding="utf-8")

    df['bugId'] = df['bugId'].astype(str)
    bugid = str(bugid).strip()

    # git_commit = df[df['bugId']==bugid]['fixedCmit'].values[0]
    git_commit = df[df['bugId']==bugid]['fixedCmit']

    if git_commit.empty:
        return None
    return git_commit.values[0]

def read_commit_to_df(git_commit,dataset):
    datapath ='../data/CK_Metrics/'+dataset+'/'+git_commit+'class.csv'
    try:
        df = pd.read_csv(datapath,usecols=[0,1],encoding='utf-8')
    except (UnicodeDecodeError,FileNotFoundError):
        try:
            df = pd.read_csv(datapath, encoding='latin1')
        except (UnicodeDecodeError,FileNotFoundError) as e:
            print("Exception: ", e.__class__.__name__,datapath)
            df = pd.DataFrame()
    return df

def readBugIdByTime(dataset):
    datapath = f"../data/get_info/{dataset}/bug_info.csv"
    try:
        df = pd.read_csv(datapath,encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(datapath, encoding='latin1')
        except UnicodeDecodeError as e:
            print("Exception: ", e.__class__.__name__,datapath)
            df = pd.DataFrame()
    # df['opendate'] = pd.to_datetime(df['opendate'], format='%a, %d %b %Y %H:%M:%S %z')
    df = df.sort_values(by = 'opendateUnixTime')
    # newdf = df [['bugId','opendate']]
    return df['bugId']






# if __name__ == '__main__':
#     readBugIdByTime()