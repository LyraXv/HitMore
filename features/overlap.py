#从bug报告的CCList和Comments中提取报告人员和开发人员，以及那些对代码文件做出贡献的人员(通过挖掘Git提交历史)，以检查他们之间有多少人重叠。
import csv
import os
import re
import xml.etree.ElementTree as ET

# 分割contributor字符串为姓名和邮箱，再通过邮箱提取用户名，如Benjamin Reed <breed@apache.org>
def parse_contributor(contributor):
    match = re.match(r'([^<]+) <([^>]+)>', contributor)
    if match:
        name = match.group(1).strip()
        email = match.group(2).strip()
        # 如果name都是小写字母且无空格
        if(bool(re.match(r'^[a-z]+$', name)) == True):
            username = name
        else:
            username = email.split('@')[0]
        # return name, email, username

        return username   #对于zookeeper等返回username
        # return name  # 对于tomcat和aspectj返回name
    return contributor.strip(), None


def get_contributors(commit, file):
    # 构建CSV文件的名称
    filename = f"{commit}.csv"
    filepath = os.path.join('Contributors', filename)

    # 检查文件是否存在
    if not os.path.isfile(filepath):
        print(f"文件 {filepath} 不存在.")
        return ""

    contributors = ""

    # 读取Contributors文件下的CSV文件
    with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)

        # 跳过CSV文件的标题行
        next(csv_reader)

        # 遍历每一行
        for row in csv_reader:
            if file in row[0]:
                # 如果文件路径匹配，提取贡献者名字
                # 合并后续所有列并分割贡献者名字
                contributors = ','.join(row[1:]).split(',')
                break

    # 解析贡献者名字和邮箱，提取name/username
    parsed_contributors = [parse_contributor(contributor) for contributor in contributors]
    return parsed_contributors
    # return contributors


def get_reporters_and_developers(bug_id):
    tree = ET.parse("zookeeper.xml")   # 载入xml数据
    root = tree.getroot()   # 获取根节点

    for br in root.findall('item'):
        id = int(br.find("id").text)

        if id == bug_id:
            # 存储reporter和comments作者的列表
            rep_dev = []

            # 提取reporter
            reporter = br.find('reporter')
            username = reporter.find('username').text
            name = reporter.find('name').text

            rep_dev.append(username)  # zookeeper等
            # rep_dev.append(name)   # tomcat/aspectj

            # 提取comments中的author
            developers = []
            comments = br.find('comments')
            for comment in comments.findall('value'):
                developer = comment.find('author').text   # zookeeper等 这里是username形式
                # developer = comment.find('developer').text   # tomcat 这里是name形式
                # developer = comment.find('commenter').text   # aspectj 这里是name形式

                rep_dev.append(developer)

            return rep_dev
    return None


def calculate_overlap(contributors, rep_dev):
    # 计算贡献者与报告者开发者之间的重叠部分
    overlap = set(contributors) & set(rep_dev)
    overlap_count = len(overlap)
    if(min(len(contributors), len(rep_dev))) == 0:
        return 0
    return overlap_count / min(len(contributors), len(rep_dev))


if __name__ == "__main__":
    # 示例输入
    commit = "5c85a236c9eb0805ea8389a52dab3b1bc0efadac"   # 输入的commit编号
    file = "zookeeper-client/zookeeper-client-java/src/main/java/org/apache/zookeeper/client/ConnectStringParser.java"   # 输入的文件路径

    contributors = get_contributors(commit, file)
    print("contributors:", contributors)

    bug_id = 3113 # 输入的bug id
    rep_dev = list(set(get_reporters_and_developers(bug_id)))
    print("reporters and developers:", rep_dev)

    overlap = calculate_overlap(contributors, rep_dev)
    print("overlap:", overlap)



