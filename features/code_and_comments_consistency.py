import re
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import cossim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk


# 下载NLTK数据包
# nltk.download('punkt')
# nltk.download('stopwords')

# 示例文本
code = "LDA is a generative probabilistic model for collections of discrete data such as text corpora."
comments = "LDA is used for topic modeling and document classification in natural language processing."
#需要从CodeCorpus读取源文件的代码Code列和注释Comments列

ps = PorterStemmer()

# 数据预处理
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    # 分割驼峰命名法单词并进行词干提取
    processed_words = []
    for word in words:
        if word.isalnum() and word not in stop_words:
            split_words = re.sub(r"([A-Z])", r" \1", word).split()
            stemmed_words = [ps.stem(base) for base in split_words]
            processed_words.extend(stemmed_words)
    return processed_words

# 预处理后的文本
texts = [preprocess(code), preprocess(comments)]

# 生成词典和语料库
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 训练LDA模型
lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# 计算两段文字的主题分布
vec1 = lda_model[corpus[0]]
vec2 = lda_model[corpus[1]]

# 计算主题分布的余弦相似性
similarity = cossim(vec1, vec2)

print(f"Similarity between the two texts: {similarity:.4f}")
