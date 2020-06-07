import  xml.dom.minidom
import time
import string
import nltk
from nltk.stem import WordNetLemmatizer
from allennlp.common.checks import ConfigurationError
from tqdm import tqdm

def Preprocessing(text):

    text = text.lower() # 将所有的单词转换成小写字母

    for c in string.punctuation:
        text = text.replace(c," ")  # 将标点符号转换成空格

    filtered = nltk.word_tokenize(text)  # 分词

    # stem
    wl = WordNetLemmatizer()
    filtered = [wl.lemmatize(w) for w  in filtered]  # 词形还原

    return " ".join(filtered)

label={'PerfectMatch':"1",
        'Relevant':"1",
        'Irrelevant':"0"}

def get_label(s):
    if s in label.keys():
        return label[s]
    raise ConfigurationError(f"%s label not found"%s)

#数据集的问题：
#1. 有些数据集没有orgbody，有些body很长
#2. orgsubject可能信息更重要，考虑放在前边或者分开处理
#3. 有很多些符号，缩略词，标点，会有表情符号
#4. 很多口语化表述，没有逻辑
#5. orgbody长短不一
#6. 有些relevant和irrelevant排名不一致，排名高的并不relevant

if __name__ == '__main__':
    # file_name="./train.txt"
    # files=['./Final Task/train/SemEval2016-Task3-CQA-QL-train-part2.xml','./Final Task/train/SemEval2016-Task3-CQA-QL-train-part1.xml']

    file_name="./dev.txt"
    files=['./Final Task/dev/SemEval2016-Task3-CQA-QL-dev.xml']
    with open(file_name,'w') as f:
        for file in files:
            dom=xml.dom.minidom.parse(file)
            root=dom.documentElement
            NewQ=root.getElementsByTagName('OrgQuestion')
            # s1=0
            # s2=0
            # matrix={'PerfectMatch':0,
            #         'Relevant':0,
            #         'Irrelevant':0}
            last_q1id=''
            print('begin to parse file : %s'%file)
            for q1 in tqdm(NewQ):
                q1_subject=Preprocessing(q1.getElementsByTagName('OrgQSubject')[0].firstChild.data)
                question1=Preprocessing(q1.getElementsByTagName('OrgQBody')[0].firstChild.data)
                # q1_id=q1.getAttribute('ORGQ_ID')
                # if last_q1id=='':
                #     last_q1id=q1_id
                # elif last_q1id!=q1_id:
                #     print("...........................")
                q2=q1.getElementsByTagName('RelQuestion')[0]
                q2_id = q2.getAttribute('RELQ_ID')
                q2_subject=Preprocessing(q2.getElementsByTagName('RelQSubject')[0].firstChild.data)
                q2_rank = q2.getAttribute('RELQ_RANKING_ORDER')
                q2_relevance = q2.getAttribute('RELQ_RELEVANCE2ORGQ')
                try:
                    question2=Preprocessing(q2.getElementsByTagName('RelQBody')[0].firstChild.data)
                except:
                    question2=''
                    # s2+=1
                f.write(q1_subject+' '+question1+'\t'+q2_subject+' '+question2+'\t'+get_label(q2_relevance)+'\n')
            # last_q1id=q1_id





