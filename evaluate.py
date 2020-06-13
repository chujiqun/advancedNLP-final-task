import json
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.common import Params
from allennlp.common.util import import_submodules
import torch
import  xml.dom.minidom
from preprocess import Preprocessing
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
import numpy

label_dict={'PerfectMatch':1,
        'Relevant':1,
        'Irrelevant':0}

def get_instance_from_link(q1,q2,vocab,get_rel=True):
    """

    :param q1:  The pointer of xml files
    :param vocab:  vocaburary
    :return: instance: instance; label : int;
    """
    q1_subject=Preprocessing(q1.getElementsByTagName('OrgQSubject')[0].firstChild.data)
    try:
        question1=Preprocessing(q1.getElementsByTagName('OrgQBody')[0].firstChild.data)
    except:
        question1=''
    q2_subject=Preprocessing(q2.getElementsByTagName('RelQSubject')[0].firstChild.data)
    q2_relevance = q2.getAttribute('RELQ_RELEVANCE2ORGQ')
    try:
        question2=Preprocessing(q2.getElementsByTagName('RelQBody')[0].firstChild.data)
    except:
        question2=''
    OrgQ=q1_subject+" "+question1
    RelQ=q2_subject+" "+question2
    x=datareader.text_to_instance(OrgQ,RelQ)
    x.index_fields(vocab)
    if get_rel:
        return x,label_dict[q2_relevance]
    else:
        return x

def MAP(scores,label_list):
    """
    calculate the map of this rank problem
    :param scores:  tensor [0.9,0.9,0.5,0.8,....]这样的一维列表
    :param label_list: [1,0,1,0,1,0,1,1,1,1] 这样的列表，都是整数，只能是1和0,1代表相关
    :return: map score：float
    """
    sorted, indices = torch.sort(scores,descending=True)
    map=0
    score_rank=0.
    rel_num=0.
    for index in indices:
        score_rank+=1
        if label_list[index]==1:
            rel_num+=1
            map+= rel_num/score_rank
    if rel_num==0:
        return None
    else:
        return map/rel_num

class Write_outfile(object):
    def __init__(self,Wfile_name):
        self.name=Wfile_name
        self.f=open(Wfile_name,"w")

    def __del__(self):
        print("The result file is in %s"%self.name)
        self.f.close()

    def write_line(self,str):
        self.f.write(str)

if __name__ == '__main__':
    # model_name = "mv_lstm"
    model_name = "match_pyramid"
    cuda_device = -1
    library='library'
    # files=['./Final Task/Test/SemEval2017-task3-English-test-input.xml']
    files=['./Final Task/dev/SemEval2016-Task3-CQA-QL-dev.xml']
    attn="_cos_hyper"
    calculate_map=True
    write_file=False
    Wfile_name="out.txt"

    import_submodules(library)
    model_config = "config/%s_eval.jsonnet" % model_name
    overrides=overrides = json.dumps({"trainer": {"cuda_device": cuda_device}})
    params=Params.from_file(model_config, overrides)
    model_file='checkpoint/%s%s/'% (model_name,attn)
    iterator = DataIterator.from_params(params.pop("iterator"))

    torch.manual_seed(0)
    numpy.random.seed(0)

    if write_file:
        wf=Write_outfile(Wfile_name)

    print("Loading vocabulary")
    vocab=Vocabulary.from_files(model_file+'vocabulary')

    print('Initialing model')
    model=Model.from_params(vocab=vocab,params=params.pop('model'))
    print("Loading Model file from %s"%(model_file+'best.th'))
    with open(model_file+'best.th','rb') as f:
        model.load_state_dict(torch.load(f,encoding='utf-8'))

    iterator.index_with(vocab)
    dataset_reader_params=params.pop('dataset_reader')
    datareader=DatasetReader.from_params(dataset_reader_params)
    model.eval()

    #读取文件数据
    for file in files:
        dom=xml.dom.minidom.parse(file)
        root=dom.documentElement
        OrgQ_list=root.getElementsByTagName('OrgQuestion')
        q1_last=None
        map=0
        iter=0
        rel_list=[]
        label_list=[]
        OrgQ_idlist=[]
        RelQ_idlist=[]
        for q1 in OrgQ_list:
            q1_id=q1.getAttribute('ORGQ_ID')
            if q1_last is None:
                q1_last=q1_id
            if q1_last!=q1_id:
                a=torch.tensor(rel_list)
                if calculate_map:
                    map_score=MAP(a,label_list)
                    if map_score is not None:
                        iter += 1
                        map=(map*(iter-1)+map_score)/iter
                        print("iter: %d; map: %f" % (iter, map_score))
                        if map_score<0.5:
                            print(q1_last)
                            print(a,label_list)
                    label_list = []

                if write_file:
                    for i in range(len(a)):
                        wf.write_line('%s\t%s\t%d\t%f\t%s\n'%(OrgQ_idlist[i],RelQ_idlist[i],1,a[i].data,"true"))

                rel_list=[]
                OrgQ_idlist=[]
                RelQ_idlist=[]
            q2=q1.getElementsByTagName('RelQuestion')[0]
            q2_id = q2.getAttribute('RELQ_ID')
            if calculate_map:
                x,label=get_instance_from_link(q1,q2,vocab)
                label_list.append(label)
            else:
                x=get_instance_from_link(q1,q2,vocab,get_rel=False)
            y=x.as_tensor_dict(x.get_padding_lengths())
            y['Orgquestion']['tokens']=y['Orgquestion']['tokens'].unsqueeze(0)
            y['Relquestion']['tokens']=y['Relquestion']['tokens'].unsqueeze(0)
            rel_score=model(**y)
            rel_list.append(rel_score['label_logits'].squeeze(1).data)
            OrgQ_idlist.append(q1_id)
            RelQ_idlist.append(q2_id)
            q1_last=q1_id
        print("total map is %f"%map)

