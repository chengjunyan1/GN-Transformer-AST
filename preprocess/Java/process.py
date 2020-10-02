import numpy as np
import javalang,jpype,math,time,sys,tqdm,subprocess

from datatools import *
from code_tokenizer import CodeTokenizer


GENERATING_DATASET=True

if GENERATING_DATASET:
    # Tokenizer
    T=javalang.tokenizer
    
    # Subtokenizer
    S = CodeTokenizer()

def tokenize(s):
    token=[]
    posit=[]
    types=[]
    t=T.tokenize(s)
    for i in list(t):
        token.append(S.tokenize(i.value))
        posit.append(tuple(i.position))
        types.append(str(i).split(' ')[0])
    return token, posit, types

def slice_str(s,start,end):
    start_row=start[0]-1
    start_col=start[1]-1
    end_row=end[0]-1
    end_col=end[1]-1
    s=s.split('\n')
    res=''
    for i in range(start_row,end_row+1):
        if i==start_row: 
            if end_row==start_row: res+=s[i][start_col:end_col+1]
            else: res+=s[i][start_col::]
        elif i==end_row: res+=s[i][:end_col+1]
        else: res+=s[i]
        res+='\n'
    return res[:-1]

def tuple_le(a,b):
    if a[0]<b[0]: 
        return True
    elif a[0]>b[0]: 
        return False
    else: 
        return a[1]<=b[1]
    
def fetch_ids(tk,ps,node):
    low=(node['begin_row'],node['begin_col'])
    high=(node['end_row'],node['end_col'])
    ids=[]
    for i in range(len(tk)):
        if tuple_le(low,ps[i]) and tuple_le(ps[i],high):
            ids.append(i)
    return ids

def tok2id(tok,voc):
    index=[]
    for t in tok:
        ti=[]
        for i in t:
            if i!='':
                ti.append(voc[i])
        index.append(ti)
    return index

def type2id(tps,voc):
    index=[]
    for i in tps:
        index.append(voc[i])
    return index

def ast2id(ast,tok,pos,nodevoc):
    init=[]
    types=[]
    nodes=[]
    edges=[]
    for i in ast:
        if (i['begin_row']==0 and i['begin_col']==0 
            and i['end_row']==0 and i['end_col']==0):
            types.append(nodevoc['null'])
        else:
            types.append(nodevoc[i['type']])    
        nodes.append(i['id'])
        for e in i['children']:
            edges.append((i['id'],e))
        init_ids=fetch_ids(tok,pos,i)
        init.append(init_ids)
    return types,init,nodes,edges

def com2id(com,voc):
    index=[]
    for i in com:
        index.append(voc[i])
    return index

def expand_subword(words,wtype,ginit):
    new_coord={}
    new_words=[]
    new_wtype=[]
    new_ginit=[]
    offset=0
    for i in range(len(words)):
        new_coord[i]=[j+offset for j in range(len(words[i]))]
        new_words+=words[i]
        for j in range(len(words[i])): new_wtype.append(wtype[i])
        offset+=len(words[i])
    for n in ginit:
        giniti=[]
        for j in n: giniti+=new_coord[j]
        new_ginit.append(giniti)
    return new_words,new_wtype,new_ginit

class data_obj: 
    def __init__(self, fid, ast, tok, pos, tps):
        self.fid = fid
        self.ast = ast
        self.tok = tok
        self.pos = pos 
        self.tps = tps 

def make_dataset(rawpath='./Raw/'):
    astok_train=[]
    astok_test=[]
    astok_valid=[]
    for mode in ['test','valid','train']: 
        print('Loading '+mode+' data.')
        raw=loadjsonb(rawpath+'ast/'+mode+'_ast.json')
        print('Loaded '+mode+' data.')  
        for k in raw:
            fidk=int(k['fid'])
            astk=k['ast']
            srck=k['src']
            tk,ps,tp=tokenize(srck)
            if mode=='train': astok_train.append(data_obj(fidk,astk,tk,ps,tp))
            if mode=='test': astok_test.append(data_obj(fidk,astk,tk,ps,tp))
            if mode=='valid': astok_valid.append(data_obj(fidk,astk,tk,ps,tp))
        
    print('Generating dataset.')
    for mode in ['train','test','valid']:
        if mode=='train': d=astok_train
        if mode=='test': d=astok_test
        if mode=='valid': d=astok_valid
        fids=[]
        with open(rawpath+mode+'_com.json','r') as f: 
            tcoms=json.load(f)
        for i in d:
            fids.append(int(i.fid))
            mname='dev' if mode=='valid' else mode
            with open('./java/'+mname+'/code.original_subtoken','a',encoding='utf-8') as f:
                f.write(tok2seq(i.tok[4:-1],i.tps[4:-1])+'\n')
            with open('./java/'+mname+'/javadoc.original','a') as f:
                f.write(tcoms[i.fid])
        mname='dev' if mode=='valid' else mode
        with open('./java/'+mname+'/fids','a') as f:
            f.write(str(fids))
            
    print('Generating rawdata.')
    subword_vocab={'<PAD>':0,'<UNK>':1}
    word_types_vocab={'<PAD>':0,'<UNK>':1}
    node_types_vocab={'<PAD>':0,'<UNK>':1,'null':2}
    subword_index=len(subword_vocab)
    word_types_index=len(word_types_vocab)
    node_types_index=len(node_types_vocab)
    for mode in ['train','test','valid']:
        if mode=='train': d=astok_train
        if mode=='test': d=astok_test
        if mode=='valid': d=astok_valid
        for data in d:
            tok=data.tok
            tps=data.tps
            ast=data.ast
            tok=tok_normalize(tok,tps)
            for t in tok:
                for w in t: 
                    if w not in subword_vocab:
                        subword_vocab[w]=subword_index
                        subword_index+=1
            for t in tps:
                if t not in word_types_vocab:
                    word_types_vocab[t]=word_types_index
                    word_types_index+=1
            for t in ast:
                if t['type'] not in node_types_vocab:
                    node_types_vocab[t['type']]=node_types_index
                    node_types_index+=1
    com_vocab,com_vocab_r,max_coms,avg_coms=create_coms_vocab()
    vocab={}
    vocab['subword_vocab']=subword_vocab
    vocab['word_types_vocab']=word_types_vocab
    vocab['node_types_vocab']=node_types_vocab
    savepkl(vocab,rawpath+'vocabs')
    
    for mode in ['train','test','valid']:
        if mode=='train': d=astok_train
        if mode=='test': d=astok_test
        if mode=='valid': d=astok_valid
        dataset={}
        for data in d:
            token=data.tok
            types=data.tps
            ast=data.ast
            pos=data.pos
            fid=data.fid
            token=tok_normalize(token,types)
            words=tok2id(token,subword_vocab)
            wtype=type2id(types,word_types_vocab)
            gtype,ginit,gnode,gedge=ast2id(ast,token,pos,node_types_vocab)
            words,wtype,ginit=expand_subword(words,wtype,ginit)
            data_line=[words,wtype,gnode,gedge,gtype,ginit]
            dataset[fid]=data_line
        with open(rawpath+mode+'_datas.pkl','wb') as f: 
            pickle.dump(dataset, f)

def create_coms_vocab(rawpath='./Raw/'):
    with open(rawpath+'train_com.json','r') as f:
        d=json.load(f)
    with open(rawpath+'test_com.json','r') as f:
        d1=json.load(f)
    with open(rawpath+'valid_com.json','r') as f:
        d2=json.load(f)
    d.update(d1)
    d.update(d2)
    max_com_length=0
    avg_com_length=0
    vocab={'<PAD>':0,'<SOS>':1,'<EOS>':2,'<UNK>':3}
    vocab_r={0:'<PAD>',1:'<SOS>',2:'<EOS>',3:'<UNK>'}
    index=len(vocab)
    for i in d:
        coms=d[i]
        if len(coms)>max_com_length:
            max_com_length=len(coms)
        avg_com_length+=len(coms)
        for j in coms:
            if j not in vocab:
                vocab[j]=index
                vocab_r[index]=j
                index+=1
    return vocab,vocab_r,max_com_length,avg_com_length/len(d)


def tok_normalize(tok,tps):
    nt=[]
    for i in range(len(tok)):
        if tps[i]=='String':
            nt.append(['<STR>'])
        elif tps[i] in ['DecimalInteger','OctalInteger','HexInteger','FloatingPoint',
                        'BinaryInteger','DecimalFloatingPoint','Integer','HexFloatingPoint']:
            nt.append(['<NUM>'])
        elif tok[i]==['Unknown', 'Class']:
            nt.append(['<CLS>'])
        else:
            nt.append(tok[i])
    return nt

def tok2seq(tok,tps=None):
    s=''
    if tps!=None: tok=tok_normalize(tok,tps)
    for i in tok:
            for k in i:
                if k!='': s+=k+' '
    return s[:-1]

def count_file_lines(file_path):
    num = subprocess.check_output('find /c /v "" '+file_path)
    return int(num.split()[-1])
 

def truncate_syntax(d,wcut):
    wtype,gnode,gedge,gtype,ginit=d
    wtype=wtype[4:-1]
    vwtype=wtype[:wcut]
    boundary=wcut+4
    vgnode=[]
    vgtype=[]
    vginit=[]
    nonleaf=set()
    for e in gedge: nonleaf.add(e[0])
    for n in range(len(ginit)):
        add=True
        if ginit[n]==[] and gnode[n] not in nonleaf: add=False
        for i in ginit[n]:
            if i>=boundary or i<4: add=False
        if add:
            for i in range(len(ginit[n])): ginit[n][i]-=4
            vgnode.append(gnode[n])
            vgtype.append(gtype[n])
            vginit.append(ginit[n])
    vgedge=[]
    for e in gedge:
        if e[0] in vgnode and e[1] in vgnode:
            vgedge.append(e)
    return [vwtype,vgnode,vgedge,vgtype,vginit]

def make_adddata(wcut=150):
    for mode in ['train','test','valid']:
        with open('./Data/'+mode+'_datas.pkl','rb') as f: 
            d=pickle.load(f)
        with open('./java/'+mode+'/fids','rb') as f: 
            s=f.read()
        for ind in eval(s):
            words,wtype,gnode,gedge,gtype,ginit=d[ind]
            line=truncate_syntax([wtype,gnode,gedge,gtype,ginit],wcut)
            mname='dev' if mode=='valid' else mode
            with open('./java/'+mname+'/syntax.adddata','a') as f:
                f.write(str(line)+'\n')
        

if __name__=='__main__':
    save='./java/'
    for mode in ['train','test','dev']:
        if not os.path.exists(save+mode): os.makedirs(save+mode)
        if not os.path.exists(save+mode): os.makedirs(save+mode)
        if not os.path.exists(save+mode): os.makedirs(save+mode)
    make_dataset()
    make_adddata()