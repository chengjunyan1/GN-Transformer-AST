import ast,re,os
from queue import Queue
from io import BytesIO
import numpy as np

from ast_utils import parse_source
from code_tokenizer import CodeTokenizer
import tokenize as T

S = CodeTokenizer()
nodevoc={'null':0}
nodevoc_id=len(nodevoc)   
typevoc={'null':0}
typevoc_id=len(nodevoc) 


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

def ast2id(astnodes,tok,pos):
    global nodevoc,nodevoc_id
    init=[]
    types=[]
    nodes=[]
    edges=[]
    for i in astnodes:
        init_ids=fetch_ids(tok,pos,i)
        if (i['begin_row']==0 and i['begin_col']==0 
            and i['end_row']==0 and i['end_col']==0):
            types.append(nodevoc['null'])
        else:
            if i['type'] not in nodevoc: 
                nodevoc[i['type']]=nodevoc_id
                nodevoc_id+=1
            types.append(nodevoc[i['type']])    
        nodes.append(i['id'])
        for e in i['children']:
            edges.append((i['id'],e))
        init.append(init_ids)
    return nodes,edges,types,init

def tok2seq(tok):
    s=''
    for i in tok:
        for k in i:
            s+=k+' '
    return s[:-1]
 
def truncate_syntax(d,wcut):
    wtype,gnode,gedge,gtype,ginit=d
    vwtype=wtype[:wcut]
    boundary=wcut
    vgnode=[]
    vgtype=[]
    vginit=[]
    nonleaf=set()
    for e in gedge: nonleaf.add(e[0])
    for n in range(len(ginit)):
        add=True
        if ginit[n]==[] and gnode[n] not in nonleaf: add=False
        for i in ginit[n]:
            if i>=boundary: add=False
        if add:
            vgnode.append(gnode[n])
            vgtype.append(gtype[n])
            vginit.append(ginit[n])
    vgedge=[]
    for e in gedge:
        if e[0] in vgnode and e[1] in vgnode:
            vgedge.append(e)
    return [vwtype,vgnode,vgedge,vgtype,vginit]

def to_src(s):
    r=''
    for i in s.strip().split():
        if i=='': continue
        elif i=='DCNL': r+='\n'
        elif i=='DCSP': r+='\t'
        else: r+=i+' '
    return r[:-1]

def tokenize(q):
    global typevoc,typevoc_id
    toks=T.tokenize(BytesIO(q.encode('utf-8')).readline)
    tks=[]
    tps=[]
    pos=[]
    while(True):
        tok=next(toks,-1)
        if tok==-1: break
        if tok.string in ['','\n','\t']: continue
        if tok.start==tok.end: continue
        sts=[]
        for i in S.tokenize(tok.string):
            if i!='': sts.append(i)
        if sts==[]: continue
        if tok.type==2: tks.append(['<NUM>'])
        elif tok.type==3: tks.append(['<STR>'])
        else: tks.append(sts)
        if tok.type not in typevoc:
            typevoc[tok.type]=typevoc_id
            typevoc_id+=1
        tps.append(typevoc[tok.type])
        pos.append(tok.start)
    return tks,pos,tps

class node:
    def __init__(self):
        self.d={}
    
def traversal(q):
    tree=parse_source(q)
    Q=Queue()
    N=Queue()
    ind=-1
    root=node()
    root.d['parent']=-1
    Q.put(tree)
    N.put(root)
    Nodes=[root]
    while not Q.empty():
        tree=Q.get()
        cur=N.get()
        ind+=1
        cur.d['id']=ind
        cur.d['type']=tree.__class__.__name__
        if cur.d['type']=='Load': continue
        if not hasattr(tree,'first_token'): continue
        if not hasattr(tree,'last_token'): continue
        cur.d['begin_row']=tree.first_token.start[0]
        cur.d['begin_col']=tree.first_token.start[1]
        cur.d['end_row']=tree.last_token.end[0]
        cur.d['end_col']=tree.last_token.end[1]
        tree_it=ast.iter_child_nodes(tree)
        while(True):
            child=next(tree_it,-1)
            if child==-1: break
            Q.put(child)
            vn=node()
            vn.d['parent']=ind
            N.put(vn)
            Nodes.append(vn)
    children={}
    for n in Nodes:
        i=n.d
        if 'begin_row' not in i: continue
        if i['parent']==-1: continue
        if i['parent'] not in children:
            children[i['parent']]=[]
        children[i['parent']].append(i['id'])
    astnodes=[]
    for n in Nodes:
        i=n.d
        if 'begin_row' not in i: continue
        i['children']=children[i['id']] if i['id'] in children else []
        astnodes.append(i)
    return astnodes

def process_line(q,wcut=400):
    astnodes=traversal(q)
    tks,pos,wtype=tokenize(q)
    vtype=[]
    for i in range(len(tks)):
        for j in range(len(tks[i])): vtype.append(wtype[i])
    nodes,edges,types,inits=ast2id(astnodes,tks,pos)
    src=tok2seq(tks)
    return src,truncate_syntax([vtype,nodes,edges,types,inits],wcut)


# from: https://github.com/wanyao1992/code_summarization_public/blob/master/script/github/python_process.py
def clean_comment(description):
    description = description.replace(' DCNL DCSP', ' ')
    description = description.replace(' DCNL ', ' ')
    description = description.replace(' DCSP ', ' ')

    description = description.lower()

    description = description.replace("this's", 'this is')
    description = description.replace("that's", 'that is')
    description = description.replace("there's", 'there is')

    description = description.replace('\\', '')
    description = description.replace('``', '')
    description = description.replace('`', '')
    description = description.replace('\'', '')

    removes = re.findall("(?<=[(])[^()]+[^()]+(?=[)])", description)
    for r in removes:
        description = description.replace('('+r+')', '')

    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description)
    for url in urls:
        description = description.replace(url, 'URL')

    description = description.split('.')[0]
    description = description.split(',')[0]
    description = description.split(':param')[0]
    description = description.split('@param')[0]
    description = description.split('>>>')[0]

    description = description.strip().strip('\n') + ' .'

    return description


def process(rawpath='./Raw/'):
    allcoms=[]
    allsrcs=[]
    for mode in ['train','test','valid']:
        with open(rawpath+'/'+mode+'/data_ps.declbodies.'+mode,'r',encoding='utf8') as f:
            s=f.readlines()
        with open(rawpath+'/'+mode+'/data_ps.descriptions.'+mode,'r',encoding='utf8', errors='ignore') as f:
            c=f.readlines()
        for i in range(len(s)):
            if len(s[i].split())>1680: continue
            allcoms.append(clean_comment(c[i]))
            allsrcs.append(s[i])
    pidx = list(np.random.permutation(len(allsrcs)))
    split={'train':pidx[0:int(len(allsrcs)*0.6)],
           'test':pidx[int(len(allsrcs)*0.6):int(len(allsrcs)*0.8)],
           'valid':pidx[int(len(allsrcs)*0.8):int(len(allsrcs))]}
    srcs=allsrcs
    coms=allcoms
    count=0
    for mode in ['train','test','valid']:
        print('Processing',mode)
        for i in split[mode]:
            src=to_src(srcs[i])
            try:
                s,dataline=process_line(src)
            except: continue
            if s=='' or coms[i]=='': continue
            if coms[i]==' .': continue
            mname='dev' if mode=='valid' else mode
            with open('./python/'+mname+'/code.original_subtoken','a',encoding='utf8') as f:
                f.write(s+'\n')
            with open('./python/'+mname+'/javadoc.original','a',encoding='utf8') as f:
                f.write(coms[i]+'\n')
            with open('./python/'+mname+'/syntax.adddata','a',encoding='utf8') as f:
                f.write(str(dataline)+'\n')
            count+=1
            print('Processed',count)
    with open('./python/nodevoc.txt','w',encoding='utf8') as f:
        f.write(str(nodevoc))
    with open('./python/typevoc.txt','w',encoding='utf8') as f:
        f.write(str(typevoc))


if __name__=='__main__':
    save='./python/'
    for mode in ['train','test','dev']:
        if not os.path.exists(save+mode): os.makedirs(save+mode)
        if not os.path.exists(save+mode): os.makedirs(save+mode)
        if not os.path.exists(save+mode): os.makedirs(save+mode)
    process()