import torch
import numpy as np
import dgl
from queue import Queue
from c2nl.inputters.get_dists import get_dm_da


def vectorize(ex, model):
    """Vectorize a single example."""
    src_dict = model.src_dict
    tgt_dict = model.tgt_dict

    code, summary, syntax = ex['code'], ex['summary'], ex['syntax']
    vectorized_ex = dict()
    vectorized_ex['id'] = code.id
    vectorized_ex['language'] = code.language

    vectorized_ex['code'] = code.text
    vectorized_ex['code_tokens'] = code.tokens
    vectorized_ex['code_word_rep'] = torch.LongTensor(code.vectorize(word_dict=src_dict))

    vectorized_ex['summ'] = None
    vectorized_ex['summ_tokens'] = None
    vectorized_ex['stype'] = None
    vectorized_ex['summ_word_rep'] = None
    vectorized_ex['summ_char_rep'] = None
    vectorized_ex['target'] = None

    if summary is not None:
        vectorized_ex['summ'] = summary.text
        vectorized_ex['summ_tokens'] = summary.tokens
        vectorized_ex['stype'] = summary.type
        vectorized_ex['summ_word_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict))
        if model.args.use_tgt_char:
            vectorized_ex['summ_char_rep'] = torch.LongTensor(summary.vectorize(word_dict=tgt_dict, _type='char'))
        # target is only used to compute loss during training
        vectorized_ex['target'] = torch.LongTensor(summary.vectorize(tgt_dict))

    vectorized_ex['src_vocab'] = code.src_vocab
    vectorized_ex['use_tgt_word'] = model.args.use_tgt_word
    vectorized_ex['use_tgt_char'] = model.args.use_tgt_char
    vectorized_ex['use_word_type'] = model.args.use_word_type
    vectorized_ex['use_word_fc'] = model.args.use_word_fc
    vectorized_ex['src_vocab_size'] = model.args.src_vocab_size

    try:
        vectorized_ex['use_dense_connection']=model.args.use_dense_connection
        vectorized_ex['add_top_down_edges'] = model.args.add_top_down_edges
        vectorized_ex['add_bottom_up_edges'] = model.args.add_bottom_up_edges
        vectorized_ex['use_rpe'] = model.args.use_rpe
        vectorized_ex['rpe_c'] = model.args.rpe_c
        vectorized_ex['rpe_m'] = model.args.rpe_m
        vectorized_ex['rpe_approx'] = model.args.rpe_approx
        vectorized_ex['rpe_mode'] = model.args.rpe_mode
    except:
        vectorized_ex['use_dense_connection']=False
        vectorized_ex['add_top_down_edges'] = True
        vectorized_ex['add_bottom_up_edges'] = True
        vectorized_ex['use_rpe'] = False
        vectorized_ex['rpe_c'] = None
        vectorized_ex['rpe_m'] = None
        vectorized_ex['rpe_approx'] = None
        vectorized_ex['rpe_mode'] = None
    try: vectorized_ex['rpe_all'] = model.args.rpe_all
    except: vectorized_ex['rpe_all']=True


    vectorized_ex['wtype'] = None
    vectorized_ex['wparents']=None
    if model.args.use_word_type:
        vectorized_ex['wtype'] = torch.LongTensor(syntax[0])
        assert vectorized_ex['wtype'].shape == vectorized_ex['code_word_rep'].shape
    vectorized_ex['gnode'] = syntax[1]
    vectorized_ex['gedge'] = syntax[2]
    vectorized_ex['gtype'] = syntax[3]
    vectorized_ex['ginit'] = syntax[4]

    if not vectorized_ex['use_dense_connection']:
        gnode=syntax[1]
        gedge=syntax[2]
        ginit=syntax[4]
        ast_parent={}
        ast_children={}
        for e in gedge: 
            if e[0] not in ast_children: ast_children[e[0]]=[]
            ast_parent[e[1]]=e[0]
            ast_children[e[0]].append(e[1])
        ast_root=[]
        for i in gnode:
            if i not in ast_parent: 
                ast_root.append(i)
        ast_layers={}
        Q=Queue()
        for i in gnode: 
            ast_layers[i]=0
            Q.put(i)
        while not Q.empty():
            cur=Q.get()
            if cur in ast_root: continue
            par=ast_parent[cur]
            if ast_layers[par]<ast_layers[cur]+1: ast_layers[par]=ast_layers[cur]+1
            Q.put(par)
        wparents={}
        for n in range(len(ginit)): 
            for w in ginit[n]:
                node=gnode[n]
                if w not in wparents: wparents[w]=node
                if ast_layers[node]<ast_layers[wparents[w]]: wparents[w]=node
        vectorized_ex['wparents'] = wparents

    return vectorized_ex


def batchify(batch):
    """Gather a batch of individual examples into one batch."""

    # batch is a list of vectorized examples
    batch_size = len(batch)
    use_tgt_word = batch[0]['use_tgt_word']
    use_tgt_char = batch[0]['use_tgt_char']
    use_word_type = batch[0]['use_word_type']
    use_word_fc = batch[0]['use_word_fc']
    src_vocab_size = batch[0]['src_vocab_size']
    use_dense_connection=batch[0]['use_dense_connection']
    add_top_down_edges=batch[0]['add_top_down_edges']
    add_bottom_up_edges=batch[0]['add_bottom_up_edges']
    use_rpe=batch[0]['use_rpe']
    rpe_all=batch[0]['rpe_all']
    rpe_c=batch[0]['rpe_c']
    rpe_m=batch[0]['rpe_m']
    rpe_approx=batch[0]['rpe_approx']
    rpe_mode=batch[0]['rpe_mode']
    ids = [ex['id'] for ex in batch]
    language = [ex['language'] for ex in batch]

    # --------- Prepare Code tensors ---------
    code_words = [ex['code_word_rep'] for ex in batch]
    max_code_len = max([d.size(0) for d in code_words])

    # Batch Code Representations
    code_len_rep = torch.zeros(batch_size, dtype=torch.long)
    code_word_rep = torch.zeros(batch_size, max_code_len, dtype=torch.long)

    source_maps = []
    src_vocabs = []
    for i in range(batch_size):
        code_len_rep[i] = code_words[i].size(0)
        code_word_rep[i, :code_words[i].size(0)].copy_(code_words[i])
        context = batch[i]['code_tokens']
        vocab = batch[i]['src_vocab']
        src_vocabs.append(vocab)
        # Mapping source tokens to indices in the dynamic dict.
        src_map = torch.LongTensor([vocab[w] for w in context])
        source_maps.append(src_map)

    # --------- Prepare Summary tensors ---------
    no_summary = batch[0]['summ_word_rep'] is None
    if no_summary:
        summ_len_rep = None
        summ_word_rep = None
        summ_char_rep = None
        tgt_tensor = None
        alignments = None
    else:
        summ_words = [ex['summ_word_rep'] for ex in batch]
        summ_chars = [ex['summ_char_rep'] for ex in batch]
        max_sum_len = max([q.size(0) for q in summ_words])
        if use_tgt_char:
            max_char_in_summ_token = summ_chars[0].size(1)

        summ_len_rep = torch.zeros(batch_size, dtype=torch.long)
        summ_word_rep = torch.zeros(batch_size, max_sum_len, dtype=torch.long) \
            if use_tgt_word else None
        summ_char_rep = torch.zeros(batch_size, max_sum_len, max_char_in_summ_token, dtype=torch.long) \
            if use_tgt_char else None

        max_tgt_length = max([ex['target'].size(0) for ex in batch])
        tgt_tensor = torch.zeros(batch_size, max_tgt_length, dtype=torch.long)
        alignments = []
        for i in range(batch_size):
            summ_len_rep[i] = summ_words[i].size(0)
            if use_tgt_word:
                summ_word_rep[i, :summ_words[i].size(0)].copy_(summ_words[i])
            if use_tgt_char:
                summ_char_rep[i, :summ_chars[i].size(0), :].copy_(summ_chars[i])
            #
            tgt_len = batch[i]['target'].size(0)
            tgt_tensor[i, :tgt_len].copy_(batch[i]['target'])
            target = batch[i]['summ_tokens']
            align_mask = torch.LongTensor([src_vocabs[i][w] for w in target])
            alignments.append(align_mask)

    # --------- Prepare Syntax tensors ---------
    gs=[]
    astok=[]
    wtypes=[]
    wpos=[]
    dm,da,nids=None,None,None
    rpe_mask=None
    nnode=0
    for ex in batch:
        words = ex['code_word_rep'].tolist()
        wtype = ex['wtype']
        gnode = ex['gnode']
        gedge = ex['gedge']
        gtype = ex['gtype']
        ginit = ex['ginit']
        wparents = ex['wparents']
        edges_u=[]
        edges_v=[]
        coord={}
        num_nodes=max_code_len+len(gnode)
        if use_word_type:
            wtypes+=wtype
            for i in range(num_nodes-len(wtype)): wtypes.append(0)
        astok+=words
        for i in range(max_code_len-len(words)): astok.append(0)
        astok+=(np.array(gtype)+src_vocab_size).tolist()
        wpos+=np.arange(1,len(words)+1).tolist()
        for i in range(num_nodes-len(words)): wpos.append(0)
        for i in range(len(gnode)): coord[gnode[i]]=i+max_code_len
        for e in gedge: 
            if add_top_down_edges:
                edges_u.append(coord[e[0]])
                edges_v.append(coord[e[1]])
            if add_bottom_up_edges:
                edges_u.append(coord[e[1]])
                edges_v.append(coord[e[0]])
        if use_dense_connection:
            for n in range(len(ginit)): 
                for w in ginit[n]:
                    parent_w=coord[gnode[n]]
                    edges_u.append(parent_w)
                    edges_v.append(w)
                    edges_u.append(w)
                    edges_v.append(parent_w)
        else:
            for w in wparents: 
                parent_w=coord[wparents[w]]
                edges_u.append(parent_w)
                edges_v.append(w)
                edges_u.append(w)
                edges_v.append(parent_w)
        if use_word_fc:
            for i in range(len(words)):
                for j in range(len(words)):
                    if i!=j: 
                        edges_u.append(i)
                        edges_v.append(j)
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)    
        g.add_edges(edges_u,edges_v)
        gs.append(g)
        if use_rpe: 
            if rpe_all: nidi=g.nodes()
            else:
                nidi=g.nodes()+1
                if rpe_mask==None: rpe_mask=list(np.ones(len(words)))+[0]*(len(gnode)+max_code_len-len(words))
                else: rpe_mask+=list(np.ones(len(words)))+[0]*(len(gnode)+max_code_len-len(words))
            if nids==None: nids=nidi
            else: nids=torch.cat([nids,nidi])
            if rpe_mode!='abs':
                dmi,dai=get_dm_da(g,rpe_m,rpe_c,rpe_approx)
                if dm==None: dm=dmi 
                else: dm=torch.cat([dm,dmi])
                if da==None: da=dai 
                else: da=torch.cat([da,dai+nnode])
                nnode+=num_nodes
    bg = dgl.batch(gs)    
    wtype_rep=torch.LongTensor([wtypes]) if use_word_type else None
    astok=torch.LongTensor([astok])
    wpos=torch.LongTensor([wpos])
    if rpe_mask!=None: rpe_mask=torch.LongTensor(rpe_mask)

    return {
        'ids': ids,
        'language': language,
        'batch_size': batch_size,
        'code_word_rep': code_word_rep,
        'code_len': code_len_rep,
        'summ_word_rep': summ_word_rep,
        'summ_char_rep': summ_char_rep,
        'summ_len': summ_len_rep,
        'tgt_seq': tgt_tensor,
        'code_text': [ex['code'] for ex in batch],
        'code_tokens': [ex['code_tokens'] for ex in batch],
        'summ_text': [ex['summ'] for ex in batch],
        'summ_tokens': [ex['summ_tokens'] for ex in batch],
        'src_vocab': src_vocabs,
        'src_map': source_maps,
        'alignment': alignments,
        'stype': [ex['stype'] for ex in batch],
        'wtype': wtype_rep,
        'wpos': wpos,
        'astok': astok,
        'bg':bg,
        'dm':dm,
        'da':da,
        'nids':nids,
        'rpe_mask':rpe_mask,
        'seqlen':max_code_len
    }
