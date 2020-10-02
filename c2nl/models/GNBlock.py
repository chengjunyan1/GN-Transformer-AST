import dgl
import torch as th
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from c2nl.models.GATConv import GATConv
from c2nl.modules.position_ffn import PositionwiseFeedForward
from c2nl.utils.misc import aeq


class EncoderBase(nn.Module):
    def _check_args(self, src, lengths=None, hidden=None):
        n_batch, _, _ = src.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None):
        raise NotImplementedError


# PGNN layer as RPE
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable
        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)
        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def count_parameters(self):
        params = list(self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)
    
    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()
        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                    feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = th.cat((messages, self_feature), dim=-1)
        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d
        out_structure = th.mean(messages, dim=1)  # n*d
        return out_structure

class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))

class PGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                  layer_num=2, dropout=True, **kwargs):
        super(PGNN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1: hidden_dim = output_dim
        self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def count_parameters(self):
        params = list(self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)
    
    def forward(self, x, dists_max, dists_argmax):
        x = self.conv_first(x, dists_max, dists_argmax)
        if self.layer_num == 1: return x
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, dists_max, dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, dists_max, dists_argmax)
        return x


class GALayer(nn.Module):
    def __init__(self, embed_size, d_k, d_v, num_heads, rate=0.2):
        super(GALayer, self).__init__()
        self.num_heads=num_heads
        self.F_=int(embed_size//num_heads)
        self.gat=GATConv(in_feats=embed_size, out_feats=self.F_, d_k=d_k, d_v=d_v,
                    num_heads=num_heads, attn_drop=rate)
        self.dropout = nn.Dropout(rate)
        self.norm = nn.LayerNorm(embed_size)
        
    def forward(self, bg, h):
        return self.norm(self.dropout(self.gat(bg,h,h,h))+h)

class GABlock(nn.Module):
    def __init__(self, embed_size, d_k, d_v, num_heads, num_layers, hus, rate=0.2):
        super(GABlock, self).__init__()
        self.num_layers=num_layers
        self.GALayers=nn.ModuleList([GALayer(embed_size, d_k, d_v, num_heads, rate)
             for i in range(num_layers)])
        self.feed_forward = PositionwiseFeedForward(embed_size, hus, rate)

    def forward(self, bg, h):
        for i in range(self.num_layers):
            h = self.GALayers[i](bg, h)
        return self.feed_forward(h)

class GAEncoder(EncoderBase):
    def __init__(self, 
                 num_blocks,
                 d_model, 
                 heads, 
                 d_k, 
                 d_v, 
                 d_ff,  
                 num_layers=1,
                 dropout=0.2,
                 RPE=False,
                 RPE_mode='sum',
                 RPE_size=128,
                 RPE_layer=2,
                 RPE_share_emb=False,
                 RPE_all=True):
        super(GAEncoder, self).__init__()
        self.num_blocks=num_blocks
        self.GABlocks=nn.ModuleList([GABlock(d_model, d_k, d_v, heads,
            num_layers, d_ff, dropout) for i in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self.RPE=RPE
        self.RPE_mode=RPE_mode
        self.RPE_share_emb=RPE_share_emb
        self.RPE_all=RPE_all
        if RPE: 
            if not RPE_share_emb:
                self.rpe_embeddings=nn.Embedding(1000, d_model)
            if RPE_mode=='sum':
                self.PGNN=PGNN(d_model,RPE_size,d_model,RPE_layer)
            elif RPE_mode=='concat': 
                self.PGNN=PGNN(d_model,RPE_size,RPE_size,RPE_layer)
                self.reduce_dim=nn.Linear(RPE_size+d_model,d_model)
            elif RPE_mode=='abs': pass
        
    def count_parameters(self):
        params = list(self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, bg, h, dm=None, da=None, nids=None, mask=None, seq_length=None): # h: Word first, then nodes!   
        if self.RPE: 
            if not self.RPE_share_emb:
                pe=self.rpe_embeddings(nids)
            else: pe=h
            if self.RPE_mode=='abs': h=h+pe
            else:
                p=self.PGNN(pe,dm,da) 
                if not self.RPE_all: 
                    mask_emb=self.rpe_embeddings(mask)
                    mask=mask.unsqueeze(1)
                    p=p*mask+mask_emb*(1-mask)
                if self.RPE_mode=='sum': h=h+p
                elif self.RPE_mode=='concat': h=self.reduce_dim(th.cat([h,p],1))
        for i in range(self.num_blocks): 
            h = self.GABlocks[i](bg, h)
        if seq_length!=None:
            bg.ndata['hv']=h
            h=th.stack([i.ndata['hv'][:seq_length,:] 
                        for i in dgl.unbatch(bg)])
        return h


if __name__=='__main__':
    
    from prettytable import PrettyTable
    from c2nl.inputters.get_dists import get_dm_da

    def layer_wise_parameters(model):
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in model.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
        return table


    def buildDG(nodes,edges):
        node_map={}
        num_node=len(nodes)
        g = dgl.DGLGraph()
        g.add_nodes(num_node)   
        for i in range(num_node):
            node_map[nodes[i]]=i
        for e in edges:
            g.add_edge(node_map[e[0]],node_map[e[1]])
        return g

    def buildBG(nodes,edges):    
        gs=[]
        for i in range(len(nodes)):
            gs.append(buildDG(nodes[i],edges[i]))
        return dgl.batch(gs)

    embed_size=512
    RPE_size=512
    RPE_layer=2
    seq_length=2
    num_heads=8
    num_block=2
    d_k=64
    d_v=64
    d_ff=2048 
    
    # gat1=GALayer(embed_size, num_heads)
    # gat1=GABlock(embed_size, num_heads, num_layers, hid_units)
    gat1=GAEncoder(num_block,embed_size,num_heads,d_k=d_k,d_v=d_v,d_ff=d_ff,
                   RPE=True,RPE_mode='concat',RPE_size=RPE_size,RPE_layer=RPE_layer)
    
    print(layer_wise_parameters(gat1))
    print(gat1.count_parameters())
    
    gnode=[[1,3,5,9],[1,7]]  
    gedge=[[[3,1],[5,1],[9,1]],[[1,7]]]
    bg=buildBG(gnode,gedge)
    h=th.randn(6,embed_size)
    dm,da=get_dm_da(bg)
    # print(h.shape)
    nids=th.cat([g.nodes() for g in dgl.unbatch(bg)])
    q=gat1(bg,h,dm,da,nids)
    # q=gat1(bg,h)
    # print(q.shape)