import networkx as nx
import random,torch
import numpy as np


# PGNN RPE

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    dists_dict=single_source_shortest_path_length_range(graph, nodes, cutoff)
    return dists_dict

def precompute_dist_data(edge_index, num_nodes, approximate=0):
    graph = nx.Graph()
    edge_list = edge_index.transpose(1,0).tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist!=-1: dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array

def get_random_anchorset(n,m=0,c=0.5):
    m = int(np.log2(n)) if m==0 else m
    m = 1 if m<1 else m
    copy = int(c*m)
    copy = 1 if copy<1 else copy
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        anchor_size=1 if anchor_size<1 else anchor_size
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id)))
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long()
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = dist_argmax_temp
    return dist_max, dist_argmax

def preselect_anchor(num_nodes, dists, m, c):
    anchorset_id = get_random_anchorset(num_nodes,m,c)
    dists_max, dists_argmax = get_dist_max(anchorset_id, dists)
    return dists_max,dists_argmax

def get_dm_da(g,m=0,c=0.2,approx=0):
    eg=g.edges()
    eg=np.array([eg[0].numpy(),eg[1].numpy()])
    vg=list(g.nodes())
    dists = precompute_dist_data(eg, len(vg), approximate=approx)
    dists = torch.from_numpy(dists).float()
    return preselect_anchor(len(vg),dists,m,c)


if __name__=='__main__':
    # PGNN
    # from model import PGNN_layer
    import dgl

    dg = dgl.DGLGraph()
    dg.add_nodes(7)    
    for i in [[0,1],[0,3],[3,5],[1,2],[1,4],[3,6]]:
        dg.add_edges(i[0],i[1])
        dg.add_edges(i[1],i[0])
    
    # dm,da=get_dm_da(dg)
    # h=torch.randn(7,512)
    
    # M=PGNN_layer(512,512,dist_trainable=False)
    # q,w=M(h,dm,da)