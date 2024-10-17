import torch
import numpy as np

def gen_input(graphs, bkd_gids, nodemax):

    As = {}
    Xs = {}
    Adj = {}
    for gid in bkd_gids:
        Adj[gid] = graphs[gid].edge_mat   
        
        if gid not in As: As[gid] = Adj[gid].clone()
        if gid not in Xs: Xs[gid] = graphs[gid].node_features.clone()
    Ainputs = {}
    Xinputs = {}
    
    for gid in bkd_gids:
        if gid not in Ainputs: Ainputs[gid] = As[gid].clone().detach()
        if gid not in Xinputs: Xinputs[gid] = torch.mm(Ainputs[gid].float(), Xs[gid])
                
    # pad each input into maxi possible size (N, N) / (N, F)

    for gid in Ainputs.keys():
        a_input = Ainputs[gid]
        x_input = Xinputs[gid]
        
        add_dim = nodemax - a_input.shape[0]
        Ainputs[gid] = np.pad(a_input, ((0, add_dim), (0, add_dim))).tolist()
        Xinputs[gid] = np.pad(x_input, ((0, add_dim), (0, 0))).tolist()
        Ainputs[gid] = torch.tensor(Ainputs[gid])
        Xinputs[gid] = torch.tensor(Xinputs[gid])

    return Ainputs, Xinputs
    