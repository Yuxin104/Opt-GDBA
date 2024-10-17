import numpy as np
import torch
import copy

def gen_mask(featdim, nodenums, bkd_gids, choose_gids, bkd_nid_groups):
   
    N = nodenums
    F = featdim
    topomask = {}
    featmask = {}
    
    for i in choose_gids:
        groups = bkd_nid_groups[i]
        if i not in topomask: topomask[i] = torch.zeros(N, N)
        if i not in featmask: featmask[i] = torch.zeros(N, F)
        
        for nid in groups:
            topomask[i][nid][groups] = 1
            topomask[i][nid][nid] = 0
            featmask[i][nid][::] = 1
                
    return topomask, featmask
    
    
def recover_mask(Ni, mask, for_whom):
    
    recovermask = copy.deepcopy(mask)

    if for_whom == 'topo':
        recovermask = recovermask[:Ni, :Ni]
    elif for_whom == 'feat':
        recovermask = recovermask[:Ni]
    
    return recovermask
    