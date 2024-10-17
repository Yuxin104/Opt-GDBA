import sys, os
sys.path.append(os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from mask import gen_mask
from input import gen_input
from main import init_trigger

from sklearn.cluster import KMeans

import networkx as nx
import random
import hashlib


def detection1(score):
    score = score.numpy()
    nrefs = 10
    ks = range(1, 8)
    if len(score) < 8:
        ks = range(1, len(score))
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    #print(gapDiff)
    select_k = 3
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    return select_k
def detection(score, k_value):
    score = score.numpy()
    estimator = KMeans(n_clusters=k_value)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    trigger_size = {}
    temp_max = 0
    temp_size = 0
    for i in range(k_value):
        trigger_size[i] = np.mean(score[label_pred==i])

    for i in range(k_value):
        if trigger_size[i] > temp_max:
            temp_max = trigger_size[i]
            temp_size = len(label_pred==i)
    return  int(temp_size)       
    

def trigger_top(rank_value, rank_id, trigger_size, number_id):
    local_id = []
    if number_id < trigger_size:
        trigger_size = number_id
    for i in range(int(trigger_size)):
        local_id.append(rank_id[i,0].tolist())
    return local_id

def trigger_top_c(rank_value, rank_id):
    k = detection1(rank_value)
    if k == 1:
        trigger_size = 3
    else:
        trigger_size = detection(rank_value, k)
        if trigger_size > 5:
            trigger_size = 5
        elif trigger_size < 3:
            trigger_size = 3
    
    local_id = []
    for i in range(trigger_size):
        local_id.append(rank_id[i,0].tolist())
    return local_id

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>=thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None
    
class Generator(nn.Module):
    def __init__(self, sq_dim, feat_dim, layernum, trigger_size, dropout=0.05):
        super(Generator, self).__init__()
        layers = []
        layers_feat = []
        view = []
        view_feat = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(sq_dim, sq_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sq_dim, sq_dim))
        
        if dropout > 0:
            layers_feat.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers_feat.append(nn.Linear(feat_dim, feat_dim))
            layers_feat.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers_feat.append(nn.Dropout(p=dropout))
        layers_feat.append(nn.Linear(feat_dim, feat_dim))

        if dropout > 0:
            view.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            view.append(nn.Linear(sq_dim, sq_dim))
            view.append(nn.ReLU(inplace=True))
            if dropout > 0:
                view.append(nn.Dropout(p=dropout))
        view.append(nn.Linear(sq_dim, sq_dim))

        if dropout > 0:
            view_feat.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            view_feat.append(nn.Linear(feat_dim, feat_dim))
            view_feat.append(nn.ReLU(inplace=True))
            if dropout > 0:
                view_feat.append(nn.Dropout(p=dropout))
        view_feat.append(nn.Linear(feat_dim, feat_dim))
        
        self.sq_dim = sq_dim
        self.feat_dim = feat_dim
        self.trigger_size = trigger_size
        self.layers = nn.Sequential(*layers)
        self.layers_feat = nn.Sequential(*layers_feat)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool_feat = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Linear(1, sq_dim*sq_dim)
        self.mlp_feat = nn.Linear(1, sq_dim*feat_dim)
        self.view = nn.Sequential(*view)
        self.view_feat = nn.Sequential(*view_feat)
        #self.mlp_pool = nn.AdaptiveAvgPool1d(1)
               
    def forward(self, args, id, graphs_train, bkd_gids_train, Ainput, Xinput, nodenums_id, 
                nodemax, is_Customized , is_test , trigger_size , device=torch.device('cpu'), binaryfeat=False): 
        bkd_nid_groups = {}
        GW = GradWhere.apply

        graphs = copy.deepcopy(graphs_train)
        nodes_len = 0
        for gid in bkd_gids_train:#tqdm(bkd_gids_train):
            rst_bkdA_backbone = self.view(Ainput[gid])
            if args.topo_activation=='relu':
                rst_bkdA_backbone = F.relu(rst_bkdA_backbone)
            elif args.topo_activation=='sigmoid':
                rst_bkdA_backbone = torch.sigmoid(rst_bkdA_backbone)    # nn.Functional.sigmoid is deprecated
            rst_bkdA_backbone = self.avg_pool(rst_bkdA_backbone)   
            
            rst_bkdX_backbone = self.view_feat(Xinput[gid])
            if args.feat_activation=='relu':
                rst_bkdX_backbone = F.relu(rst_bkdX_backbone)
            elif args.feat_activation=='sigmoid':
                rst_bkdX_backbone = torch.sigmoid(rst_bkdX_backbone)    
            rst_bkdX_backbone = self.avg_pool_feat(rst_bkdX_backbone)

            trigger_id = torch.mul(rst_bkdA_backbone[:nodenums_id[gid]],
                                 rst_bkdX_backbone[:nodenums_id[gid]])

            trigger_l = GW(trigger_id, torch.mean(trigger_id), device)
            rank_value, rank_id = torch.sort(trigger_id, dim=0, descending=True)
            
            bkd_nid_groups[gid] = trigger_top(rank_value, rank_id, self.trigger_size,nodenums_id[gid]) 
        init_dr = init_trigger(
                        args, graphs, bkd_gids_train, bkd_nid_groups, 0.0)
        bkd_dr = copy.deepcopy(init_dr)
        topomask, featmask = gen_mask(
                        graphs[0].node_features.shape[1], nodemax, bkd_dr, bkd_gids_train, bkd_nid_groups)
        Ainput_trigger, Xinput_trigger = gen_input(init_dr, bkd_gids_train, nodemax)

        id = torch.as_tensor(float(id)).unsqueeze(0)
        id_output = self.mlp(id)
        id_output = id_output.reshape(self.sq_dim,self.sq_dim)

        id_output_feat = self.mlp_feat(id)
        id_output_feat = id_output_feat.reshape(self.sq_dim,self.feat_dim)
        for gid in bkd_gids_train:
            Ainput_trigger[gid] = Ainput_trigger[gid] * id_output
            rst_bkdA = self.layers(Ainput_trigger[gid])
            if args.topo_activation=='relu':
                rst_bkdA = F.relu(rst_bkdA)
            elif args.topo_activation=='sigmoid':
                rst_bkdA = torch.sigmoid(rst_bkdA)    # nn.Functional.sigmoid is deprecated

            for_whom='topo'
            if for_whom == 'topo':  
                rst_bkdA = torch.div(torch.add(rst_bkdA, rst_bkdA.transpose(0, 1)), 2.0)
            if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
                rst_bkdA = GW(rst_bkdA, args.topo_thrd, device)
            rst_bkdA = torch.mul(rst_bkdA, topomask[gid]) 

            bkd_dr[gid].edge_mat = torch.add(init_dr[gid].edge_mat, rst_bkdA[:nodenums_id[gid], :nodenums_id[gid]]) 
            for i in range(nodenums_id[gid]):
                for j in range(nodenums_id[gid]):
                    if rst_bkdA[i][j] == 1 and i < j:
                        bkd_dr[gid].g.add_edge(i, j)
            bkd_dr[gid].node_tags = list(dict(bkd_dr[gid].g.degree).values())
         
            for_whom='feat'
            Xinput_trigger[gid] = Xinput_trigger[gid]*id_output_feat
            rst_bkdX = self.layers_feat(Xinput_trigger[gid])
            if args.feat_activation=='relu':
                rst_bkdX = F.relu(rst_bkdX)
            elif args.feat_activation=='sigmoid':
                rst_bkdX = torch.sigmoid(rst_bkdX)
                
            if for_whom == 'topo': # not consider direct yet
                rst_bkdX = torch.div(torch.add(rst_bkdX, rst_bkdX.transpose(0, 1)), 2.0)
            # binaryfeat = True
            if for_whom == 'topo' or (for_whom == 'feat' and binaryfeat):
                rst_bkdX = GW(rst_bkdX, args.feat_thrd, device)
            rst_bkdX = torch.mul(rst_bkdX, featmask[gid])
            
            bkd_dr[gid].node_features = torch.add( 
                    rst_bkdX[:nodenums_id[gid]].detach().cpu(), torch.Tensor(init_dr[gid].node_features)) 
            
        edges_len_avg = 0
        return bkd_dr, bkd_nid_groups, edges_len_avg, self.trigger_size, trigger_id, trigger_l
    
def SendtoCUDA(gid, items):
    """
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    cuda = torch.device('cuda')
    for item in items:
        item[gid] = torch.as_tensor(item[gid], dtype=torch.float32).to(cuda)
        
        
def SendtoCPU(gid, items):
    """
    Used after SendtoCUDA, target object must be torch.tensor and already in cuda.
    
    - items: a list of dict / full-graphs list, 
             used as item[gid] in items
    - gid: int
    """
    
    cpu = torch.device('cpu')
    for item in items:
        item[gid] = item[gid].to(cpu)