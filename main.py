import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import OptGDBA as OptGDBA
from mask import gen_mask
from input import gen_input
from util import *
from graphcnn import Discriminator
import pickle
import copy

criterion = nn.CrossEntropyLoss()

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def train(args, model, device, train_graphs, optimizer, epoch, tag2index):
    model.train()

    total_iters = args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    #for pos in pbar:
    for pos in range(total_iters):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        
        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        #pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    #print("loss training: %f" % (average_loss))

    return average_loss

def train_G(args, model, generator, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    model.eval()#train()
    generator.train()
    total_iters = 1 #args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    loss_poison_total = 0
    #for pos in pbar:
    #local_feat = torch.zeros(nodemax,nodemax)
    for pos in range(total_iters):
        selected_idx = bkd_gids_train #np.random.permutation(len(train_graphs))[:args.batch_size]
        sub_loss = nn.MSELoss() 
        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        output_graph, trigger_group, edges_len, nodes_len, trigger_id, trigger_l = generator(args, id, train_graphs_trigger, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
        output = model(output_graph)
        output_graph_poison = torch.stack([output[idx] for idx in selected_idx])

        labels_poison = torch.LongTensor([args.target for idx in selected_idx]).to(device)

        loss_poison = criterion(output_graph_poison, labels_poison)

        loss = sub_loss(trigger_id, trigger_l.detach()) #Intermediate Supervision
    average_loss = 0
    return loss, loss_poison, edges_len, nodes_len

def train_D(args, model, generator, id, device, train_graphs_trigger, epoch, tag2index, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, binaryfeat=False):
    model.train()
    generator.eval()#train()
    total_iters = args.iters_per_epoch
    #pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    loss_poison_total = 0
    #for pos in pbar:
    for pos in range(total_iters):
        selected_idx = bkd_gids_train 

        batch_graph = [train_graphs_trigger[idx] for idx in selected_idx]

        output_graph, _, _, _, _, _ = generator(args, id, train_graphs_trigger, bkd_gids_train, Ainput_train, Xinput_train, nodenums_id, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
  
        output = model(output_graph)
        labels = torch.LongTensor([graph.label for graph in output_graph]).to(device)

        # compute loss
        loss = criterion(output, labels) 
        loss_accum += loss

    average_loss = loss_accum / total_iters

    return average_loss
def optimize_D(loss, global_model, optimizer_D):
    global_model.zero_grad() 
    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()
    return
def optimize_G(alpha, loss1, loss2, model, optimizer_G):
    model.zero_grad()
    optimizer_G.zero_grad()
    loss =  alpha * loss1 + loss2 
    loss.backward()
    optimizer_G.step()
    return
###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def test(args, model, device, test_graphs, tag2index):
    model.eval()

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    #print("pred:",pred)

    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    # print(labels)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy test: %f" % acc_test)

    return acc_test

def bkd_cdd_test(graphs, target_label):
    
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label: 
            backdoor_graphs_indexes.append(graph_idx)
        
    return backdoor_graphs_indexes
def bkd_cdd(num_backdoor_train_graphs, graphs, target_label, dataset):
    if dataset == 'MUTAG':
        num_backdoor_train_graphs = 1
    
    temp_n = 0
    backdoor_graphs_indexes = []
    for graph_idx in range(len(graphs)):
        if graphs[graph_idx].label != target_label and temp_n < num_backdoor_train_graphs:
            backdoor_graphs_indexes.append(graph_idx)
            temp_n += 1
    
    return backdoor_graphs_indexes

def init_trigger(args, x, bkd_gids: list, bkd_nid_groups: list, init_feat: float):
    if init_feat == None:
        init_feat = - 1
        print('init feat == None, transferred into -1')

    graphs = copy.deepcopy(x)   
    for idx in bkd_gids:
        
        edges = [list(pair) for pair in graphs[idx].g.edges()]
        edges.extend([[i, j] for j, i in edges])
        
        for i in bkd_nid_groups[idx]:
            for j in bkd_nid_groups[idx]:
                if [i, j] in edges:
                    edges.remove([i, j])
                if (i, j) in graphs[idx].g.edges():
                    graphs[idx].g.remove_edge(i, j)
        edge_mat_temp = torch.zeros(len(graphs[idx].g),len(graphs[idx].g))
        for [x_i,y_i] in edges:
            edge_mat_temp[x_i,y_i] = 1
        graphs[idx].edge_mat = edge_mat_temp
        # change graph labels
        assert args.target is not None
        graphs[idx].label = args.target
        graphs[idx].node_tags = list(dict(graphs[idx].g.degree).values()) 
    
        # change features in-place
        featdim = graphs[idx].node_features.shape[1]
        a = np.array(graphs[idx].node_features)
        a[bkd_nid_groups[idx]] = np.ones((len(bkd_nid_groups[idx]), featdim)) * init_feat
        graphs[idx].node_features = torch.Tensor(a.tolist())
            
    return graphs  
    
def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(
        description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--port', type=str, default="acm4",
                        help='name of sever')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset (default: MUTAG)')
    parser.add_argument('--num_agents', type=int, default=20,
                        help="number of agents:n")
    parser.add_argument('--num_corrupt', type=int, default=4,
                        help="number of corrupt agents")
    parser.add_argument('--frac_epoch', type=float, default=0.5,
                        help='fraction of users are chosen') 
    parser.add_argument('--is_Customized', type=int, default=0,
                        help='is_Customized') 
    parser.add_argument('--is_test', type=int, default=0,
                        help='is_test')           
    parser.add_argument('--is_defense', type=int, default=0,
                        help='is_defense')                                    
    parser.add_argument('--triggersize', type=int, default=4,
                        help='number of nodes in a clique (trigger size)')
    parser.add_argument('--target', type=int, default=0,
                        help='targe class (default: 0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=1,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--n_epoch', type=int, default=1,
                        help='Ratio of training rounds')
    parser.add_argument('--num_backdoor_train_graphs', type=int, default=1,
                        help='Ratio of malicious training data -> number')                    
    parser.add_argument('--n_train_D', type=int, default=1,
                        help='training rounds')
    parser.add_argument('--n_train_G', type=int, default=1,
                        help='training rounds')   
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='coefficient')                                    
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", default=False,
                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')
    parser.add_argument('--degree_as_tag', action="store_true", default=True,
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--topo_thrd', type=float, default=0.5, 
                       help="threshold for topology generator")
    parser.add_argument('--gtn_layernum', type=int, default=3, 
                        help="layer number of GraphTrojanNet")
    parser.add_argument('--topo_activation', type=str, default='sigmoid', 
                        help="activation function for topology generator")
    parser.add_argument('--feat_activation', type=str, default='relu', 
                       help="activation function for feature generator")
    parser.add_argument('--feat_thrd', type=float, default=0, 
                       help="threshold for feature generator (only useful for binary feature)")
    parser.add_argument('--filename', type=str, default="output",
                        help='output file')
    parser.add_argument('--filenamebd', type=str, default="output_bd",
                        help='output backdoor file')
    args = parser.parse_args()

    cpu = torch.device('cpu')
    # set up seeds and gpu device
    torch.manual_seed(0) 
    np.random.seed(0) 
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    graphs, num_classes, tag2index = load_data(args.dataset, args.degree_as_tag)

    train_graphs, test_graphs, test_idx = separate_data(graphs, args.seed, args.fold_idx)

    print('#train_graphs:', len(train_graphs), '#test_graphs:', len(test_graphs))

    print('input dim:', train_graphs[0].node_features.shape[1])
    
    train_data_size = len(train_graphs)
    client_data_size=int(train_data_size/(args.num_agents))
    split_data_size = [client_data_size for i in range(args.num_agents-1)]
    split_data_size.append(train_data_size-client_data_size*(args.num_agents-1))
    train_graphs = torch.utils.data.random_split(train_graphs,split_data_size)
    
    global_model = Discriminator(args.num_layers, args.num_mlp_layers, train_graphs[0][0].node_features.shape[1],
                        args.hidden_dim, \
                        num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type,
                        args.neighbor_pooling_type, device).to(device)
    
    optimizer_D = optim.Adam(global_model.parameters(), lr=args.lr)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.1)

    test_graphs_trigger = copy.deepcopy(test_graphs)
    test_backdoor = bkd_cdd_test(test_graphs_trigger, args.target)

    #nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
    nodenums = [len(graphs[idx].g.adj) for idx in range(len(graphs))]
    nodemax = max(nodenums) 
    #featdim = np.array(self.benign_dr.data['features'][0]).shape[1] 
    featdim = train_graphs[0][0].node_features.shape[1]
    
    generator = {}
    optimizer_G = {}
    scheduler_G = {}
    for g_i in range(args.num_corrupt):
        generator[g_i] = OptGDBA.Generator(nodemax, featdim, args.gtn_layernum, args.triggersize)
        optimizer_G[g_i] = optim.Adam(generator[g_i].parameters(), lr=args.lr)
        scheduler_G[g_i] = optim.lr_scheduler.StepLR(optimizer_G[g_i], step_size=50, gamma=0.1)
    
    # init test data
    # NOTE: for data that can only add perturbation on features, only init the topo value
   
    Ainput_test, Xinput_test = gen_input(test_graphs_trigger, test_backdoor, nodemax) 
    
    with open(args.filenamebd, 'w+') as f:
        f.write("acc_train acc_clean acc_backdoor\n")
        bkd_gids_train = {}
        Ainput_train = {}
        Xinput_train = {}
        nodenums_id = {}
        train_graphs_trigger = {}
        
        for id in range(args.num_corrupt):          
            train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
            nodenums_id[id] = [len(train_graphs_trigger[id][idx].g.adj) for idx in range(len(train_graphs_trigger[id]))]
            bkd_gids_train[id] = bkd_cdd(args.num_backdoor_train_graphs, train_graphs_trigger[id], args.target, args.dataset)
            Ainput_train[id], Xinput_train[id] = gen_input(train_graphs_trigger[id], bkd_gids_train[id], nodemax)
        
        global_weights = global_model.state_dict() 

        for epoch in tqdm(range(1, args.epochs + 1)):
            local_weights, local_losses = [], []
            m = max(int(args.frac_epoch * args.num_agents), 1)
            idxs_users = np.random.choice(range(args.num_agents), m, replace=False)
            print("idxs_users:", idxs_users)

            for id in idxs_users: 
                global_model.load_state_dict(copy.deepcopy(global_weights)) 
                if id < args.num_corrupt: 
                    train_graphs_trigger[id] = copy.deepcopy(train_graphs[id])
                    for kk in range(args.n_train_D):
                        loss = train_D(args, global_model, generator[id], id, device, train_graphs_trigger[id], 
                                        epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                        Xinput_train[id], nodenums_id[id], nodemax, 
                                        binaryfeat=False)
                        optimize_D(loss, global_model, optimizer_D)
                    if epoch % args.n_epoch == 0:
                        for kk in range(args.n_train_G):
                            loss, loss_poison, edges_len, nodes_len = train_G(args, global_model, generator[id], id, device, train_graphs_trigger[id], 
                                            epoch, tag2index, bkd_gids_train[id], Ainput_train[id], 
                                            Xinput_train[id], nodenums_id[id], nodemax, 
                                            binaryfeat=False)
                            optimize_G(args.alpha, loss, loss_poison, generator[id], optimizer_G[id])
                else:
                    loss = train(args, global_model, device, train_graphs[id], optimizer_D, epoch, tag2index)

                l_weights = global_model.state_dict()
                local_weights.append(l_weights)
                local_losses.append(loss)

            scheduler_D.step()     
            global_weights = average_weights(local_weights)   
            global_model.load_state_dict(global_weights)
             
            loss_avg = sum(local_losses) / len(local_losses)    
            
            #----------------- Evaluation -----------------#
            if epoch%5 ==0:
                id = 0
                args.is_test = 1
                test_backdoor0 = copy.deepcopy(test_backdoor)
                generator[id].eval()
                nodenums_test = [len(test_graphs[idx].g.adj) for idx in range(len(test_graphs))]
                bkd_dr_test, bkd_nid_groups_test, _, _, _, _= generator[id](args, id, test_graphs_trigger, test_backdoor0, Ainput_test, Xinput_test, nodenums_test, nodemax, args.is_Customized, args.is_test, args.triggersize, device=torch.device('cpu'), binaryfeat=False)
                for gid in test_backdoor: 
                    for i in bkd_nid_groups_test[gid]:
                        for j in bkd_nid_groups_test[gid]:
                            if i != j:
                                bkd_dr_test[gid].edge_mat[i][j] = 1
                                if (i,j) not in bkd_dr_test[gid].g.edges():
                                    bkd_dr_test[gid].g.add_edge(i, j)
                                                        
                    bkd_dr_test[gid].node_tags = list(dict(bkd_dr_test[gid].g.degree).values())
                args.is_test = 0

                acc_test_clean = test(args, global_model, device, test_graphs, tag2index)
                bkd_dr_ = [bkd_dr_test[idx] for idx in test_backdoor]
                acc_test_backdoor = test(args, global_model, device, bkd_dr_, tag2index)

                f.flush()
            #scheduler.step() 
    f = open('./saved_model/' + str(args.dataset) + '_triggersize_' + str(args.triggersize), 'wb')

    pickle.dump(global_model, f)
    f.close()


if __name__ == '__main__':
    main()