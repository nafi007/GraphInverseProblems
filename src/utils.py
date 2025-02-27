import os
import torch
from torch_geometric.datasets.gnn_benchmark_dataset import GNNBenchmarkDataset
from torch_geometric_temporal.signal import temporal_signal_split

from customMETRLA import METRLADatasetLoader
from pygt_dataloader import DataLoader as BatchDataLoader
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from graphForwardOps import graph_smooth, graphMask, graphPath, graph_edgeRecovery
import networks
from customCPOX import ChickenpoxDatasetLoader
import math 
import random 
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant


def TSVD_recovery(A, data):
     U,S,V =  torch.linalg.svd(A)
     Z = U.t() @ data
     S_inv = S / (S**2 + 1e-3) 
     Z = Z * S_inv.unsqueeze(1) 
     out = V @ Z
     return out 

    
def get_experiment_name(args, time_):
    if args.task == 'mask':
            if args.classify == 1:
                # classification task
                exp_name = args.task + '_mask_per_class_budget_' + str(args.mask_per_class_budget) + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
                args.channels) + '_cglsIter_' + str(
                args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_
            else:
                # regression task
                exp_name = args.task + '_mask_per_snapshot_budget_' + str(args.mask_per_snapshot_budget) + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
                args.channels) + '_cglsIter_' + str(
                args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_

    elif args.task == 'deblur':
            exp_name = args.task + '_blurCount_' + args.blur_count + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
            args.channels) + '_cglsIter_' + str(
            args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_

    elif args.task == 'path':
            exp_name = args.task + '_pathLength_' + str(args.pathLength) + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
            args.channels) + '_cglsIter_' + str(
            args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_
         
    else:
        exp_name = args.task + '_' + args.dataset + '_' + args.regnet + '_layers_' + str(args.layers) + '_chan_' + str(
            args.channels) + '_cglsIter_' + str(
            args.cglsIter) + '_netIter_' + str(args.solveIter) + "_" + time_
    return exp_name

def get_forward_op(args, hid_channels, label_channels, device):

    if args.method == "tikhonov_regularization" or args.method == 'laplacian_regularization' or args.method == 'laplacian_explicit':
        learn_embedding = False
    else:
        learn_embedding = True

    if args.task == 'deblur':

        forward_op = graph_smooth(nin=label_channels, embdsize=hid_channels, learnEmb=learn_embedding, device=device, k=int(args.blur_count))
    elif args.task == 'mask':

        forward_op = graphMask(nin=label_channels, ind=torch.arange(25),
                        embdsize=hid_channels, learnEmb=learn_embedding, device=device)
    elif args.task == 'path':

        forward_op = graphPath(embdsize=hid_channels, nin=label_channels, learnEmb=learn_embedding, device=device,
                        pathLength=args.pathLength)
    elif args.task == 'edgeRecovery':

        forward_op = graph_edgeRecovery(nin=label_channels, embdsize=hid_channels, learnEmb=learn_embedding, K=3, device=device)

    return forward_op

def get_network(args, forward_op, hid_channels, label_channels, feat_channels, device):
    
    # feat_channels = number of dimensions of the feature vector for a sample
    # label_channels = number of dimensions of the target vector for a sample
    if args.method == 'drip':
        if args.regnet == 'LA':
            reg_model = networks.graphLeastActionNet(nlayers=args.layers, nchanels=hid_channels, nfixPointIter=2, imsize=10).to(
                device)  
        elif args.regnet == 'hyper':
            reg_model = networks.graphHyperResNet(num_layers=args.layers, nopen=hid_channels, nfeatures=hid_channels,
                                        dropout=args.dropout)
            if args.task == 'edgeRecovery':
                reg_model = networks.graphHyperResNet(num_layers=args.layers, nopen=hid_channels, nfeatures=hid_channels,
                                            dropout=args.dropout)

        if args.task == 'edgeRecovery':
            proj_model = networks.edge_recovery_proj(forward_op, mu=args.mu)
        else:
            proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)

        num_params_reg_model = count_trainable_parameters(reg_model)
        num_params_proj_model = count_trainable_parameters(proj_model)

        net = networks.graph_inverseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
                                            input_feat_dim=feat_channels, rnfPE=args.rnfPE,
                                            task=args.task, learn_emb=True)
        num_params_total_net = count_trainable_parameters(net)

    elif args.method == 'pgd':
        args.cglsIter = 1
        reg_model = networks.graphResNetFO(num_layers=args.layers, nopen=hid_channels, nfeatures=label_channels,
                                dropout=args.dropout)
        proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)

        num_params_reg_model = count_trainable_parameters(reg_model)
        num_params_proj_model = count_trainable_parameters(proj_model)
        net = networks.graph_NeuralProximalGradient(reg_model, proj_model, forward_op, niter=args.solveIter,
                                                    input_feat_dim=feat_channels, rnfPE=args.rnfPE)
        num_params_total_net = count_trainable_parameters(net)
    
    elif args.method == "tikhonov_regularization":
        #  reg_model = None
        #  proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)
        #  net = networks.graph_inverseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
        #                                     input_feat_dim=feat_channels, rnfPE=args.rnfPE,
        #                                     task=args.task, learn_emb=False)
        net = networks.Laplace_noReg_Net(forward_op, args, reg='tikhonov_regularization', device=device)
        num_params_proj_model = count_trainable_parameters(net)
        num_params_reg_model = num_params_proj_model
        num_params_total_net = num_params_proj_model
    elif args.method == 'laplacian_regularization':
            
        net = networks.Laplace_noReg_Net(forward_op, args, reg='laplacian_regularization', device=device)
        num_params_proj_model = count_trainable_parameters(net)
        num_params_reg_model = num_params_proj_model
        num_params_total_net = num_params_proj_model
    elif args.method == 'inv_scale_space':
       
        reg_model = networks.graphScaleSpaceNet(num_layers=args.layers, nopen=hid_channels, nfeatures=hid_channels,
                                        dropout=args.dropout)
        proj_model = networks.graph_CGLS(forward_op, CGLSit=args.cglsIter, eps=1e-5)
        num_params_reg_model = count_trainable_parameters(reg_model)
        num_params_proj_model = count_trainable_parameters(proj_model)
        net = networks.graph_inverseSolveNet(reg_model, proj_model, forward_op, niter=args.solveIter,
                                            input_feat_dim=feat_channels, rnfPE=args.rnfPE,
                                            task=args.task, learn_emb=True)
        num_params_total_net = count_trainable_parameters(net)

    elif args.method == 'laplacian_explicit':
        net = networks.Laplace_noReg_Net(forward_op, args, reg='laplacian_explicit', device=device)
        num_params_proj_model = count_trainable_parameters(net)
        num_params_reg_model = num_params_proj_model
        num_params_total_net = num_params_proj_model

    else:
        print("Error! Your args.method is not a valid choice!")
    
    print(f"Number of parameters in regnet = {num_params_reg_model}")
    print(f"Number of parameters in proj_model = {num_params_proj_model}")
    print(f"Number of parameters in total net = {num_params_total_net}")
    return net


def get_fractional_dataset(dataset, fraction):
    num_samples = len(list(dataset))
    num_samples_to_use = math.ceil(fraction * num_samples)
    
    # Randomly select num_samples_to_use samples from the dataset
    selected_indices = random.sample(range(num_samples), num_samples_to_use)
    
    # Create a subset of the dataset using the selected indices
    fractional_dataset = [dataset[i] for i in selected_indices]
    
    return fractional_dataset


def get_data_and_loaders(args):
    if args.dataset in ['CLUSTER', 'PATTERN']:
        train_dataset = GNNBenchmarkDataset(root=args.datapath, name=args.dataset, split='train')
        test_dataset = GNNBenchmarkDataset(root=args.datapath, name=args.dataset, split='test')
        # Get only the specified fraction of the train dataset
        # train_dataset = get_fractional_dataset(train_dataset, args.train_frac)
        
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
        label_channels = test_dataset.num_classes if args.classify else 1  #
        feat_channels = test_dataset.num_features

    elif args.dataset == 'CPOX':
        lags = args.CPOX_lags
        loader = ChickenpoxDatasetLoader()
        dataset = loader.get_dataset(lags=lags)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.9)
        # Get only the specified fraction of the train dataset
        # train_dataset = get_fractional_dataset(train_dataset, args.train_frac)
        
        train_loader = DataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        test_loader = DataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        label_channels = 1 #lags #1
        feat_channels =  1 #just a 1 dimensional target (number of cases) #lags  #1 

    elif 'METRLA' in args.dataset:
        datapath = os.path.join(args.datapath, 'temporal_data')
        datapath = os.path.join(datapath, args.dataset)
        loader = METRLADatasetLoader(raw_data_dir=datapath)

        # METRLADatasetLoader
        # dataset = loader.get_dataset(num_timesteps_in=1, num_timesteps_out=0)  # args.pred #1 time step for training
        dataset = loader.get_dataset(num_timesteps_in=1, num_timesteps_out=0) # 3 time steps for training
        # train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
        number_of_train_snapshots = int(0.7 * dataset.snapshot_count)
        number_of_val_snapshots = int(0.8 * dataset.snapshot_count)

        # batch_of_train_snapshots = args.train_batch_size  #5   #int(0.01 * dataset.snapshot_count) #1
        # val_snapshots = int(0.2 * dataset.snapshot_count)

        train_dataset = dataset[0:number_of_train_snapshots] 
        # Get only the specified fraction of the train dataset
        train_dataset = get_fractional_dataset(train_dataset, args.train_frac)
        # val_dataset = dataset[number_of_train_snapshots:number_of_val_snapshots]
        test_dataset = dataset[number_of_val_snapshots:]
        test_dataset = get_fractional_dataset(test_dataset, args.test_frac)
        train_loader = BatchDataLoader(list(train_dataset), batch_size=args.train_batch_size, shuffle=True)
        test_loader = BatchDataLoader(list(test_dataset), batch_size=args.test_batch_size, shuffle=False)
        # val_loader = BatchDataLoader(list(val_dataset), batch_size=args.batch_size, shuffle=False)
        label_channels = dataset.num_classes if args.classify else 1  #
        feat_channels = 20

    elif 'SHAPENET' in args.dataset:
        category = None  # Pass in `None` to train on all categories.
        path = args.datapath
        path = path + 'ShapeNet'
        fixed_points_transform = T.FixedPoints(1024, replace=False)
        #################################
        transform = T.Compose([
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
            fixed_points_transform,
            T.NormalizeScale(),
            T.KNNGraph(k=10,num_workers=16)
            ])
        pre_transform = None
        train_dataset = ShapeNet(path, category, split='trainval', transform=transform,
                                pre_transform=pre_transform)
        test_dataset = ShapeNet(path, category, split='test', transform=transform,
                                pre_transform=pre_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                num_workers=6)
        test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False,
                                num_workers=6)
        label_channels = 50 #1
        feat_channels  = 6+16  #normal vectors (3), position vectors (3), and one hot encoded categories(16)

    return train_dataset, test_dataset, train_loader, test_loader, label_channels, feat_channels

def process_graph_for_shapeNet(args, graph, num_categories):
    if args.use_meta_data==0:
        # Create a tensor of ones with shape [num_nodes, 22]
        graph.x = torch.ones((graph.num_nodes, 22), dtype=torch.float)
        
        # Remove the pos attribute
        if hasattr(graph, 'pos'):
            del graph.pos
        return graph

    # Step 1: Concatenate pos (xyz positions) to x (normal vectors) 
    x = torch.cat([graph.x, graph.pos], dim=1)  # Shape: [num_nodes, 6]
    
    # Step 2: One-hot encode the categories
    category_one_hot = F.one_hot(graph.category, num_classes=num_categories).float()  # Shape: [num_graphs, 16]
    
    # Step 3: Use the batch attribute to map each node to its corresponding category
    category_one_hot_expanded = category_one_hot[graph.batch]  # Shape: [num_nodes, 16]
    
    # Step 4: Concatenate the expanded one-hot encoding to each node's 6D vector
    x = torch.cat([x, category_one_hot_expanded], dim=1)  # Shape: [num_nodes, 22]
    
    # Update the graph's x attribute
    graph.x = x

    # Remove the pos attribute (xyz positions) since it's already been concatenated into graph.x 
    if hasattr(graph, 'pos'):
        del graph.pos
    
    return graph

def sparse_adj_to_edge_index_weight(adj_t):
    # Extract row and col from the sparse adjacency matrix
    row, col, val = adj_t.coo()  # .coo() returns row (source), col (destination) and values of edges for SparseTensor

    # Stack row and col to create edge_index
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = val
    return edge_index, edge_weight

def process_data(args, graph):
    if args.use_meta_data==0:
        constant_transform = Constant(value=1.0, cat=False)
    if 'CPOX' in args.dataset:

        graph.y = graph.x.clone()[:,0] # target are cpox cases
        graph.x = graph.x.clone()[:,1] # features are time indices (weeks), so 1 dimensional feature vector per node
        if len(graph.x.shape) == 1:
            graph.x = graph.x.unsqueeze(-1)
        if len(graph.y.shape) == 1:
            graph.y = graph.y.unsqueeze(-1) 
        if args.use_meta_data == 0:
            graph = constant_transform(graph)

    if 'METRLA' in args.dataset:
        graph.x = graph.x.squeeze()
        # now take only data (speed) dimension (and not the time encoding dimensions) below
        graph.y = graph.x.clone()[:, 0].unsqueeze(-1) 
        graph.x = graph.x[:, 1:]  #Each row has a time encoding. All rows at time "t" have the same value 
        if args.use_meta_data == 0:
            graph.x = torch.full_like(graph.x, 1.0)
    if 'SHAPENET' in args.dataset:
            graph = process_graph_for_shapeNet(args, graph, num_categories=16)
         
    edge_index, _ = remove_self_loops(graph.edge_index)
    graph.edge_index = edge_index
    edge_index, edge_weight = gcn_norm(graph.edge_index, add_self_loops=True)
    graph.edge_index = edge_index
    graph.edge_weight = edge_weight

    if args.classify:
        if 'SHAPENET' in args.dataset:
                graph.y = F.one_hot(graph.y.long(), num_classes=50).float()
        else:
                graph.y = F.one_hot(graph.y.long()).float()
    else:
        graph.y = graph.y.float()

    return graph


def task_specific_modifiers(graph, args, forward_op, dataset):
    if args.task == 'mask':
        if args.classify == 1:
            # mask_budget = n_classes * args.mask_per_class_budget * batch_size 
            n_classes = dataset.num_classes
            total_mask_budget_across_batches_per_class = int(args.train_batch_size*args.mask_per_class_budget)
            mask_indices = []
            for c in range(n_classes):
                class_ind = torch.where(graph.y.argmax(dim=-1) == c)[0]
                sampled_shuffled_class_ind = list(class_ind[torch.randperm(len(class_ind))[:total_mask_budget_across_batches_per_class]])
                mask_indices = mask_indices + sampled_shuffled_class_ind
            mask_indices = torch.Tensor(mask_indices, device=graph.x.device).long()
            rand_mask = mask_indices
        else:
            # regression
            mask_indices = []
            for c in range(args.train_batch_size):
                # args.train_batch_size is batch_of_train_snapshots
                snapshot_ind = torch.where(graph.batch == c)[0]
                sampled_shuffled_class_ind = list(snapshot_ind[torch.randperm(len(snapshot_ind))[:args.mask_per_snapshot_budget]])
                mask_indices = mask_indices + sampled_shuffled_class_ind
            mask_indices = torch.Tensor(mask_indices, device=graph.x.device).long()
            rand_mask = mask_indices
        
        forward_op.ind = rand_mask

    elif args.task == 'path':
        # if epoch == 0:
        forward_op.gen_paths(nnodes=graph.x.shape[0], edge_index=graph.edge_index)
    # elif args.task == 'edgeRecovery':
    #     if epoch == 0:
    #         global rand_edge_weight
    #         rand_edge_weight = torch.rand(graph.edge_index.shape[-1],
    #                                         device=graph.edge_weight.device)  # normalize with D^-1
    #         # torch_geometric.utils.
    #         rand_edge_weight = rand_edge_weight  # / rand_edge_weight.sum(dim=1, keepdim=True)
    #     graph.edge_weight = rand_edge_weight  # graph.edge_attr#rand_edge_weight.clone()
    
    return graph, forward_op


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters())