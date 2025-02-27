import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from scipy.sparse.linalg import spsolve

import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from forwardOps import Embed
# from torch_geometric.utils import get_laplacian
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import to_scipy_sparse_matrix, to_torch_coo_tensor
from torch_geometric.data import Data
# from utils import compute_weighted_laplacian, laplacian_smoothing_loss
import scipy.sparse as sp 
import scipy.sparse.linalg as splinalg

def compute_weighted_laplacian_sparse(edge_index):
    data = Data(edge_index=edge_index) 
    # Convert to a PyTorch sparse tensor 
    A = to_torch_coo_tensor(data.edge_index) 
    d = A.sum(dim=1).values()
    indices = torch.arange(len(d)).to(d.device)
    indices = torch.stack([indices, indices]) 
    # Create the sparse diagonal matrix 
    D = torch.sparse_coo_tensor(indices, d, size=(len(d), len(d)))
    L = D - A  # Laplacian matrix
    return L
def speye(n):
    # Generate indices for the diagonal elements 
    indices = torch.arange(n) 
    indices = torch.stack([indices, indices]) 
    # # Create a tensor of ones for the diagonal values 
    values = torch.ones(n) 
    # # Create the sparse identity matrix 
    sparse_identity_matrix = torch.sparse_coo_tensor(indices, values, size=(n, n))
    return sparse_identity_matrix

def to_scipy_sparse_matrix(tensor): 
    indices = tensor.coalesce().indices().numpy() 
    values = tensor.coalesce().values().numpy() 
    shape = tensor.shape 
    return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
def solve_sparse(A, b):
    A_scipy = to_scipy_sparse_matrix(A) # Solve A_inverse * b using SciPy 
    b_numpy = b.numpy() 
    x_numpy = splinalg.spsolve(A_scipy, b_numpy) 
    # Convert the result back to a PyTorch tensor 
    x_torch = torch.from_numpy(x_numpy)
    return x_torch

def compute_weighted_laplacian(edge_index, edge_weight, num_nodes):
    A = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]  # Weighted adjacency matrix
    D = torch.diag(A.sum(dim=1))  # Degree matrix
    L = D - A  # Laplacian matrix
    return L

# Laplacian smoothing loss
def laplacian_smoothing_loss(pred, L):
    smoothness_term = 0.5 * torch.matmul(torch.matmul(pred.t(), L), pred).trace()
    return smoothness_term

class Laplace_noReg_Net(nn.Module):
    def __init__(self, forward_op, args, reg, device):
        super(Laplace_noReg_Net, self).__init__()
        self.forward_op = forward_op
        self.args = args
        self.lr = args.lr 
        self.classify = args.classify
        self.device = device
        self.reg = reg
        self.tol = args.LapNoRegNet_tol
        self.iters = args.solveIter
    def forward(self, forward_data, graph, num_iters=16, alpha=0.1):
        num_iters = self.iters
        tol = self.tol

        X = torch.zeros_like(graph.y, device=self.device) 
        if self.args.classify == 1:
            X.requires_grad = True

        if self.reg == 'tikhonov_regularization':
            # laplacian = torch.eye(graph.x.shape[0], device=self.device)
            laplacian = speye(graph.x.shape[0]).to(self.device)
            # L = torch.eye(graph.x.shape[0], device=self.device)
            alpha = 0.0
        elif self.reg == 'laplacian_regularization':
            # case where we use Equation (10) in the GRIPs paper (https://arxiv.org/pdf/2408.10436)
            # to solve the equivalent problem
            laplacian = compute_weighted_laplacian_sparse(graph.edge_index)
            I = speye(laplacian.shape[-1]).to(self.device)
            laplacian = laplacian + 1e-1*I
            alpha = 0.0 
        elif self.reg == 'laplacian_explicit':
            # case where we solve the laplacian regularization case without using Equation (10) in the paper 
            laplacian = compute_weighted_laplacian_sparse(graph.edge_index)
            I = speye(laplacian.shape[-1]).to(self.device)
            laplacian = laplacian + 1e-1*I
            alpha = alpha

        for iter in range(num_iters):
            if self.args.task == 'edgeRecovery':
                forward_data_rec = self.forward_op(graph.y, graph.edge_index, X.mean(dim=-1), emb=False)
            else:
                if self.args.classify ==1:
                    forward_data_rec = self.forward_op(X, graph.edge_index, graph.edge_weight, emb=False)
                else:
                    forward_data_rec = self.forward_op(X, graph.edge_index, graph.edge_weight, emb=False)
             
            laplace_loss_term = laplacian_smoothing_loss(X, L = laplacian)

            if self.args.classify == 1:
                nf = (forward_data**2).mean()
                reg = laplace_loss_term
                misfit = F.mse_loss(forward_data_rec, forward_data)
                loss = misfit + alpha*reg
                d_loss = torch.autograd.grad(loss, X, create_graph=False)[0]
                if self.reg == 'laplacian_explicit': 
                    dX = d_loss
                else:
                    dX = solve_sparse(laplacian.cpu(), d_loss.cpu())
                    dX = dX.to(self.device)
                X = X - self.lr*dX
                if X.grad is not None:
                    X.grad.zero_()
                print(f"iter: {iter}, misfit: {misfit}, reg:{reg}")
            else:
                with torch.no_grad():
                    nf = (forward_data**2).mean()
                    r = forward_data_rec - forward_data
                    misfit = (r**2).mean()/(2*nf)
                    reg = laplace_loss_term
                    loss =  misfit + alpha*reg
                    #####
                    d_misfit = self.forward_op.adjoint(r, graph.edge_index, graph.edge_weight) / nf
                    d_reg = (laplacian @ X) 
                    d_loss = d_misfit + alpha*d_reg 
                    dX = solve_sparse(laplacian.cpu(), d_loss.cpu())
                    dX = dX.to(self.device)
                    if len(dX.shape) == 1:
                        dX = dX.unsqueeze(-1)

                    X = X - self.lr*dX
                    if iter % 1 == 0:
                        print(f"iter: {iter}, loss: {loss},misfit: {misfit}, reg:{reg}") 
            # # Print loss 
            # if (iter + 1) % 100 == 0:
            #     print(f'Iteration [{iter + 1}/{num_iters}], Loss: {loss.item():.4f}')
            if misfit <= tol :
                break
        return X.detach()
class graphResNetFO(nn.Module):
    def __init__(self, num_layers, nopen, nfeatures, dropout=0.0):
        super(graphResNetFO, self).__init__()
        self.dropout = dropout
        self.h = 0.1
        self.num_layers = num_layers
        self.nfeatures = nfeatures
        self.Kf = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen + nfeatures, nfeatures)))
        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nfeatures, nfeatures)))
        self.bns = nn.ModuleList()
        self.nchannels = nopen
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(nfeatures))

    def forward(self, Z, Zall=[], f=None, edge_index=None, edge_weight=None):
        Zold = Z.clone()
        # Z = F.conv2d(Z, self.Kin, padding=self.Kin.shape[-1] // 2)

        for i in range(self.num_layers):
            # dZ = F.conv2d(Z, self.K[i], padding=self.K[i].shape[-1] // 2)
            # dZ = F.instance_norm(dZ)
            # dZ = F.leaky_relu(dZ, negative_slope=0.2)
            # dZ = F.conv_transpose2d(dZ, self.K[i], padding=self.K.shape[-1] // 2)
            if self.dropout > 0:
                Z = F.dropout(Z, p=self.dropout, training=self.training)
            if f is not None:
                Z = torch.cat([Z, f], dim=-1)  # nX(e+c)
                Z = F.silu(Z @ self.Kf[i])  # nXe
            # desired: phi = \sum((sigma(G@Z@K))), dphi/dz = G.T@(sigma(G@Z@K))@K.T
            # currently: phi = (sigma(A@Z@K))
            dZ = Z @ self.K[i]  # F.conv2d(Z, K, padding=K.shape[-1] // 2)
            if edge_index is not None:
                # adj = D^-0.5 @ A @ D^-0.5
                # dZ = (adj @ dZ)
                dZ = nodeGrad(dZ, edge_index, edge_weight)
                # TODO: replace by grad
            # dZ = F.instance_norm(dZ)
            dZ = F.leaky_relu(dZ, negative_slope=0.2)
            # dZ = F.silu(dZ)

            # apply div. to dZ
            if edge_index is not None:
                dZ = edgeDiv(dZ, edge_index, edge_weight, Z.shape[0])
            dZ = dZ @ self.K[i].t()  # F.conv_transpose2d(dZ, K, padding=K.shape[-1] // 2)

            tmp = Z.clone()
            Z  = Z - dZ  #Z = 2 * Z - Zold - 1 * dZ
            # Zold = tmp

        # close
        return Z, Z

class resNetFO(nn.Module):
    def __init__(self, num_layers, nopen, embed_proj=False):
        super(resNetFO, self).__init__()
        self.embed_proj = embed_proj
        if self.embed_proj:
            # self.embed = torch.nn.Conv2d(in_channels=3, out_channels=nopen, kernel_size=3, padding=1)
            # self.proj = torch.nn.Conv2d(in_channels=nopen, out_channels=3, kernel_size=1, padding=0)
            self.proj = Embed(embdsize=nopen, nin=3)
            self.embed = Embed(embdsize=3, nin=nopen)

            self.bn_embed = torch.nn.BatchNorm2d(nopen)

        self.h = 0.1
        self.num_layers = num_layers

        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen, nopen, 3, 3)))
        self.K2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen, nopen, 3, 3)))

    def forward(self, Z, Zall=[]):
        if self.embed_proj:
            Z = self.embed(Z)
        for i in range(self.num_layers):
            dZ = F.conv2d(Z, self.K[i], padding=self.K[i].shape[-1] // 2)
            dZ = F.instance_norm(dZ)
            dZ = F.leaky_relu(dZ, negative_slope=0.2)
            if self.embed_proj:
                dZ = F.conv2d(dZ, self.K2[i], padding=self.K[i].shape[-1] // 2)
            Z = Z + dZ
        # close
        if self.embed_proj:
            Z = self.proj(Z)
        return Z, Z


class leastActionNet(nn.Module):
    def __init__(self, nlayers, nchanels, nfixPointIter, imsize):
        super(leastActionNet, self).__init__()

        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nlayers, nchanels, nchanels, 3, 3)))
        self.nlayers = nlayers
        self.nfixPointIter = nfixPointIter
        self.X0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, nchanels, imsize[0], imsize[1])))

    def layer(self, Z, K):
        dZ = F.conv2d(Z, K, padding=K.shape[-1] // 2)
        dZ = F.instance_norm(dZ)
        dZ = F.leaky_relu(dZ, negative_slope=0.2)
        dZ = F.conv_transpose2d(dZ, K, padding=K.shape[-1] // 2)
        return dZ

    def getNoNlinRHS(self, Z, XN):
        Y = torch.zeros_like(Z)
        for i in range(Y.shape[0]):
            Y[i] = -self.layer(Z[i].clone(), self.K[i])
        Y[-1] = Y[-1].clone() + XN
        Y[0] = Y[0].clone() + self.X0

        return Y

    def triDiagSolve(self, Z):
        # forward pass
        nlayers = Z.shape[0]
        Y = torch.zeros_like(Z)
        Y[0] = np.sqrt(1 / 2.0) * Z[0].clone()
        for i in range(1, nlayers):
            a = np.sqrt((i + 1) / (i + 2))
            b = np.sqrt((i) / (i + 1))
            Y[i] = a * (b * Y[i - 1].clone() + Z[i].clone())
        # backward pass
        W = torch.zeros_like(Z)
        a = np.sqrt(nlayers / (nlayers + 1))
        W[-1] = a * Y[-1].clone()
        for i in np.flip(range(nlayers - 1)):
            a = np.sqrt((i + 1) / (i + 2))
            W[i] = a * (a * W[i + 1].clone() + Y[i].clone())

        return W

    def forward(self, X, Z=[]):
        if len(Z) == 0:
            Z = torch.zeros_like(X).unsqueeze(0).repeat_interleave(self.nlayers, dim=0)

        for k in range(self.nfixPointIter):
            Z = self.getNoNlinRHS(Z, X)
            Z = self.triDiagSolve(Z)
        return Z[-1], Z


class neuralProximalGradient(nn.Module):
    def __init__(self, regNet, dataProj, forOp, niter=1):
        super(neuralProximalGradient, self).__init__()
        self.net = regNet
        self.dataProj = dataProj
        self.forOp = forOp
        self.niter = niter

        self.proj = Embed(embdsize=3, nin=3)
        self.mu = torch.nn.Parameter(torch.Tensor([0.01]))

    def forward(self, D):
        # initial recovey
        Z = self.forOp.adjoint(D, emb=False)
        Az = self.forOp.forward(Z, emb=False)
        alpha = (D * Az).mean(dim=(1, 2, 3), keepdim=True) / (Az * Az).mean(dim=(1, 2, 3), keepdim=True)
        Z = alpha * Z
        # Zref = torch.zeros((D.shape[0], 3, D.shape[2], D.shape[-1])).to(D.device)
        # Z = self.forOp.adjoint(D)
        # Zref = torch.zeros_like(Z)
        # Z, R = self.dataProj(D, Zref)
        # Zall = []
        Zall = []

        for i in range(self.niter):
            # network
            Zref, Zall = self.net(Z, Zall)
            # Zref = Z
            R = D - (self.forOp.forward(Zref, emb=False))
            # print("Iter:", i, ", Rnorm/Dnorm:", (R.norm()/D.norm()).item())
            G = self.forOp.adjoint(R, emb=False)
            Ag = self.forOp.forward(G, emb=False)
            mu = (R * Ag).mean(dim=(1, 2, 3), keepdim=True) / (Ag * Ag).mean(dim=(1, 2, 3), keepdim=True)
            Z = Zref + mu * G

        X = Z
        Xref = Zref
        return X, Xref, torch.Tensor([0])

class graph_inverseSolveNet(nn.Module):
    def __init__(self, regNet, dataProj, forOp, niter=1, input_feat_dim=None, rnfPE=False, task='path',
                 learn_emb=False):
        super(graph_inverseSolveNet, self).__init__()
        self.rnfPE = rnfPE
        self.net = regNet
        self.task = task
        self.learn_emb = learn_emb
        if rnfPE:
            if regNet is not None:
                self.rnf_embed = torch.nn.Linear(self.net.nchannels, self.net.nchannels)
                self.rnf_combine = torch.nn.Linear(2 * self.net.nchannels, self.net.nchannels)
        self.dataProj = dataProj
        self.forOp = forOp
        self.niter = niter
        self.feat_channels = input_feat_dim
        if self.feat_channels is not None:
            if regNet is not None:
                self.feat_embed = torch.nn.Linear(input_feat_dim, self.net.nchannels)

        self.solEmbed = self.net.nchannels if learn_emb else 1

        if task == 'edgeRecovery':
            self.edge_sum_mlp = torch.nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(self.solEmbed, self.solEmbed)))
            self.edge_grad_mlp = torch.nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(self.solEmbed, self.solEmbed)))

    def forward(self, D, edge_index, edge_weights, f=None, graph=None, xE=None):
        if f is not None:
            if self.net is not None:
                f = self.feat_embed(f)
                if self.rnfPE:
                    rnf = torch.randn(f.shape[0], self.net.nchannels, device=D.device)
                    rnf = torch.sin(self.rnf_embed(rnf))
                    f = torch.cat([f, rnf], dim=-1)
                    f = self.rnf_combine(f)

        nnodes = len(torch.unique(edge_index))
        # adj = torch.zeros((nnodes, nnodes), device=D.device)
        # adj[edge_index[0, :], edge_index[1, :]] = edge_weights

        # initial recovery
        # graph unsmoothing/deblurring, D = A @ f, adj(D) = A.T @ A @ f , A.T = A
        # self, seq, edge_weights, node_features, edge_index, emb = True

        if self.task == 'edgeRecovery':
            # graph.xN = f
            edge_weights_pred = torch.randn((edge_weights.shape[0], self.solEmbed), device=f.device).squeeze()
            edge_weights_pred.requires_grad = True
            # xNprime, xE, graph, emb = False
            # graph.xN = f
            Z = self.forOp.adjoint(D, edge_weights_pred, graph, emb=self.learn_emb)
        else:

            Z = self.forOp.adjoint(D, edge_index, edge_weights, emb=self.learn_emb)

        Zref = torch.zeros_like(Z)

        if self.task == 'edgeRecovery':
            # Zref = 0.1 * torch.randn(Z)
            if xE is not None:
                Zref = xE.clone()
            Zref.requires_grad = True
            Z, R = self.dataProj(D, Zref, graph, emb=self.learn_emb, niter=10000)
        else:
            Z, R = self.dataProj(D, Zref, edge_index, edge_weights, emb=self.learn_emb)
        # with torch.no_grad():
        # Z = getInitialLabels(D, self.forOp, edge_index, edge_weights, gamma=1e-1, niters=10, tol=1e-3, emb=True)
        # Z = Z.detach()
        Zall = []

        if self.net is not None:
            #case where regularization (reg_model, which self.net here) is set to 'None'
            for i in range(self.niter):
                # network

                if self.task == 'edgeRecovery':
                    Z = Z.unsqueeze(-1) if not self.learn_emb else Z
                    z_to_phi = edgeDiv(Z, graph.edge_index, graph.edge_weight,
                                    nnodes=f.shape[0])  # TODO: Work on edge features instead of nodes.
                else:
                    z_to_phi = Z

                Zref, Zall = self.net(z_to_phi,
                                    Zall,
                                    f,
                                    edge_index,
                                    graph.edge_weight if self.task == 'edgeRecovery' else edge_weights)

                # data projection
                if self.task == 'edgeRecovery':
                    # Zref = self.edge_mlp(torch.cat([nodeGrad(Z, graph.edge_index, graph.edge_weight), nodeSum(f, graph.edge_index, graph.edge_weight)], dim=-1))
                    Zref_grad = (F.tanh(
                        nodeGrad(Z, graph.edge_index, graph.edge_weight) @ self.edge_grad_mlp)) @ self.edge_grad_mlp.t()
                    Zref_sum = (F.tanh(
                        nodeSum(Z, graph.edge_index, graph.edge_weight) @ self.edge_sum_mlp)) @ self.edge_sum_mlp.t()
                    Zref = Zref_grad + Zref_sum
                    Zref = Zref.squeeze()
                    Z, R = self.dataProj(D, Zref.squeeze(), graph, emb=self.learn_emb)

                else:
                    Z, R = self.dataProj(D, Zref, edge_index, edge_weights, emb=self.learn_emb)  # , emb=self.learn_emb
        
        if self.learn_emb:
            X = self.forOp.Emb(Z)  # .unsqueeze(-1)
            Xref = self.forOp.Emb(Zref)
        else:
            X = Z
            Xref = Zref
        return X, Xref, R


class graphLeastActionNet(nn.Module):
    def __init__(self, nlayers, nchanels, nfixPointIter, imsize):
        super(graphLeastActionNet, self).__init__()
        self.nchannels = nchanels
        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nlayers, nchanels, nchanels)))
        self.nlayers = nlayers
        self.nfixPointIter = nfixPointIter
        # self.X0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, nchanels, imsize[0], imsize[1])))
        self.K_features = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nlayers, 2 * nchanels, nchanels)))
        # self.X0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(nchanels, imsize[0], imsize[1])))

    def layer(self, Z, K, f=None, Kf=None, edge_index=None, edge_weight=None):
        nnodes = Z.shape[0]
        # Z is of shape nxe, e is the embedding of the density , Z nxe
        # f is of shape nxc, c is the input feature dimensionality f nxc
        if f is not None:
            Z = torch.cat([Z, f], dim=-1)  # nX(e+c)
            Z = F.silu(Z @ Kf)  # nXe
        # desired: phi = \sum((sigma(G@Z@K))), dphi/dz = G.T@(sigma(G@Z@K))@K.T
        # currently: phi = (sigma(A@Z@K))
        dZ = Z @ K  # F.conv2d(Z, K, padding=K.shape[-1] // 2)
        if edge_index is not None:
            # adj = D^-0.5 @ A @ D^-0.5
            # dZ = (adj @ dZ)
            dZ = nodeGrad(dZ, edge_index, edge_weight)
            # TODO: replace by grad
        # dZ = F.instance_norm(dZ)
        # dZ = F.leaky_relu(dZ, negative_slope=0.2)
        dZ = F.silu(dZ)

        # apply div. to dZ
        dZ = edgeDiv(dZ, edge_index, edge_weight, nnodes)
        dZ = dZ @ K.t()  # F.conv_transpose2d(dZ, K, padding=K.shape[-1] // 2)

        return dZ

    def getNoNlinRHS(self, Z, XN, f=None, edge_index=None, edge_weight=None):
        Y = torch.zeros_like(Z)
        for i in range(Y.shape[0]):
            # if f is not None:
            #    K = self.K_features[i]
            # else:
            K = self.K[i]
            Y[i] = -self.layer(Z[i].clone(), K, f, Kf=self.K_features[i], edge_index=edge_index,
                               edge_weight=edge_weight)
        Y[-1] = Y[-1].clone() + XN
        Y[0] = Y[0].clone()  # +   # + self.X0 % TODO: think what to do with X0
        # min
        return Y

    def triDiagSolve(self, Z):
        # forward pass
        nlayers = Z.shape[0]
        Y = torch.zeros_like(Z)
        Y[0] = np.sqrt(1 / 2.0) * Z[0].clone()
        for i in range(1, nlayers):
            a = np.sqrt((i + 1) / (i + 2))
            b = np.sqrt((i) / (i + 1))
            Y[i] = a * (b * Y[i - 1].clone() + Z[i].clone())
        # backward pass
        W = torch.zeros_like(Z)
        a = np.sqrt(nlayers / (nlayers + 1))
        W[-1] = a * Y[-1].clone()
        for i in np.flip(range(nlayers - 1)):
            a = np.sqrt((i + 1) / (i + 2))
            W[i] = a * (a * W[i + 1].clone() + Y[i].clone())

        return W

    def forward(self, X, Z=[], f=None, edge_index=None, edge_weight=None):
        if len(Z) == 0:
            Z = torch.zeros_like(X).unsqueeze(0).repeat_interleave(self.nlayers, dim=0)

        for k in range(self.nfixPointIter):
            Z = self.getNoNlinRHS(Z, X, f, edge_index, edge_weight)
            Z = self.triDiagSolve(Z)
        return Z[-1], Z


class edge_recovery_proj(nn.Module):
    def __init__(self, forOp, mu=0.01, device='cuda'):
        super(edge_recovery_proj, self).__init__()
        self.forOp = forOp
        self.mu = mu

    def forward(self, D, xE, graph, emb=True, niter=10):
        # xE = Zref.mean(dim=-1)
        emb = (True if len(xE.shape) > 1 else False) or emb
        normr0 = 0
        for ii in range(niter):
            if emb == False:
                xEin = xE  # .unsqueeze(-1)
            else:
                xEin = xE
            Dc = self.forOp(xEin, graph, emb=emb)
            r = Dc - D
            if ii == 0:
                normr0 = r.abs().mean()
            if len(D.shape) == 2:
                w = ((D.std(dim=1).squeeze() + 1e-5)).unsqueeze(1).unsqueeze(1)
            else:
                w = ((D.std(dim=1).squeeze() + 1e-5)).unsqueeze(1)

            # r = w*r
            g = self.forOp.adjoint(r, xEin, graph, emb=emb)
            # print("r norm:", r.abs().mean() / normr0, "g norm:", g.abs().mean())
            xE = xE - self.mu * g

        rnorm = r.abs().mean()
        if rnorm > normr0:
            r = torch.Tensor([-1]).to(r.device)

        return xE, r


class graph_CGLS(nn.Module):
    def __init__(self, forOp, CGLSit=10, eps=1e-2, device='cuda'):
        super(graph_CGLS, self).__init__()
        self.forOp = forOp
        self.nCGLSiter = CGLSit
        self.eps = eps

    def forward_landweber(self, b, xref, edge_index, edge_weights, zref=[], xN=None, emb=True):
        x = xref

        r = b - self.forOp(x, edge_index, edge_weights, emb=emb)
        if r.norm() / b.norm() < self.eps:
            return x, r
        s = self.forOp.adjoint(r, edge_index, edge_weights)
        for k in range(self.nCGLSiter):
            g = self.forOp.adjoint(r, edge_index, edge_weights, emb=emb)
            Ag = self.forOp(g, edge_index, edge_weights, emb=emb)
            delta = torch.norm(Ag) ** 2
            gamma = torch.norm(g) ** 2
            alpha = gamma / delta

            x = x + alpha * g
            r = r - alpha * Ag

            if torch.norm(r) / torch.norm(b) < self.eps:
                return x, r
            print("iter, ", k, ", r=", r.norm())

        return x, r

    def forward(self, b, xref, edge_index, edge_weights, zref=[], xN=None, emb=False):
        x = xref

        r = b - self.forOp(x, edge_index, edge_weights, emb=emb)
        if r.norm() / b.norm() < self.eps:
            return x, r
        s = self.forOp.adjoint(r, edge_index, edge_weights, emb=emb)
        # Initialize
        p = s
        norms0 = torch.norm(s)
        gamma = norms0 ** 2

        for k in range(self.nCGLSiter):
            q = self.forOp(p, edge_index, edge_weights, emb=emb)
            delta = torch.norm(q) ** 2
            alpha = gamma / delta

            x = x + alpha * p
            r = r - alpha * q

            # print(k, r.norm().item() / b.norm().item())
            if r.norm() / b.norm() < self.eps:
                return x, r

            s = self.forOp.adjoint(r, edge_index, edge_weights, emb=emb)

            norms = torch.norm(s)
            gamma1 = gamma
            gamma = norms ** 2
            beta = gamma / gamma1
            p = s + beta * p
            # print("iter, ", k, ", r=", r.norm())
        return x, r


def getInitialLabels(D, forOp, edge_index, edge_weights, gamma=1e-1, niters=100, tol=1e-3, emb=False):
    b = forOp.adjoint(D, edge_index, edge_weights, emb=emb)
    x = torch.zeros_like(b, device=D.device)

    def A(x, edge_index, edge_weights, gamma, forOp):
        # lap_edge_index, lap_edge_weight = get_laplacian(edge_index, edge_weights)
        # lap = torch.zeros(x.shape[0], x.shape[0], device=x.device)
        # lap[lap_edge_index[0, :], lap_edge_index[1, :]] = lap_edge_weight
        # adj = torch.zeros(x.shape[0], x.shape[0], device=x.device)
        # adj[edge_index[0, :], edge_index[1, :]] = edge_weights
        # Sf = nodeSum(x, edge_index, edge_weights)
        # GTSf = edgeDiv(Sf, edge_index, edge_weights, nnodes=x.shape[0])

        Gf = nodeGrad(x, edge_index, edge_weights)
        GTGf = edgeDiv(Gf, edge_index, edge_weights, nnodes=x.shape[0])
        # r = gamma * (lap @ x)
        r = gamma * GTGf
        q = forOp(x, edge_index, edge_weights, emb=emb)
        q = forOp.adjoint(q, edge_index, edge_weights, emb=emb)
        return r + q

    r = b.clone()
    p = r
    for i in range(niters):
        ap = A(p, edge_index, edge_weights, gamma, forOp)
        alpha = (r * r).mean() / (p * ap).mean()
        x = x + alpha * p
        rnew = r - alpha * ap
        beta = (rnew * rnew).mean() / (r * r).mean()
        r = rnew.clone()
        p = r + beta * p
        if r.norm() < tol:
            break
        res = D - forOp(x, edge_index, edge_weights, emb=emb)
        # print("iter:", i, ", residual:", r.norm() / b.norm(),", data fit res:", res.norm())
    return x


def nodeGrad(f, edge_index, edge_weight):
    Gf = edge_weight.unsqueeze(-1) * (f[edge_index[0, :], :] - f[edge_index[1, :], :])
    return Gf


def edgeDiv(Gf, edge_index, edge_weight, nnodes):
    GTGf = torch.zeros(nnodes, Gf.shape[-1], device=Gf.device)
    GTGf.index_add_(0, edge_index[0, :], Gf)
    return GTGf


def nodeSum(f, edge_index, edge_weight):
    Sf = edge_weight.unsqueeze(-1) * (f[edge_index[1, :], :])
    return Sf


class graph_NeuralProximalGradient(nn.Module):
    def __init__(self, regNet, dataProj, forOp, niter=1, input_feat_dim=None, rnfPE=False):
        super(graph_NeuralProximalGradient, self).__init__()
        self.net = regNet
        self.dataProj = dataProj
        self.forOp = forOp
        self.niter = niter

        # self.proj = Embed(embdsize=3, nin=3)
        self.mu = torch.nn.Parameter(torch.Tensor([0.01]))

        self.rnfPE = rnfPE
        self.net = regNet

        if rnfPE:
            self.rnf_embed = torch.nn.Linear(self.net.nchannels, self.net.nchannels)
            self.rnf_combine = torch.nn.Linear(2 * self.net.nchannels, self.net.nchannels)
        self.dataProj = dataProj
        self.forOp = forOp
        self.niter = niter
        self.feat_channels = input_feat_dim
        if self.feat_channels is not None:
            self.feat_embed = torch.nn.Linear(input_feat_dim, self.net.nchannels)

    def forward(self, D, edge_index, edge_weights, f=None):

        if f is not None:
            f = self.feat_embed(f)
            if self.rnfPE:
                rnf = torch.randn(f.shape[0], self.net.nchannels, device=D.device)
                rnf = torch.sin(self.rnf_embed(rnf))
                f = torch.cat([f, rnf], dim=-1)
                f = self.rnf_combine(f)

        # initial recovey
        Z = self.forOp.adjoint(D, edge_index, edge_weights, emb=False)
        Az = self.forOp.forward(Z, edge_index, edge_weights, emb=False)
        alpha = (D * Az).mean(dim=(0, 1), keepdim=True) / (Az * Az).mean(dim=(0, 1), keepdim=True)
        Z = alpha * Z
        # Zref = torch.zeros((D.shape[0], 3, D.shape[2], D.shape[-1])).to(D.device)
        # Z = self.forOp.adjoint(D)
        # Zref = torch.zeros_like(Z)
        # Z, R = self.dataProj(D, Zref)
        # Zall = []
        Zall = []
        step = 0.1
        for i in range(self.niter):
            # network
            Zref, Zall = self.net(Z, Zall, f, edge_index, edge_weights)
            Zref = Z + Zref
            # Zref, Zall = self.net(Z, Zall, f, edge_index, edge_weights) #this is the original code
            # Zref = Z 
            R = D - (self.forOp.forward(Zref, edge_index, edge_weights, emb=False))
            # print("Iter:", i, ", Rnorm/Dnorm:", (R.norm()/D.norm()).item())
            G = self.forOp.adjoint(R, edge_index, edge_weights, emb=False)
            Ag = self.forOp.forward(G, edge_index, edge_weights, emb=False)
            mu = (R * Ag).mean(dim=(0, 1), keepdim=True) / (Ag * Ag).mean(dim=(0, 1), keepdim=True)
            Z = Zref + mu * G
            # print(R.norm()/D.norm(), i)
        X = Z
        Xref = Zref
        return X, Xref, torch.Tensor([0])


class graphHyperResNet(nn.Module):
    def __init__(self, num_layers, nopen, nfeatures, dropout=0.0):
        super(graphHyperResNet, self).__init__()
        self.dropout = dropout
        self.h = 0.1
        self.num_layers = num_layers
        self.nfeatures = nfeatures
        self.Kf = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen + nfeatures, nfeatures)))
        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nfeatures, nfeatures)))
        self.bns = nn.ModuleList()
        self.nchannels = nopen
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(nfeatures))

    def forward(self, Z, Zall=[], f=None, edge_index=None, edge_weight=None):
        Zold = Z.clone()
        # Z = F.conv2d(Z, self.Kin, padding=self.Kin.shape[-1] // 2)

        for i in range(self.num_layers):
            # dZ = F.conv2d(Z, self.K[i], padding=self.K[i].shape[-1] // 2)
            # dZ = F.instance_norm(dZ)
            # dZ = F.leaky_relu(dZ, negative_slope=0.2)
            # dZ = F.conv_transpose2d(dZ, self.K[i], padding=self.K.shape[-1] // 2)
            if self.dropout > 0:
                Z = F.dropout(Z, p=self.dropout, training=self.training)
            if f is not None:
                Z = torch.cat([Z, f], dim=-1)  # nX(e+c)
                Z = F.silu(Z @ self.Kf[i])  # nXe
            # desired: phi = \sum((sigma(G@Z@K))), dphi/dz = G.T@(sigma(G@Z@K))@K.T
            # currently: phi = (sigma(A@Z@K))
            dZ = Z @ self.K[i]  # F.conv2d(Z, K, padding=K.shape[-1] // 2)
            if edge_index is not None:
                # adj = D^-0.5 @ A @ D^-0.5
                # dZ = (adj @ dZ)
                dZ = nodeGrad(dZ, edge_index, edge_weight)
                # TODO: replace by grad
            # dZ = F.instance_norm(dZ)
            dZ = F.leaky_relu(dZ, negative_slope=0.2)
            # dZ = F.silu(dZ)

            # apply div. to dZ
            if edge_index is not None:
                dZ = edgeDiv(dZ, edge_index, edge_weight, Z.shape[0])
            dZ = dZ @ self.K[i].t()  # F.conv_transpose2d(dZ, K, padding=K.shape[-1] // 2)

            tmp = Z.clone()
            Z = 2 * Z - Zold - 1 * dZ
            Zold = tmp
        # close
        return Z, Z


class graphScaleSpaceNet(nn.Module):
    def __init__(self, num_layers, nopen, nfeatures, dropout=0.0):
        super(graphScaleSpaceNet, self).__init__()
        self.dropout = dropout
        self.h = 0.1
        self.num_layers = num_layers
        self.nfeatures = nfeatures
        self.Kf = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nopen + nfeatures, nfeatures)))
        self.K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_layers, nfeatures, nfeatures)))
        self.bns = nn.ModuleList()
        self.nchannels = nopen
        for i in range(num_layers):
            self.bns.append(nn.BatchNorm2d(nfeatures))

    def forward(self, Z, Zall=[], f=None, edge_index=None, edge_weight=None):
        # Z = F.conv2d(Z, self.Kin, padding=self.Kin.shape[-1] // 2)

        for i in range(self.num_layers):
            # dZ = F.conv2d(Z, self.K[i], padding=self.K[i].shape[-1] // 2)
            # dZ = F.instance_norm(dZ)
            # dZ = F.leaky_relu(dZ, negative_slope=0.2)
            # dZ = F.conv_transpose2d(dZ, self.K[i], padding=self.K.shape[-1] // 2)
            if self.dropout > 0:
                Z = F.dropout(Z, p=self.dropout, training=self.training)
            if f is not None:
                Z = torch.cat([Z, f], dim=-1)  # nX(e+c)
                Z = F.silu(Z @ self.Kf[i])  # nXe
            # desired: phi = \sum((sigma(G@Z@K))), dphi/dz = G.T@(sigma(G@Z@K))@K.T
            # currently: phi = (sigma(A@Z@K))
            dZ = Z @ self.K[i]  # F.conv2d(Z, K, padding=K.shape[-1] // 2)
            if edge_index is not None:
                # adj = D^-0.5 @ A @ D^-0.5
                # dZ = (adj @ dZ)
                dZ = nodeGrad(dZ, edge_index, edge_weight)
                # TODO: replace by grad
            # dZ = F.instance_norm(dZ)
            dZ = F.leaky_relu(dZ, negative_slope=0.2)
            # dZ = F.silu(dZ)

            # apply div. to dZ
            if edge_index is not None:
                dZ = edgeDiv(dZ, edge_index, edge_weight, Z.shape[0])
            dZ = dZ @ self.K[i].t()  # F.conv_transpose2d(dZ, K, padding=K.shape[-1] // 2)

            Z = Z - dZ

        # close
        return Z, Z
    