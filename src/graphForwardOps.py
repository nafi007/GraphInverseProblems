import os, sys
import torch
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.sparse as sparse
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
import networks
from torch_geometric.utils import get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import Node2Vec

from torch_cluster import knn, random_walk


class graphEmbed(nn.Module):
    def __init__(self, embdsize, nin=3, learned=True, device='cuda'):
        super(graphEmbed, self).__init__()
        if learned:
            self.k = 9
            self.Emb = (nn.init.xavier_uniform_(torch.empty(embdsize, nin, device=device)))
            # id = torch.zeros((self.k, self.k))
            # id[self.k // 2, self.k // 2] = 1
            # self.Emb[0, 0, :, :] = id
            # self.Emb[1, 1, :, :] = id
            # self.Emb[2, 2, :, :] = id

            self.Emb = nn.Parameter(self.Emb)
            self.bias_back = nn.Parameter(torch.zeros(nin))
            self.bias_for = nn.Parameter(torch.zeros(embdsize))

        else:
            self.Emb = torch.eye(nin, nin, device=device)  # .unsqueeze(-1).unsqueeze(-1)
            self.bias_back = torch.zeros(embdsize)
            self.bias_for = torch.zeros(nin)

    def forward(self, I):
        Emb = self.Emb.to(I.device)
        #I = F.conv1d(I.t().unsqueeze(0), weight=Emb.unsqueeze(-1))  # , bias=self.bias_for

        #I = I.squeeze().t()
        I = I @ self.Emb
        return I  # I @ Emb#F.conv2d(I, Emb, padding=self.Emb.shape[-1] // 2)

    def backward(self, I):
        Emb = self.Emb.to(I.device)
        #I = F.conv_transpose1d(I.t().unsqueeze(0), weight=Emb.unsqueeze(-1))  # , bias=self.bias_back
        #I = I.squeeze().t()
        I = I @ Emb.t()
        return I  # I @ Emb.t()#F.conv_transpose2d(I, Emb, padding=self.Emb.shape[-1] // 2)


class maskImage(nn.Module):
    def __init__(self, ind, imsize, embdsize, nin=3, device='cuda', learnEmb=True):
        super(maskImage, self).__init__()
        ind = ind.to(device)
        self.ind = ind
        self.imsize = imsize
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, emb=True):
        if emb and self.learnEmb:
            I = self.Emb(I)
        Ic = I.reshape(I.shape[0], I.shape[1], -1)
        Ic = Ic[:, :, self.ind]
        return Ic

    def adjoint(self, Ic, emb=True):
        I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        I[:, :, self.ind] = Ic
        I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I


class graphMask(nn.Module):
    def __init__(self, ind, embdsize, nin=3, device='cuda', learnEmb=True):
        super(graphMask, self).__init__()
        ind = ind.to(device)
        self.ind = ind 
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            # I = I.unsqueeze(0)
            I = self.Emb(I)
        # Ic = I.reshape(I.shape[0], I.shape[1], -1)
        # Ic = Ic[:, :, self.ind]
        # if len(I.shape) == 1:
        #     I = I.unsqueeze(-1)
        Ic = I[self.ind, :]

        return Ic

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        # I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        nnodes = len(edge_index.unique())
        I = torch.zeros(nnodes, Ic.shape[1], device=Ic.device)

        I[self.ind, :] = Ic
        # I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I


class graphPath(nn.Module):
    def __init__(self, embdsize, nin=3, device='cuda', learnEmb=True, pathLength=3):
        super(graphPath, self).__init__()
        self.pathLength = pathLength
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)
        self.learnEmb = learnEmb
        self.node_seq = None
        self.device = device

    def gen_paths(self, nnodes, edge_index):
        L = self.pathLength
        # nodesIdx = torch.arange(nnodes).cuda() 
        # edge_index = edge_index.cuda()  # must be on the same device as nodesIdx
        nodesIdx = torch.arange(nnodes, device=self.device)
        edge_index = edge_index.to(self.device)
        # nodesIdx = torch.repeat_interleave(nodesIdx, dim=0, repeats=4)
        node_seq = random_walk(edge_index[0, :], edge_index[1, :], start=nodesIdx,
                               walk_length=L - 1, p=1, q=1).t().to(self.device)
        self.node_seq = node_seq

        T = torch.zeros(nnodes, nnodes, device=self.device)

        L = self.node_seq.shape[0]
        batch_indices = torch.arange(self.node_seq.shape[1]).to(T.device)
        batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L).flatten().to(T.device)
        batch_indices_j = self.node_seq.flatten().to(T.device)
        # for iii, vec in enumerate(self.node_seq.t()):
        #    for s in vec:
        #        T[iii, s] += 0.5
        T[batch_indices_i, batch_indices_j] += 0.5

        self.T = T
        return node_seq

    def forward(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            # I = I.unsqueeze(0)
            I = self.Emb(I)
        # Ic = I.reshape(I.shape[0], I.shape[1], -1)
        # Ic = Ic[:, :, self.ind]
        # Ic = I[self.ind, :]
        if False:
            Ic = 0.5 * I[self.node_seq, :].sum(dim=0)
        else:
            # Test dense:
            # nnodes = len(edge_index.unique())

            # T = torch.zeros(nnodes, nnodes, device=I.device)
            # for iii, vec in enumerate(self.node_seq):
            #     for s in vec:
            #         T[iii, s] += 0.5

            Ic = self.T @ I #self.T @ I.to(self.T.device)

        return Ic

    def adjoint(self, Ic, edge_index=None, edge_weight=None, emb=True):
        # I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        nnodes = len(edge_index.unique())
        # I = torch.zeros(nnodes, Ic.shape[1], device=Ic.device)

        # I[self.ind, :] = Ic
        # I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        if False:
            # batch_indices = torch.arange(node_seq.shape[0])
            # batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L)
            # adj_for_x = 0.5 * for_x[batch_indices_i.t(), :].sum(dim=0)

            L = self.node_seq.shape[0]
            batch_indices = torch.arange(self.node_seq.shape[1])
            batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L)

            # batch_indices_perm[]
            I = 0.5 * Ic[batch_indices_i.t(), :].sum(dim=0)  # P @ x , P shape nxn, x shape nxc,
            # 0 [ 1 3 4]
            # 1 [ 2 3 5]
            # 0 1 0 1 1 0 Beams x Length
            # 0 0 1 1 0 1
        else:
            # Test dense:
            # T = torch.zeros(nnodes, nnodes, device=Ic.device)
            # for iii, vec in enumerate(self.node_seq):
            #     for s in vec:
            #         T[iii, s] += 0.5

            I = self.T.t() @ Ic

        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I

    def forward2(self, I, edge_index=None, edge_weight=None, emb=True):
        if emb and self.learnEmb:
            # I = I.unsqueeze(0)
            I = self.Emb(I)
        # Ic = I.reshape(I.shape[0], I.shape[1], -1)
        # Ic = Ic[:, :, self.ind]
        # Ic = I[self.ind, :]
        Ic = I[self.node_seq, :]  # / self.node_seq.shape[0]
        Ic = 0.5 * (Ic[:-1, :] + Ic[1:, :])
        Ic = Ic.sum(dim=0)  # / self.node_seq.shape[0]

        return Ic

    def adjoint2(self, Ic, edge_index=None, edge_weight=None, emb=True):
        # I = torch.zeros(Ic.shape[0], Ic.shape[1], self.imsize[0] * self.imsize[1], device=Ic.device)
        nnodes = len(edge_index.unique())
        # I = torch.zeros(nnodes, Ic.shape[1], device=Ic.device)

        # I[self.ind, :] = Ic
        # I = I.reshape(Ic.shape[0], Ic.shape[1], self.imsize[0], self.imsize[1])
        L = self.node_seq.shape[0]
        batch_indices = torch.arange(self.node_seq.shape[1])
        batch_indices_i = torch.repeat_interleave(batch_indices.unsqueeze(-1), dim=1, repeats=L)
        I = Ic[batch_indices_i.t(), :]
        I = 0.5 * (I[:-1, :] + I[1:, :])
        I = I.sum(dim=0)  # / self.node_seq.shape[0]  # P @ x , P shape nxn, x shape nxc,
        if emb and self.learnEmb:
            I = self.Emb.backward(I)
        return I


class blur(nn.Module):
    def __init__(self, K, embdsize, learnEmb=True, device='cuda'):
        super(blur, self).__init__()
        K = K.to(device)
        nin = 3
        self.K = K
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb, device=device)

    def forward(self, I, emb=True):
        if emb:
            I = self.Emb(I)
        Ic = F.conv2d(I, self.K)

        return Ic

    def adjoint(self, Ic, emb=True):
        I = F.conv_transpose2d(Ic, self.K)
        if emb:
            I = self.Emb.backward(I)
        return I


class graph_smooth(nn.Module):
    def __init__(self, nin, embdsize, learnEmb=True, device='cuda', k=3):
        super(graph_smooth, self).__init__()
        self.nin = nin
        self.Emb = graphEmbed(embdsize, self.nin, learned=learnEmb, device=device)
        self.k = k
    def forward(self, node_features, edge_index, edge_weights, emb=True):
        # NOTE: "node_features" in this function is the target, graph.y 
        if emb:
            node_features = self.Emb(node_features)
        # node_features = F.conv2d(node_features, self.K)
        A = torch.zeros(node_features.shape[0], node_features.shape[0], device=node_features.device)
        A[edge_index[0, :], edge_index[1, :]] = edge_weights  # make faster

        node_features_smooth = node_features.clone()
        # for i in range(self.k):
        #     node_features_smooth = A @ node_features_smooth
        node_features_smooth = torch.linalg.matrix_power(A, self.k) @ node_features_smooth
        return node_features_smooth

    def adjoint(self, node_features, edge_index, edge_weights, emb=True):
        # I = F.conv_transpose2d(Ic, self.K)

        A = torch.zeros(node_features.shape[0], node_features.shape[0], device=node_features.device)
        A[edge_index[0, :], edge_index[1, :]] = edge_weights  # make faster

        # node_features = A.t()@(A.t()@((A.t() @ node_features))) #A.t() @ node_features


        for i in range(self.k):
            node_features = A.t() @ node_features

        if emb:
            node_features = self.Emb.backward(node_features)
        return node_features


class graph_edgeRecovery(nn.Module):
    def __init__(self, nin, embdsize, learnEmb=True, K=1, device='cuda'):
        super(graph_edgeRecovery, self).__init__()
        self.nin = nin
        self.Emb = graphEmbed(embdsize, self.nin, learned=learnEmb,device=device)
        self.K = K

    def forward(self, node_features, edge_index, edge_weights, emb=True):
        # xN' = P^k(xE)xN
        # xN = node_features
        if emb:
            node_features = self.Emb(node_features)
        # node_features = F.conv2d(node_features, self.K)

        A = torch.zeros(node_features.shape[0], node_features.shape[0], device=node_features.device)
        A[edge_index[0, :], edge_index[1, :]] = edge_weights  # make faster
        self.A = A
        node_features_smooth = node_features.clone()
        data_out = []
        for i in range(self.K):
            node_features_smooth = A @ node_features_smooth
            data_out.append(node_features_smooth)
        data_out = torch.stack(data_out, dim=0)  # [K,N,C]
        self.data_out = data_out
        self.edge_weights = edge_weights
        return data_out

    def adjoint(self, seq, node_features, edge_index, edge_weights, emb=True):
        # call forward

        # I = F.conv_transpose2d(Ic, self.K)
        N = len(edge_index.unique())
        C = node_features.shape[-1]
        K = self.K

        # grad(f)^T @ seq
        from torch.autograd import grad
        data_out = self.forward(node_features, edge_index, edge_weights, emb=emb)
        f = (data_out * seq).sum()
        #JtR = torch.autograd.functional.jvp(self.forward, (node_features, edge_index, edge_weights), seq)
        JtR = grad(f, edge_weights, create_graph=True)[0]
        JtR = torch.stack([JtR, JtR], dim=-1)
        # A = torch.zeros(node_features.shape[0], node_features.shape[0], device=node_features.device)
        # A[edge_index[0, :], edge_index[1, :]] = edge_weights  # make faster
        # A = self.A
        # node_features = A.t()@(A.t()@((A.t() @ node_features))) #A.t() @ node_features

        # for i in range(self.K):
        #    node_features = A.t() @ node_features

        if emb:
            JtR = self.Emb.backward(JtR)
        return JtR


class contactMap(nn.Module):
    def __init__(self, embdsize, sigma=1.0, device='cuda'):
        super(contactMap, self).__init__()
        self.sigma = sigma

    def forward(self, X, emb=True):
        Xsq = (X ** 2).sum(dim=1, keepdim=True)
        XX = Xsq + Xsq.transpose(1, 2)

        XTX = torch.bmm(X.transpose(2, 1), X)
        D = torch.relu(XX - 2 * XTX)

        return D

    def adjoint(self, X, dV):
        n1 = X.shape[-1]
        e2 = torch.ones(3, 1)
        e1 = torch.ones(n1, 1)
        E12 = e1 @ e2.t()
        E12 = E12.unsqueeze(0)
        E12 = torch.repeat_interleave(E12, X.shape[0], dim=0)

        P1 = 2 * X * (torch.bmm(dV, E12).transpose(-1, -2) + torch.bmm(dV.transpose(-1, -2), E12).transpose(-1, -2))
        P2 = 2 * torch.bmm(dV.transpose(-2, -1) + dV, X.transpose(-2, -1)).transpose(-2, -1)
        dX = P1 - P2

        return dX

    def jacMatVec(self, X, dX):
        XdX = torch.sum(X * dX, dim=-2, keepdim=True)
        XdXT = torch.bmm(X.transpose(-1, -2), dX)
        dXXT = torch.bmm(dX.transpose(-1, -2), X)
        V = 2 * XdX + 2 * XdX.transpose(-1, -2) - 2 * XdXT - 2 * dXXT
        return V


class blurFFT(nn.Module):
    def __init__(self, embdsize, nin, learnEmb=True, dim=256, device='cuda'):
        super(blurFFT, self).__init__()
        self.nin = nin
        self.Emb = Embed(embdsize, nin, learned=learnEmb)
        self.dim = dim
        self.device = device

    def forward(self, I, emb=True):
        if emb:
            I = self.Emb(I)
        P, center = self.psfGauss(self.dim)

        S = torch.fft.fft2(torch.roll(P, shifts=center, dims=[0, 1])).unsqueeze(0).unsqueeze(0)
        B = torch.real(torch.fft.ifft2(S * torch.fft.fft2(I)))

        return B

    def adjoint(self, Ic, emb=True):
        I = self.forward(Ic, emb=False)
        if emb:
            I = self.Emb.backward(I)
        return I

    def psfGauss(self, dim, s=[2.0, 2.0]):
        m = dim
        n = dim

        x = torch.arange(-n // 2 + 1, n // 2 + 1, device=self.device)
        y = torch.arange(-n // 2 + 1, n // 2 + 1, device=self.device)
        X, Y = torch.meshgrid(x, y)

        PSF = torch.exp(-(X ** 2) / (2 * s[0] ** 2) - (Y ** 2) / (2 * s[1] ** 2))
        PSF = PSF / torch.sum(PSF)

        # Get center ready for output.
        center = [1 - m // 2, 1 - n // 2]

        return PSF, center


class radonTransform(nn.Module):
    def __init__(self, embdsize, nin, learnEmb=True, device='cuda'):
        super(radonTransform, self).__init__()
        self.nin = nin
        self.Emb = graphEmbed(embdsize, nin, learned=learnEmb)
        self.device = device

        A = io.loadmat('radonMat18.mat')
        A = A['A']
        A = torch.tensor(A, device=device)
        A = A.type(torch.cuda.FloatTensor)
        A = A.to_sparse()
        self.A = A

    def forward(self, I, emb=True):
        if emb:
            I = self.Emb(I)

        T = I.view(I.shape[0], I.shape[1], -1)
        Tt = T.transpose(1, 2)
        Ttt = Tt.transpose(0, 1)
        Tttt = Ttt.reshape(Ttt.shape[0], -1)
        Yttt = torch.matmul(self.A, Tttt)
        Ytt = Yttt.reshape(Yttt.shape[0], -1, 3)
        Yt = Ytt.transpose(0, 1)
        Y = Yt.transpose(1, 2)
        Y = Y.reshape(Y.shape[0], 3, 18, 139)

        return Y

    def adjoint(self, Ic, emb=True):
        T = Ic.view(Ic.shape[0], Ic.shape[1], -1)
        Tt = T.reshape(-1, T.shape[2]).t()
        Yt = self.A.t() @ Tt
        Y = Yt.t()
        I = Y.reshape(-1, 3, 96, 96)
        if emb:
            I = self.Emb.backward(I)
        return I
