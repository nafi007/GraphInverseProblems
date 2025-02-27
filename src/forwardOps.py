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


class Embed(nn.Module):
    def __init__(self, embdsize, nin=3, learned=True, device='cuda'):
        super(Embed, self).__init__()
        if learned:
            self.k = 9
            self.Emb = (0 * nn.init.xavier_uniform_(torch.empty(nin, embdsize, self.k, self.k)))
            id = torch.zeros((self.k, self.k))
            id[self.k // 2, self.k // 2] = 1
            self.Emb[0, 0, :, :] = id
            self.Emb[1, 1, :, :] = id
            self.Emb[2, 2, :, :] = id

            self.Emb = nn.Parameter(self.Emb)
        else:
            self.Emb = torch.eye(3, 3, device=device).unsqueeze(-1).unsqueeze(-1)

    def forward(self, I):
        Emb = self.Emb.to(I.device)
        return F.conv2d(I, Emb, padding=self.Emb.shape[-1] // 2)

    def backward(self, I):
        Emb = self.Emb.to(I.device)

        return F.conv_transpose2d(I, Emb, padding=self.Emb.shape[-1] // 2)


class maskImage(nn.Module):
    def __init__(self, ind, imsize, embdsize, nin=3, device='cuda', learnEmb=True):
        super(maskImage, self).__init__()
        ind = ind.to(device)
        self.ind = ind
        self.imsize = imsize
        self.Emb = Embed(embdsize, nin, learned=learnEmb)
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


class blur(nn.Module):
    def __init__(self, K, embdsize, learnEmb=True, device='cuda'):
        super(blur, self).__init__()
        K = K.to(device)
        nin = 3
        self.K = K
        self.Emb = Embed(embdsize, nin, learned=learnEmb)

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
        self.Emb = Embed(embdsize, nin, learned=learnEmb)
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
