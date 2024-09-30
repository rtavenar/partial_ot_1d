#this code is inspired by https://github.com/kimiandj/gsw/blob/master/code/
import numpy as np

import torch
import ot

from partial_nb import partial_ot_1d
from partial_nb import partial_ot_1d_elbow



class GF():
    def __init__(self, ftype='linear', nofprojections=10, device='cpu', seed=0):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.device=device
        self.theta=None 
        torch.manual_seed(seed)


    def sw(self,X,Y,theta=None):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices = self.get_slice(X,theta)
        Yslices = self.get_slice(Y,theta)

        Xslices_sorted = torch.sort(Xslices,dim=0)[0]
        Yslices_sorted = torch.sort(Yslices,dim=0)[0]
        return torch.sum((Xslices_sorted-Yslices_sorted)**2)
    
    
    def sw_partial(self,X,Y,theta=None, k = -1):
        n,dn=X.shape
        m,_=Y.shape
        if theta is None:
            theta=self.random_slice(dn).T
        if k==-1:
            k = min(n, m)
        if theta is None:
            theta=self.random_slice(dn).T
        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)
        cost = 0
        for i in range(theta.shape[1]):
            x_proj, y_proj = X_line[:,i].detach().numpy(), Y_line[:,i].detach().numpy()
            x_proj, y_proj = np.array(x_proj, dtype=np.float64), np.array(y_proj, dtype=np.float64)
            ind_x, ind_y, _ = partial_ot_1d(x_proj, y_proj, k)
            sorted_ind_x, sorted_ind_y = np.sort(ind_x), np.sort(ind_y)
            X_s, Y_s = X[sorted_ind_x], Y[sorted_ind_y]
            cost += torch.sum(torch.abs(X_s - Y_s))

        return cost/(theta.shape[1]*k), X_s, Y_s
    
    
    def swgg(self,X,Y,theta):
        dn = X.shape[1]
        if theta is None:
            theta=self.random_slice(dn).T
        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)
        _, u = torch.sort(X_line, axis=0)
        _, v = torch.sort(Y_line, axis=0)
        W = torch.mean(torch.sum(torch.square(X[u]-Y[v]), axis=-1), axis=0)
        idx = torch.argmin(W)
        return W[idx],theta[:,idx]


    def swgg_partial(self,X,Y,theta,k=-1):
        n,dn=X.shape
        m,_=Y.shape
        if theta is None:
            theta=self.random_slice(dn).T
        if k==-1:
            k = min(n, m)
        min_cost = np.inf
        best_w = None   
        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)
        best_x = None
        best_y = None
        for i in range(theta.shape[1]):
            x_proj, y_proj = X_line[:,i].detach().numpy(), Y_line[:,i].detach().numpy()
            x_proj, y_proj = np.array(x_proj, dtype=np.float64), np.array(y_proj, dtype=np.float64)
            ind_x, ind_y, _ = partial_ot_1d(x_proj, y_proj, k)
            sorted_ind_x, sorted_ind_y = np.sort(ind_x), np.sort(ind_y)
            X_s, Y_s = X[sorted_ind_x], Y[sorted_ind_y]
            cost = torch.sum(torch.abs(X_s - Y_s))
            if cost < min_cost:
                min_cost = cost
                best_w = theta[:,i]
                best_X = X_s
                best_Y = Y_s
        return min_cost/k, best_w, best_X, best_Y
        

    def swgg_partial_elbow(self,X,Y,theta,k=-1,s=0.01):
        n,dn=X.shape
        m,_=Y.shape
        if theta is None:
            theta=self.random_slice(dn).T
        if k==-1:
            k = min(n, m) 
        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)
        best_x = None
        best_y = None
        min_cost = np.inf
        best_w = None  
        elbow = 0
        for i in range(theta.shape[1]):
            x_proj, y_proj = X_line[:,i].detach().numpy(), Y_line[:,i].detach().numpy()
            x_proj, y_proj = np.array(x_proj, dtype=np.float64), np.array(y_proj, dtype=np.float64)
            ind_x, ind_y, _, e = partial_ot_1d_elbow(x_proj, y_proj, s)
            sorted_ind_x, sorted_ind_y = np.sort(ind_x[:e]), np.sort(ind_y[:e])
            X_s, Y_s = X[sorted_ind_x], Y[sorted_ind_y]
            cost = torch.sum(torch.abs(X_s - Y_s))
            if cost < min_cost:
                min_cost = cost
                best_w = theta[:,i]
                best_X = X_s
                best_Y = Y_s
                elbow = e
        return min_cost/k, best_w, best_X, best_Y, elbow


    def get_slice(self,X,theta):
        ''' Slices samples from distribution X~P_X
            Inputs:
                X:  Nxd matrix of N data samples
                theta: parameters of g (e.g., a d vector in the linear case)
        '''
        if self.ftype=='linear':
            return self.linear(X,theta)

    def random_slice(self,dim):
        if self.ftype=='linear':
            theta=torch.randn((self.nofprojections,dim))
            theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta])
        return theta.to(self.device)

    def linear(self,X,theta):
        if len(theta.shape)==1:
            return torch.matmul(X,theta)
        else:
            return torch.matmul(X,theta.t())