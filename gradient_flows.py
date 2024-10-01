#this code is inspired by https://github.com/kimiandj/gsw/blob/master/code/
import numpy as np

import torch
from torch import optim
import ot
from scipy.stats import ortho_group, random_correlation

from partial import partial_ot_1d


from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_blobs,make_spd_matrix
from tqdm import trange


def w2(X,Y):
    M = ot.dist(X,Y)
    a = np.ones((X.shape[0],))/X.shape[0]
    b = np.ones((Y.shape[0],))/Y.shape[0]
    return ot.emd2(a, b, M)
   


def load_data(name='swiss_roll', n_samples=1000,dim=2):
    N=n_samples
    if name == 'gaussian' :
        mu_s = np.random.randint(-10,-1,dim)
        cov_s = np.diag(np.random.randint(1,10,dim))
        cov_s = cov_s * np.eye(dim)
        temp = np.random.multivariate_normal(mu_s, cov_s, n_samples)
    elif name == 'gaussian_2d':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1))+4 for i in mu_s]) 
        #cov_s = cov_s * np.eye(2)
        cov_s = np.array([[0.5,-2], [-2, 5]])
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_2d_with_noise':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1))+4 for i in mu_s]) 
        #cov_s = cov_s * np.eye(2)
        cov_s = np.array([[0.5,-2], [-2, 5]])
        temp=np.random.multivariate_normal(mu_s, cov_s, int(N*0.8))
        noise = np.random.rand(N - int(N*0.8),2)*20 - 10
        temp = np.concatenate((temp, noise))
        #temp/=abs(temp).max()
    elif name == 'gaussian_2d_small_v':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((2, 2))*1
        cov_s = cov_s * np.eye(2)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_2d_big_v':
        mu_s = np.ones(2)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = random_correlation.rvs((.2, 1.8))*2
        #cov_s = cov_s * np.eye(2)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d_small_v':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((500, 500))*1
        cov_s = cov_s * np.eye(500)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d_big_v':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        eigs = np.random.rand(500,1)*2
        eigs = eigs / np.sum(eigs) * 500
        rr = eigs.reshape(-1)
        cov_s = random_correlation.rvs(rr)*10
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
        #temp/=abs(temp).max()
    elif name == 'gaussian_500d':
        mu_s = np.ones(500)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = np.ones((500, 500))*50
        cov_s = cov_s * np.eye(500)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'gaussian_500d_spd_cov':
        dim = 500
        mu_s = np.ones(dim)
        mu_s = np.array([i*float(np.random.rand(1,1)) for i in mu_s]) 
        cov_s = make_spd_matrix(dim, random_state=3)
        temp=np.random.multivariate_normal(mu_s, cov_s, N)
    elif name == 'swiss_roll':
        temp=make_swiss_roll(n_samples=N)[0][:,(0,2)]
        temp/=abs(temp).max()
    elif name == 'half_moons':
        temp=make_moons(n_samples=N)[0]
        temp/=abs(temp).max()
    elif name == '8gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        scale = 2.
        centers = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)), (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        temp = []
        for i in range(N):
            point = np.random.randn(2) * .02
            center = centers[np.random.choice(np.arange(len(centers)))]
            point[0] += center[0]
            point[1] += center[1]
            temp.append(point)
        temp = np.array(temp, dtype='float32')
        temp /= 1.414  # stdev
    elif name == '25gaussians':
        # Inspired from https://github.com/caogang/wgan-gp
        temp = []
        for i in range(int(N / 25)):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    temp.append(point)
        temp = np.array(temp, dtype='float32')
        np.random.shuffle(temp)
        temp /= 2.828  # stdev
    elif name == 'circle':
        temp,y=make_circles(n_samples=2*N)
        temp=temp[np.argwhere(y==0).squeeze(),:]
    else:
        raise Exception("Dataset not found: name must be 'gaussian_2d', 'gaussian_500d', swiss_roll', 'half_moons', 'circle', '8gaussians' or '25gaussians'.")
    X=torch.from_numpy(temp).float()
    return X




class GF():
    def __init__(self, ftype='linear', nofprojections=10, device='cpu'):
        self.ftype=ftype
        self.nofprojections=nofprojections
        self.device=device
        self.theta=None 

    def sw(self,X,Y,theta=None):
        N,dn = X.shape
        M,dm = Y.shape
        assert dn==dm and M==N
        if theta is None:
            theta=self.random_slice(dn)

        Xslices=self.get_slice(X,theta)
        Yslices=self.get_slice(Y,theta)

        Xslices_sorted=torch.sort(Xslices,dim=0)[0]
        Yslices_sorted=torch.sort(Yslices,dim=0)[0]
        return torch.sum((Xslices_sorted-Yslices_sorted)**2)
    
    
    def sw_partial(self,X,Y,theta=None):
        N,dn = X.shape
        M,dm = Y.shape
        if theta is None:
            theta=self.random_slice(dn).T
        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)
        cost = 0
        for i in range(theta.shape[1]):
            print(i)
            x_proj, y_proj = X_line[:,i].detach().numpy(), Y_line[:,i].detach().numpy()
            x_proj, y_proj = np.array(x_proj, dtype=np.float64), np.array(y_proj, dtype=np.float64)
            ind_x, ind_y, _ = partial_ot_1d(x_proj, y_proj, 80)
            sorted_ind_x, sorted_ind_y = np.sort(ind_x), np.sort(ind_y)
            X_s, Y_s = X[sorted_ind_x], Y[sorted_ind_y]
            cost += torch.sum(torch.abs(X_s - Y_s))
        print(cost)
        return cost
    
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


    def swgg_partial(self,X,Y,theta):
        n,dn=X.shape
        if theta is None:
            theta=self.random_slice(dn).T

        min_cost = np.inf
        best_w = None   
        X_line = torch.matmul(X, theta)
        Y_line = torch.matmul(Y, theta)
        for i in range(theta.shape[1]):
            x_proj, y_proj = X_line[:,i].detach().numpy(), Y_line[:,i].detach().numpy()
            x_proj, y_proj = np.array(x_proj, dtype=np.float64), np.array(y_proj, dtype=np.float64)
            ind_x, ind_y, _ = partial_ot_1d(x_proj, y_proj, 80)
            sorted_ind_x, sorted_ind_y = np.sort(ind_x), np.sort(ind_y)
            X_s, Y_s = X[sorted_ind_x], Y[sorted_ind_y]
            cost = torch.sum(torch.abs(X_s - Y_s))
            if cost < min_cost:
                min_cost = cost
                best_w = theta[:,i]
        return min_cost, best_w
        


##### GRADIENT DESCENT ######
    def SWGG_smooth(self,X,Y,theta,s=1,std=0):
        n,dim=X.shape
    
        X_line=torch.matmul(X,theta)
        Y_line=torch.matmul(Y,theta)
    
        X_line_sort,u=torch.sort(X_line,axis=0)
        Y_line_sort,v=torch.sort(Y_line,axis=0)
    
        X_sort=X[u]
        Y_sort=Y[v]
    
        Z_line=(X_line_sort+Y_line_sort)/2
        Z=Z_line[:,None]*theta[None,:]
    
        W_XZ=torch.sum((X_sort-Z)**2)/n
        W_YZ=torch.sum((Y_sort-Z)**2)/n
    
        X_line_extend = X_line_sort.repeat_interleave(s,dim=0)
        X_line_extend_blur = X_line_extend + 0.5 * std * torch.randn(X_line_extend.shape,device=self.device)
        Y_line_extend = Y_line_sort.repeat_interleave(s,dim=0)
        Y_line_extend_blur = Y_line_extend + 0.5 * std * torch.randn(Y_line_extend.shape,device=self.device)
    
        _,u_b=torch.sort(X_line_extend_blur,axis=0)
        _,v_b=torch.sort(Y_line_extend_blur,axis=0)

        X_extend=X_sort.repeat_interleave(s,dim=0)
        Y_extend=Y_sort.repeat_interleave(s,dim=0)
        X_sort_extend=X_extend[u_b]
        Y_sort_extend=Y_extend[v_b]
    
        bary_extend=(X_sort_extend+Y_sort_extend)/2
        bary_blur=torch.mean(bary_extend.reshape((n,s,dim)),dim=1)
    
        W_baryZ=torch.sum((bary_blur-Z)**2)/n
        return -4*W_baryZ+2*W_XZ+2*W_YZ

    def get_minSWGG_smooth(self,X,Y,lr=1e-2,num_iter=100,s=1,std=0,init=None):
        if init is None :
             theta=torch.randn((X.shape[1],), device=X.device, dtype=X.dtype,requires_grad=True)
        else :
            theta=torch.tensor(init,device=X.device, dtype=X.dtype,requires_grad=True)
        
        #optimizer = torch.optim.Adam([theta], lr=lr) #Reduce the lr if you use Adam
        optimizer = torch.optim.SGD([theta], lr=lr)
        loss_l=torch.empty(num_iter)
        for i in range(num_iter):
            theta.data/=torch.norm(theta.data)
            optimizer.zero_grad()
            
            loss = self.SWGG_smooth(X,Y,theta,s=s,std=std)
            loss.backward()
            optimizer.step()
        
            loss_l[i]=loss.data
            #proj_l[i,:]=theta.data
        res=self.SWGG_smooth(X,Y,theta.data.float(),s=1,std=0)
        return res,theta.data, loss_l#,proj_l
    

    def max_sw(self,X,Y,iterations=500,lr=1e-4):
        N,dn = X.shape
        M,dm = Y.shape
        device = self.device
        assert dn==dm and M==N
#         if self.theta is None:
        if self.ftype=='linear':
            theta=torch.randn((1,dn),device=device,requires_grad=True)
            theta.data/=torch.sqrt(torch.sum((theta.data)**2))
        self.theta=theta
        optimizer=optim.Adam([self.theta],lr=lr)
        loss_l=[]
        for i in range(iterations):
            optimizer.zero_grad()
            loss=-self.sw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
            #print('test4')
            loss_l.append(loss.data)
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data/=torch.norm(self.theta.data)
            #print('test5')

        res = self.sw(X.to(self.device),Y.to(self.device),self.theta.to(self.device))
        return res,self.theta.to(self.device).data,loss_l


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