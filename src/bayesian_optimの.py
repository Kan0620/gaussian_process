import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
from torch.nn import Module, Parameter
from torch.optim import Adam, SGD
from mpl_toolkits.mplot3d import Axes3D

'''hyper parameter'''


theta3 = 0.1
min_x = -5.0
max_x = 5.0
delta_x = 200
sig_y = 1e-10

def objective_func(x):
    #return x
    return 3 * np.cos(x-1) - np.abs(x-1) + x
    #return (x+10)*(x-1)*(x-5)*(x-8)/4000


def gaussian_kernel(data_x, data_x_prime, theta1 = 1.0, theta2 = 0.5):
    '''
    data_x = np.array(data_x)
    data_x_prime = np.array(data_x_prime)
    data_x = data_x.reshape((-1, len(data_x), 1))
    data_x_prime = data_x_prime.reshape((-1, 1, len(data_x_prime)))
    r = ((data_x-data_x_prime) ** 2).sum(axis=0)
    '''
    
    data_x = data_x.reshape((1, len(data_x), -1))
    data_x_prime = data_x_prime.reshape((1, len(data_x_prime), -1))
    data_x = data_x.transpose((2, 1, 0))
    data_x_prime = data_x_prime.transpose((2, 0, 1))
    r = ((data_x-data_x_prime) ** 2).sum(axis=0)
    delta = np.where(r == 0, 1, 0)
    
    return theta1 * np.exp(- r / theta2)
    #return np.exp(- np.sqrt(r) / theta2)
    #return np.exp(theta1 * np.cos(np.sqrt(r) / theta2))
    
'''
def gaussian_kernel(data_x, data_x_prime):
    
    n = len(data_x)
    m = len(data_x_prime)
    g = np.ones((n, m))
    
    data_x = np.array(data_x)
    data_x_prime = np.array(data_x_prime)
    data_x = data_x.reshape((len(data_x), -1))
    data_x_prime = data_x_prime.reshape((len(data_x_prime), -1))
    
    for i in range(n):
        
        for j in range(m):
            
            g[i, j] = theta1 * np.exp(-((data_x[i] - data_x_prime[j])**2).sum() / theta2)
            
    if n == m:
        
        print(g)
    return g
'''

def gaussian_regression_f(data_x, data_y):
    
    x_domain = np.linspace(min_x, max_x, delta_x)
    #x_domain = np.array([3])
    K = gaussian_kernel(data_x.copy(), data_x.copy()) #+ sig_y * np.eye(len(data_x))
    print(K.shape)
    k_ = gaussian_kernel(data_x.copy(), x_domain.copy())
    print(k_.shape)
    k__ = gaussian_kernel(x_domain.copy(), x_domain.copy())
    print(k__.shape)
    #try:
    #kTKinv = np.dot(k_.T, np.linalg.inv(K))
    kTKinv = k_.T @ np.linalg.inv(K)
    #except:
        #print(K)
        #kTKinv = k_.T
    #print(np.where(np.diag(k__ - np.dot(kTKinv, k_)) < 0, 1, 0))
    index = np.where(np.diag(k__ - np.dot(kTKinv, k_)) < 0)[0]
    sig = np.diag(k__ - np.dot(kTKinv, k_)).copy()
    print(index)
    if len(index) != 0:
        sig[np.array(index)] = 0
    sig = np.sqrt(sig)
    return (kTKinv @ np.array(data_y).reshape((len(data_y), 1))).reshape(-1), sig

def data2plot(n):
    data_x = np.linspace(min_x, max_x, n)
    data_y = objective_func(data_x)
    x_domain = np.linspace(min_x, max_x, delta_x)
    mu, sig = gaussian_regression_f(data_x, data_y)
    
    x_domain = np.linspace(min_x, max_x, delta_x)
    obj_func = objective_func(x_domain)
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.scatter(np.array(data_x).reshape(-1), data_y, color = 'red', label = 'observed point')
    ax.plot(x_domain, mu, color = "blue", label = 'prediction mu')
    ax.plot(x_domain, obj_func, color = "black", label = 'object function')
    ax.fill_between(x_domain, mu - 2*sig, mu + 2*sig, facecolor='orange', alpha=0.3 ,label = 'prediction 2*sig')
    ax.legend()

def acquisition_function(mu, sig, n):
    
    return mu + np.sqrt(np.log(n)/n) * sig
    
def next_x(x_domain, mu, sig, n):
    
    return x_domain[np.argmax(acquisition_function(mu, sig, n))]
    
def fit(n_search = 10):
    
    #x =(max_x - min_x) * np.random.rand() + min_x
    x = min_x
    y = objective_func(x)
    data_x = []
    data_y = []
    data_x.append(x)
    data_y.append(y)
    x_domain = np.linspace(min_x, max_x, delta_x)
    obj_func = objective_func(x_domain)
    print(x)
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    for n in range(n_search):
        
        mu, sig = gaussian_regression_f(data_x, data_y)
        x = next_x(x_domain, mu, sig, n+1)
        data_x.append(x)
        data_y.append(objective_func(x))
        
        
        print(x)
        
        
    print(data_x)
    print(data_y)
    ax.grid()
    ax.scatter(data_x, data_y, color = 'red', label = 'observed point')
    ax.plot(x_domain, mu, color = "blue", label = 'prediction mu')
    ax.fill_between(x_domain, mu - 2*sig, mu + 2*sig, facecolor='orange', alpha=0.3 ,label = 'prediction 2*sig')
    ax.plot(x_domain, obj_func, color = "black", label = 'object function')
    ax.legend()
    
        
def gaussian_process_():
    
    x_domain = np.linspace(min_x, max_x, delta_x)
    K = gaussian_kernel(x_domain.copy(), x_domain.copy())
    f = np.random.multivariate_normal(np.zeros(len(x_domain)), K, size=1).reshape(-1)
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.plot(x_domain, f, color = "blue", label = 'periodic_kernel')
    ax.legend()
    
    

class minus_log_likelihood(Module):
    
    def __init__(self, x, y):
        
        super(minus_log_likelihood,self).__init__()
        
        self.theta1 = Parameter(torch.Tensor([1.0]))
        self.theta2 = Parameter(torch.Tensor([1.0]))
        #self.theta3 = Parameter(torch.Tensor(1))
        x = x.reshape((1, len(x), -1))
        x_ = x.transpose((2, 1, 0))
        x__ = x.transpose((2, 0, 1))
        self.r = tensor(((x_-x__) ** 2).sum(axis=0), dtype = float)
        self.y = tensor(np.array(y).reshape(-1,1), dtype = float)
        
    def forward(self):
        
        K = self.theta1 * torch.exp(- self.r / self.theta2)
        yTK_inv = torch.mm(self.y.reshape(1, -1), K.inverse())
        
        return torch.log(K.det()) + torch.mm(yTK_inv, self.y).reshape(-1)
    
    def theta(self):
        
        return np.array(self.theta1.detach())[0], np.array(self.theta2.detach())[0]
        
        

def plot(n, n_epoch):
    
    data_x = np.linspace(min_x, max_x, n)
    data_y = objective_func(data_x)
    loss = minus_log_likelihood(data_x, data_y)
    print(loss.r)
    print(loss.theta())
    
    optim = Adam(loss.parameters(), lr=1/len(data_x))
    '''
    optim.zero_grad()
    loss().backward()
    optim.step()
    print(loss.theta())
    
    '''
    for epoch in range(n_epoch):
        
        optim.zero_grad()
        loss_ = loss()
        
        loss_.backward()
        optim.step()
        
        if epoch % 10 == 0:
            print(loss_)
            print(loss.theta())
            
    return loss.theta()
            
def objective_function2(x, y):
    
    return 1/(np.sqrt(x**2 + y**2) + 1)
    
def plot2dim():
    n = 50
    domain = np.linspace(min_x, max_x, n)
    x, y = np.meshgrid(domain, domain)
    domain = np.concatenate([x.reshape((-1,1)), y.reshape((-1,1))], axis = 1)
    K = gaussian_kernel(domain.copy(), domain.copy(), theta1 = 1., theta2 = 1)
    print(K.shape)
    f = np.random.multivariate_normal(np.zeros(n**2), K, size=1).reshape((n, n))
    fig = plt.figure(figsize = (8, 6))
    ax = Axes3D(fig)
    #ax.plot_wireframe(x, y, f, color='blue',linewidth=0.1)
    ax.plot_surface(x, y, f, rstride=1, cstride=1, cmap='bwr', linewidth=0.01, shade=True)
    

def best_theta(data_x, data_y, n = 50):
    
    mll = minus_log_likelihood(data_x, data_y)
    optim = Adam(mll.parameters(), lr = 1/len(data_x))
    
    for i in range(n):
        
        optim.zero_grad()
        loss = mll()
        loss.backward()
        optim.step()
        print(-loss.item())
        
    return mll.theta()


    
def reg2():
    
    n_domain = 30
    domain = np.linspace(min_x, max_x, n_domain)
    domain_x, domain_y = np.meshgrid(domain, domain)
    domain_z = objective_function2(domain_x, domain_y)
    domain_xy = np.concatenate([domain_x.reshape((-1,1)), domain_y.reshape((-1,1))], axis = 1)
    n_data = 3
    data = np.linspace(min_x, max_x, n_data)
    print(data)
    data_x, data_y = np.meshgrid(data, data)
    data_xy = np.concatenate([data_x.reshape((-1,1)), data_y.reshape((-1,1))], axis = 1)
    data_z = objective_function2(data_x, data_y).reshape((-1,1))
    theta1, theta2 = best_theta(data_xy, data_z)
    
    print(theta1, theta2)
    #theta1, theta2 = [1, 10]
    
    K = gaussian_kernel(data_xy.copy(), data_xy.copy(), theta1, theta2) #+ sig_y * np.eye(len(data_x))
    k_ = gaussian_kernel(data_xy.copy(), domain_xy.copy(), theta1, theta2)
    kTKinv = k_.T @ np.linalg.inv(K)
    mu = (kTKinv @ np.array(data_z).reshape((len(data_z), 1))).reshape((n_domain, n_domain))
    print(K)
    fig = plt.figure(figsize = (8, 6))
    ax = Axes3D(fig)
    ax.plot_wireframe(domain_x, domain_y, domain_z, color='green',linewidth=0.5)
    ax.plot_wireframe(domain_x, domain_y, mu, color='blue',linewidth=0.5)
    #ax.plot_surface(domain_x, domain_y, mu, rstride=1, cstride=1, cmap='bwr', linewidth=0.01, shade=True)

    






