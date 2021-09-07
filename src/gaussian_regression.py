import numpy as np
import torch
from torch import tensor
from torch.nn import Module, Parameter
from torch.optim import Adam
import matplotlib.pyplot as plt

def objective_function(x):
    
    return 3 * np.cos(x-1) - np.abs(x-1)

def gaussian_kernel(data_x, data_x_prime, theta1, theta2):
    
    data_x = data_x.reshape((1, len(data_x), -1))
    data_x_prime = data_x_prime.reshape((1, len(data_x_prime), -1))
    data_x = data_x.transpose((2, 1, 0))
    data_x_prime = data_x_prime.transpose((2, 0, 1))
    r = ((data_x-data_x_prime) ** 2).sum(axis=0)
    
    return theta1 * np.exp(- r / theta2)
    
def gaussian_regression_f(data_x, data_y, x_domain, theta1, theta2):
    
    K = gaussian_kernel(data_x.copy(), data_x.copy(), theta1, theta2) #+ sig_y * np.eye(len(data_x))
    k_ = gaussian_kernel(data_x.copy(), x_domain.copy(), theta1, theta2)
    k__ = gaussian_kernel(x_domain.copy(), x_domain.copy(), theta1, theta2)
    kTKinv = k_.T @ np.linalg.inv(K)
    index = np.where(np.diag(k__ - np.dot(kTKinv, k_)) < 0)[0]
    sig = np.diag(k__ - np.dot(kTKinv, k_)).copy()
    if len(index) != 0:
        sig[np.array(index)] = 1
        
    sig = np.sqrt(sig)
    return (kTKinv @ np.array(data_y).reshape((len(data_y), 1))).reshape(-1), sig

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
        #print(K)
        yTK_inv = torch.mm(self.y.reshape(1, -1), K.inverse())
        
        return torch.log(K.det()) + torch.mm(yTK_inv, self.y).reshape(-1)
    
    def theta(self):
        
        return np.array(self.theta1.detach())[0], np.array(self.theta2.detach())[0]
    
def best_theta(data_x, data_y, n = 500):
    
    mll = minus_log_likelihood(data_x, data_y)
    optim = Adam(mll.parameters(), lr = 1/len(data_x))
    for i in range(n):
        
        optim.zero_grad()
        loss = mll()
        loss.backward()
        optim.step()
        
    return mll.theta()

def regression(n_observed, objectve_function = objective_function,
               min_x = -10., max_x = 10., n = 201):
    
    x_domain = np.linspace(min_x, max_x, n)
    obj_func = objective_function(x_domain)
    data_x = np.linspace(min_x, max_x, n_observed)
    data_y = objective_function(data_x)
    data_mu = data_y.mean()
    data_y -= data_mu
    theta1, theta2 = best_theta(data_x, data_y)
    mu, sig = gaussian_regression_f(data_x, data_y, x_domain, theta1, theta2)
    mu += data_mu
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.scatter(np.array(data_x).reshape(-1), data_y + data_mu, color = 'red', label = 'observed point')
    ax.plot(x_domain, mu, color = "blue", label = 'prediction mu')
    ax.plot(x_domain, obj_func, color = "black", label = 'objective function')
    ax.fill_between(x_domain, mu - 2*sig, mu + 2*sig, facecolor='orange', alpha=0.3 ,label = 'prediction 2*sig')
    ax.legend()
    plt.show()

def prediction(data_x, data_y, objectve_function = objective_function,
               min_x = -10., max_x = 10., n = 201):
    
    x_domain = np.linspace(min_x, max_x, n)
    data_x = np.array(data_x)
    data_mu = np.array(data_y).mean()
    data_y -= data_mu
    theta1, theta2 = best_theta(data_x, data_y)
    mu, sig = gaussian_regression_f(data_x, data_y, x_domain, theta1, theta2)
    mu += data_mu
    
    return mu, sig

def acquisition_function(mu, sig, n):
    
    return mu + np.sqrt(np.log(n+1)/(n+1)) * sig
    
def bayesian_optim(n_search, objectve_function = objective_function,
               min_x = -10., max_x = 10., n = 201):
    data_x = []
    data_y = []
    next_x = min_x + (max_x - min_x) * np.random.rand()
    x_domain = np.linspace(min_x, max_x, n)
    for i in range(n_search):
        print(next_x)
        if not next_x in data_x:
            data_x.append(next_x)
            data_y.append(objectve_function(next_x))
        
        
        
        mu, sig = prediction(data_x, data_y, objectve_function = objective_function,
               min_x = min_x, max_x = max_x, n = n)
        next_x = x_domain[np.argmax(acquisition_function(mu, sig, i+1))]
        if i == 0:
            next_x = min_x + (max_x - min_x) * np.random.rand()
        
    
    
    
    
    
    
    
    
    
    