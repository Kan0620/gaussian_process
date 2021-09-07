import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def kernel(data_x, data_x_prime, kernel_mode, theta1 = 1.0, theta2 = 1.0):
    
    data_x = data_x.reshape((1, len(data_x), -1))
    data_x_prime = data_x_prime.reshape((1, len(data_x_prime), -1))
    data_x = data_x.transpose((2, 1, 0))
    data_x_prime = data_x_prime.transpose((2, 0, 1))
    r = ((data_x-data_x_prime) ** 2).sum(axis=0)
    if kernel_mode == 'gaussian':
        
        return theta1 * np.exp(- r / theta2)
    
    if kernel_mode == 'exponetial':
        
        return np.exp(- np.sqrt(r) / theta2)
    
    if kernel_mode == 'periodic':
        
        return np.exp(theta1 * np.cos(np.sqrt(r) / theta2))
    
def sample_1d(min_x = -5., max_x = 5., n = 101):
    
    x_domain = np.linspace(min_x, max_x, n)
    fig = plt.figure(figsize = (8, 16))
    for i, kernel_mode in enumerate(['gaussian', 'exponetial', 'periodic']):
        K = kernel(x_domain.copy(), x_domain.copy(), kernel_mode)
        f = np.random.multivariate_normal(np.zeros(len(x_domain)), K, size=1).reshape(-1)
        ax = fig.add_subplot(3, 1, i+1)
        ax.set_title('kernel: {}'.format(kernel_mode), fontsize=10)
        ax.grid()
        ax.plot(x_domain, f, color = "blue")
    
    plt.show()
    
def sample_2d(kernel_mode, min_x = -5, max_x = 5, n = 51):
    
    domain = np.linspace(min_x, max_x, n)
    xx, yy = np.meshgrid(domain, domain)
    domain = np.concatenate([xx.reshape((-1,1)), yy.reshape((-1,1))], axis = 1)
    fig = plt.figure(figsize = (10, 6))
    K = kernel(domain.copy(), domain.copy(), kernel_mode = kernel_mode)
    f = np.random.multivariate_normal(np.zeros(n**2), K, size=1).reshape((n, n))
    ax = Axes3D(fig)
    ax.set_title('kernel: {}'.format(kernel_mode), fontsize=12)
    ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='bwr', linewidth=0.01, shade=True)
    plt.show()