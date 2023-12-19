import numpy as np
from src.config.config import Parameters
import torch
import torch.autograd as autograd

def energy_calculation(arr):
	return np.max(arr, axis = 1) - np.min(arr, axis = 1)

def current_calculation(arr):
	arr_diff = np.diff(arr, axis=1)
	return np.max(arr_diff, axis = 1) - np.min(arr_diff, axis = 1)

def drifttime_calculation(arr):
    arr = (arr - np.min(arr, axis=1).reshape(arr.shape[0], 1)) / (np.max(arr, axis=1) - np.min(arr, axis=1)).reshape(arr.shape[0], 1)
    drifttimes = []
    for j in range(arr.shape[0]):
        m = max(arr[j])
        idx = list(arr[j]).index(m) #index of a max element
        for i in range(idx):
            if (arr[j][i] - 0.9*m) <= 0.01:
                k90 = i
            if (arr[j][i] - 0.1*m) <= 0.01:
                k10 = i
        drifttimes.append(k90 - k10)
    drifttimes = np.array(drifttimes)
    return drifttimes

def tail_slope_calculation(arr):
    arr = arr - np.min(arr, axis=1).reshape(arr.shape[0], 1)
    slopes = []
    for i in range(arr.shape[0]):
        m = max(arr[i])
        idx = list(arr[i]).index(m) #index of a max element
        s = 0
        for j in range(idx - 50, idx - 20):
            s += arr[i][j]
        avg1 = s / 30 #1st y-coordinate

        s = 0
        for j in range(idx + 1000, idx + 2000):
            s += arr[i][j]
        avg2 = s / 1000 #2nd y-coordinate

        x = (idx + 1500) - (idx - 35)
        y = abs(avg2 - avg1)
        slopes.append(x / y)
    return np.array(slopes)

def calculate_iou(h1,h2, bins=50):
    '''
    Calculate the histogram intersection over union (IOU)
    '''
    rg = np.histogram(np.hstack((h1, h2)), bins=bins)[1]
    
    h1 = np.array(h1)
    h2 = np.array(h2)
    
    count, _ = np.histogram(h1,bins=rg,density=True)
    count2, _ = np.histogram(h2,bins=rg,density=True)
    
    intersection = np.sum(np.minimum(count,count2))
    union = np.sum(np.maximum(count,count2))
        
    return intersection/union*100.0

# functions for wgan
def inf_train_gen(train_loader):
    while True:
        for images,targets in train_loader:
            yield images

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, LAMBDA):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
