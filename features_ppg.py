import numpy as np
from scipy import integrate


def compute_mean_hr_features(window):
    return 0

def compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    all_means = np.mean(window, axis=0)
    mag = (all_means[0]**2 + all_means[1]**2 + all_means[2]**2)**0.5
    # return np.mean(window, axis=0)
    #return np.append(all_means, mag)
    return mag

def variance(window):
    all_var = np.var(window, axis = 0)
    mag = (all_var[0]**2+all_var[1]**2+all_var[2]**2)**0.5
    #return(np.append(all_var,mag))
    return mag

def extract_features(window):
    """
        Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """
    
    x = np.array([compute_mean_hr_features(window)])
    return x
