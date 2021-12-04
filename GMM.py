import numpy as np
import random
from scipy.stats import multivariate_normal
from KMeans import *

#===============================================================================================================
#===============================================================================================================

def generate_means(data, k):
    N, D = data.shape[0], data.shape[1]

    init_means = np.zeros(shape=(k,D))
    initials = np.random.randint(0, N, k)

    for ind in range(len(initials)):
        init_means[ind,:] = data[ind,:]
    return init_means

def generate_sigma(data, k):
    D = data.shape[1]
    covariances = [np.eye(D) for i in range(k)]

    return covariances

def generate_P(data, k):
    Pi = np.ones(k)/k
    
    return Pi

#===============================================================================================================

def E_step(data, k, d_means, covariances, Pi):
    N = np.shape(data)[0]
    gamma = np.zeros(shape=(N,k))
    for i in range(k):
        gamma[:,i] = Pi[i]*multivariate_normal.pdf(data, d_means[i],covariances[i])
    for n in range(N):
        z = np.sum(gamma[n,:])
        gamma[n,:] = gamma[n,:]/z
    return gamma

#===============================================================================================================

def M_step(data, k, gamma):

    N, D = data.shape[0], data.shape[1]
    new_cov = []
    numerator = (gamma.T @ data)
    denominator = (np.sum(gamma, axis=0)).reshape(k,1)
    new_means = numerator/denominator
    new_Pi = np.sum(gamma, axis=0)/N
    
    for i in range(k):

        subtracted_data = (data-new_means[i,:])
        cov = ((gamma[:,i].reshape(N,1).T* subtracted_data.T) @ subtracted_data)/denominator[i]
        if cov.shape == (D,D):
            #print(cov)
            new_cov.append(cov)
    
    return new_means, new_cov, new_Pi

#===============================================================================================================

def GaussianMixtureModel(data, k, threshold, iterations, init_method):

    if init_method == "random":
        means = generate_means(data, k)
        covariances = generate_sigma(data, k)
        Pk = generate_P(data, k)
        print("Random initialization of parameters")
    
    if init_method == "kmeans":
        means, clusters_ids, score, stand_dev, conv = KMeans(data, k, "euclidean", 1e-16, 100)
        covariances = generate_sigma(data, k)
        Pk = generate_P(data, k)
        print("KMeans is running to initialize parameters")
    
    else:
        raise ValueError("Input Error. Init should be random or kmeans")

    threshold = threshold
    max_iter = iterations
    LogLikelihood = []
    loglh = 0
    error = 0

    for i in range(max_iter):

        gamma = E_step(data, k, means, covariances, Pk)

        loglh = sum(np.log(np.max(gamma, axis=1)))
        LogLikelihood.append(loglh)

        if loglh == LogLikelihood[i-1]:
            error +=1
            if error >= 2:
                break
        
        if loglh >= threshold:
            print("Convergence is met in round {}".format(i))
            break

        if (i+1) == max_iter:
            print("GaussianMixtureModel() has reached the maximum iterations")
        
        means, covariances, Pk = M_step(data, k, gamma)

        return means, covariances, np.argmax(gamma, axis=1), loglh, LogLikelihood

#===============================================================================================================
#===============================================================================================================
