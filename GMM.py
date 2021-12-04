import numpy as np
import random
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

def generate_initial_parameters(data, k):
    N, D = np.shape(data)[0], np.shape(data)[1]
    Pi = np.ones(k)/k
    d_means = np.zeros(shape=(k,D))
    
    initials = np.random.randint(0, N, k)

    for ind in range(len(initials)):
        d_means[ind,:] = data[ind,:]

    covariances = [np.eye(D) for i in range(k)]

    return d_means, covariances, Pi

#==========================================================================

def E_step(data, k, d_means, covariances, Pi):
    N = np.shape(data)[0]
    gamma = np.zeros(shape=(N,k))
    for i in range(k):
        gamma[:,i] = Pi[i]*multivariate_normal.pdf(data, d_means[i],covariances[i])
    for n in range(N):
        z = np.sum(gamma[n,:])
        gamma[n,:] = gamma[n,:]/z
    return gamma

#==========================================================================

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
    
#==========================================================================
#==========================================================================

def GMM(data, k, threshold, iterations):

    means, covariances, Pk = generate_initial_parameters(data, k)

    threshold = threshold
    max_iter = iterations
    LogLikelihood = np.zeros(max_iter)
    loglh = 0

    for i in range(max_iter):
        
        gamma = E_step(data, k, means, covariances, Pk)

        LogLikelihood[i] = sum(np.log(np.max(gamma, axis=1)))
        loglh = LogLikelihood[i]

        if loglh >= threshold:
            print("Convergence is met in round {}".format(i))
            break

        means, covariancess, Pk = M_step(data, k, gamma)

    return means, np.argmax(gamma, axis=1), loglh, LogLikelihood


