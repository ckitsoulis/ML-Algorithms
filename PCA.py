import numpy as np

#==========================================================================================
#==========================================================================================

def Conventional_PCA(data, k, var, method = 'x'): #input: data, number of PCs, selected variance, method (eigen or svd)
    
    if method == "eigen":
        print("Method applied: Eigen Decomposition")
        #Substract the mean of each feature(variable) from the data matrix (n x m)
        data_mean = np.mean(data, axis = 0)
        data_std = np.std(data, axis = 0)
        substracted_matrix = (data - data_mean)/data_std
        
        #Calculate the covariance matrix S, scaled by 1/(n-1) (m x m)
        covariance_S = np.cov(substracted_matrix , rowvar = False)
        
        #Eigen decomposition of covariance matrix by descending order
        eigenvalues, eigenvectors = np.linalg.eig(covariance_S)
        sorted_index = np.argsort(eigenvalues)[::-1]
        
        Λ = eigenvalues[sorted_index]
        Q = eigenvectors[:,sorted_index]
        loadings = Q*np.sqrt(Λ)
        
        #Calculate the variance explained by sorted eigenvalues of covariance matrix
        variance_explained = Λ/np.sum(Λ)
        cumulative_variance = np.cumsum(variance_explained)
        
        #Compute the optimal k for selected variance
        optimal = np.where(cumulative_variance >= var)[0][0] + 1
        
        #Project the data keeping only the k selected principal components of eigenvectors (Q)
        X_projected = np.dot(substracted_matrix,Q[:,:optimal])
        
    
    elif method == "svd":
        print("Method applied: Singular Value Decomposition")
        #substract the mean of each feature(variable) from the data matrix (n x m)
        data_mean = np.mean(data, axis = 0)
        data_std = np.std(data, axis = 0)
        substracted_matrix = (data - data_mean)/data_std
        
        #singular value decomposition of substracted matrix
        U, S, V_t = np.linalg.svd(substracted_matrix, full_matrices=True)
        
        #Compute the eigenvalues from the singular values matrix (S), scaled by 1/(n-1), variance explained and loadings
        l = np.power(S,2)/(substracted_matrix.shape[0]-1)
        variance_explained = l/np.sum(l)
        cumulative_variance = np.cumsum(variance_explained)
        loadings = V_t.T*np.sqrt(l)
        
        #Compute the optimal k for selected variance
        optimal = np.where(cumulative_variance >= var)[0][0] + 1
        
        #Project the data keeping only the k selected principal components of eigenvectors (Q)
        X_projected = np.dot(substracted_matrix, V_t.T[:,:optimal])
    
    else:
        raise Exception("WARNING: Check again the input. Method must be eigen or svd.")
     
    
    return "optimal k = {}".format(optimal), X_projected, variance_explained, loadings
  
#==========================================================================================
#==========================================================================================
