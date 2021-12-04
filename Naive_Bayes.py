import numpy as np
from scipy.stats import norm

#========================================================================================================
#========================================================================================================

def train_NBC(X, Xdtype, Y, L = 0, D_categorical = None):
    
    Xdtype = Xdtype.iloc[:,-1].tolist()

    n_samples, n_features = X.shape[0], X.shape[1]

    classes = np.unique(Y)
    
    class_split = dict()
    for c in classes:
        class_split[int(c)] = X[np.where(Y == c)[0],:]


    if all([xtype == "discrete" for xtype in Xdtype]):
        
        D = list(set(D_cat.to_numpy().T[0]))[0]
        
        parameters = dict()

        for c in class_split.keys():
            thetas = np.zeros(n_features)

            for value in range(D):
                theta = (np.count_nonzero(class_split[int(c)] == value, axis = 0) + L)/ (len(class_split[int(c)]) + L*D)
                thetas = np.vstack([thetas,theta])
            
            pi = len(np.where(Y == c)[0]) / ( len(Y) + (L*len(classes)) )
            
            parameters[int(c)] = (np.delete(thetas, 0, axis = 0), pi)

        return parameters
    
    if all([xtype == "continuous" for xtype in Xdtype]):
        
        estimates = dict()

        for c in class_split.keys():

            mu = np.mean(class_split[int(c)], axis = 0)
            sigma = np.std(class_split[int(c)], axis = 0)
            
            estimates[int(c)] = (mu, sigma, len(np.where(Y == c)[0]) / len(Y))
        
        return estimates

#========================================================================================================
#========================================================================================================

def predict_NBC(model, X, Xdtype):

    Xdtype = Xdtype.iloc[:,-1].tolist()

    if all([xtype == "discrete" for xtype in Xdtype]):
        
        predictions = []

        def posteriors(model, _array):
            
            posteriors = dict()
            
            for class_ in model.keys():
                
                posterior = 1
            
                for index, value in enumerate(_array):
                    
                    posterior = posterior * model[int(class_)][0][int(value), index]
            
                posteriors[class_] = posterior*model[int(class_)][1]
            
            return max(posteriors, key = lambda x: posteriors[x])

        
        for sample in range(len(X)):
            predictions.append(posteriors(model, X[sample, :]))

        return np.array(predictions)
    
    
    if all([xtype == "continuous" for xtype in Xdtype]):
        
        predictions = []

        def Gaussian_pdf(model, data):
            
            gaussians = dict()

            for class_ in model.keys():

                Gaussian = norm(model[int(class_)][0], model[int(class_)][1])
                P_class = model[int(class_)][2]

                gaussians[class_] = np.prod(Gaussian.pdf(data))*P_class

            return max(gaussians, key = lambda x: gaussians[x])
        
        for sample in range(len(X)):
            predictions.append(Gaussian_pdf(model, X[sample, :]))
        
        return np.array(predictions)

#========================================================================================================
#========================================================================================================
