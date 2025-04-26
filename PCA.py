## PCA
import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	
    ## standardise data 
    x_mean = np.mean(data, axis = 0)
    x_std = np.std(data, axis = 0)
    data = (data - x_mean) / x_std

    ## calculate covariance matrix
    data_cov = np.cov(data, rowvar = False)

    ## find eigenvectors and eigenvalues
    e1, e2 = np.linalg.eig(data_cov)

    ## sort
    idx = np.argsort(e1)[::-1]
    e2 = e2[:, idx]
    principal_components = e2[:, :k]
	
    return np.round(principal_components, 4)

# np.argsort(e1)返回e1从小到大排序对应的下标，而PCA的成分要求从大到小排序
# 因此需要反向切片
