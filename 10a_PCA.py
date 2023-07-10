import numpy as np
class PCA:
    def __init__(self,n_components):
        self.n_components=n_components
        self.components=None
        self.mean=None
    def fit(self,X):
        self.mean=np.mean(X,axis=0)
        X=X-self.mean
        cov=np.cov(X.T)
        eigenvectors,eigenvalues = np.linalg.eig(cov)
        eigenvectors=eigenvectors.T
        idxs=np.argsort(eigenvalues)[::-1]
        eigenvalues=eigenvalues[idxs]
        eigenvectors=eigenvectors[idxs]
        self.components = eigenvectors[:self.n_components]
    def transform(self,X):
        X=X-self.mean
        return np.dot(X,self.components.T)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    data=datasets.load_iris()
    X=data.data
    y=data.target
    pca=PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)
    print(X.shape)
    print(X_projected.shape)
    x1=X_projected[:,0]
    x2=X_projected[:,1]
    plt.scatter(x1,x2,c=y,edgecolor="None",alpha=0.8,cmap=plt.cm.get_cmap("viridis",3))
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.colorbar()
    plt.show()
