import numpy as np
class LDA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)
        A = np.linalg.inv(SW).dot(SB)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        # project data
        return np.dot(X, self.linear_discriminants.T)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    data=datasets.load_iris()
    X=data.data
    y=data.target
    lda=LDA(2)
    lda.fit(X,y)
    X_projected = lda.transform(X)
    print(X.shape)
    print(X_projected.shape)
    x1=X_projected[:,0]
    x2=X_projected[:,1]
    plt.scatter(x1,x2,c=y,edgecolor="None",alpha=0.8,cmap=plt.cm.get_cmap("viridis",3))
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.colorbar()
    plt.show()
