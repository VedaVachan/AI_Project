import numpy as np


class GMM:
    def __init__(self,K=8):
        self.K=K
        self.means=None
        self.covs=None
        self.pi=None

    def init_params(self,X):
        N,D =X.shape
        idx =np.random.choice(N,self.K,replace=False)
        self.means=X[idx]
        self.covs=[np.eye(D) for _ in range(self.K)]
        self.pi =np.ones(self.K)/self.K

    def gaussian_log(self,X,k):
        if X.ndim==1:
            X=X[None,:]

        D=X.shape[1]
        var=np.diag(self.covs[k])
        diff= X-self.means[k]
        log_det =np.sum(np.log(var +1e-8))
        maha =np.sum((diff**2)/var,axis=1)

        return -0.5*(D*np.log(2*np.pi)+log_det+maha)

    def estep(self,X):
        N=X.shape[0]
        gamma=np.zeros((N,self.K))
        for k in range(self.K):
            gamma[:, k]=np.log(self.pi[k]+1e-8)+self.gaussian_log(X,k)

        m=np.max(gamma,axis=1,keepdims=True)
        gamma =np.exp(gamma-m)
        gamma /=gamma.sum(axis=1,keepdims=True)
        return gamma

    def mstep(self,X,gamma):
        N,D =X.shape
        Nk =gamma.sum(axis=0)

        for k in range(self.K):
            if Nk[k]<1e-3:
                self.means[k]=X[np.random.randint(N)]
                self.covs[k]=np.eye(D)
                self.pi[k] =1.0/self.K
                continue


            w = gamma[:,k]
            self.means[k]=(w @ X)/Nk[k]
            diff =X-self.means[k]
            var =(w @ (diff**2))/Nk[k]
            self.covs[k]=np.diag(np.maximum(var,1e-2))
        self.pi=Nk/N

    def log_likelihood(self,X):
        N = X.shape[0]
        log_px=np.zeros((N,self.K))
        for k in range(self.K):
            log_px[:,k]=np.log(self.pi[k]+1e-8)+self.gaussian_log(X,k)

        m=np.max(log_px,axis=1,keepdims=True)
        return np.sum(m.squeeze()+np.log(np.sum(np.exp(log_px-m),axis=1)))



    def fit(self,X,max_iter=40):
        if self.means is None:
            self.init_params(X)

        prev=-np.inf
        for _ in range(max_iter):
            gamma=self.estep(X)
            self.mstep(X,gamma)
            ll=self.log_likelihood(X)
            if abs(ll-prev)<1e-4:
                break
            prev=ll

    def score(self,X):
        if X.ndim ==1:
            X =X[None,:]
        return self.log_likelihood(X)