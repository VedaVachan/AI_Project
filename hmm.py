import numpy as np


class DiscreteHMM:
    def __init__(self,N=7,M=48):
        self.N=N
        self.M=M
        self.init_parameters()

    def init_parameters(self):
        N,M=self.N,self.M   
        self.A=np.zeros((N,N))
        for i in range(N):
            if i<N-1:
                self.A[i,i]=0.6
                self.A[i,i+1]=0.4
            else:
                self.A[i,i]=1.0

        self.pi=np.zeros(N)
        self.pi[0]=1.0
        self.B=np.ones((N,M))/M

    def logsumexp(self,a,axis=None):
        m=np.max(a,axis=axis,keepdims=True)
        res=np.log(np.sum(np.exp(a-m),axis=axis)+1e-300)
        if axis is not None:
            m=m.squeeze(axis=axis)
        else:
            m=m.squeeze()
        return m+res

    def log_emission(self,O):
        return np.log(self.B[:,O]+1e-8).T

    def forward(self,log_emit):
        T,N=log_emit.shape
        logA=np.log(self.A+1e-8)

        alpha=np.full((T,N),-np.inf)
        alpha[0]=np.log(self.pi+1e-8)+log_emit[0]

        for t in range(1,T):
            temp=alpha[t-1,:,None]+logA
            alpha[t]=log_emit[t]+self.logsumexp(temp,axis=0)

        return alpha

    def backward(self,log_emit):
        T,N=log_emit.shape
        logA=np.log(self.A+1e-8)

        beta=np.zeros((T,N))

        for t in range(T-2,-1,-1):
            temp=logA+log_emit[t+1]+beta[t+1]
            beta[t]=self.logsumexp(temp,axis=1)

        return beta

    def compute_gamma(self,alpha,beta,loglik):
        return np.exp(alpha+beta-loglik)

    def compute_xi(self,alpha,beta,log_emit,loglik):
        logA=np.log(self.A+1e-8)
        xi=(alpha[:-1,:,None]+logA[None,:,:]+log_emit[1:,None,:] +beta[1:,None,:] -loglik)
        return np.exp(xi)

    def fit(self,sequences,n_iter=20):
        mask=np.zeros((self.N,self.N),dtype=bool)
        for i in range(self.N):
            mask[i,i]=True
            if i<self.N-1:
                mask[i,i+1]=True

        for _ in range(n_iter):
            A_num=np.zeros_like(self.A)
            A_den=np.zeros(self.N)
            B_num=np.zeros_like(self.B)
            B_den=np.zeros(self.N)

            for O in sequences:
                log_emit=self.log_emission(O)
                alpha=self.forward(log_emit)
                beta=self.backward(log_emit)
                loglik=self.logsumexp(alpha[-1])

                gamma=self.compute_gamma(alpha,beta,loglik)
                xi=self.compute_xi(alpha,beta,log_emit,loglik)

                A_num +=xi.sum(axis=0)
                A_den +=gamma[:-1].sum(axis=0)

                one_hot=np.eye(self.M)[O]
                B_num+=gamma.T @ one_hot
                B_den+=gamma.sum(axis=0)

            A=A_num/(A_den[:,None]+1e-8)
            A[~mask]=0.0
            A /=A.sum(axis=1, keepdims=True)+1e-8
            self.A=A
            self.B=(B_num+1e-2)/(B_den[:,None]+1e-2*self.M)

    def score(self,O):
        log_emit=self.log_emission(O)
        alpha=self.forward(log_emit)
        return self.logsumexp(alpha[-1])