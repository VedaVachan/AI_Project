import librosa
import numpy as np


#librosa load audio
def load_audio(file):
    try:
        y,sr=librosa.load(file, sr=None)
        if len(y)>0:
            return y,sr
    except:
        pass
    
    try:
        y,sr=librosa.load(file, sr=16000)
        if len(y)>0:
            return y,sr
    except:
        pass
#this is soundfile
    try:
        import soundfile as sf
        y,sr=sf.read(file)
        y=np.array(y, dtype=np.float32)
        if y.ndim>1:
            y=y.mean(axis=1)
        return y,int(sr)
    except:
        pass

    try:  
        y = np.array(y, dtype=np.float32)
        if y.ndim>1:
            y=y.mean(axis=1)
        if np.max(np.abs(y))>0:
            y=y/np.max(np.abs(y))
        return y, int(sr)
    except:
        pass

    return np.zeros(8000),16000

#now extract mfcc features
def extract_mfcc(file):
    y,sr = load_audio(file)

    mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=20,n_fft=512,hop_length=160)
    w =mfcc.shape[1]
    width=min(9, w if w%2 else w-1)
    width=max(width,3)
    d1=librosa.feature.delta(mfcc,order=1,width=width)
    d2=librosa.feature.delta(mfcc,order=2,width=width)
    return np.concatenate([mfcc,d1,d2],axis=0).T


class KMeans:
    def __init__(self,K=48,max_iter=100):
        self.K=K
        self.max_iter=max_iter
    def init_centroids(self,X):
        N=len(X)
        centroids =[X[np.random.randint(N)]]


        for _ in range(self.K-1):
            dists =np.min(
                [np.sum((X-c)**2,axis=1) for c in centroids],
                axis=0
            )
            probs=dists/(dists.sum()+1e-10)
            centroids.append(X[np.random.choice(N,p=probs)])

        return np.array(centroids)

    def fit(self,X):
        self.centroids=self.init_centroids(X)

        for _ in range(self.max_iter):
            dist =np.linalg.norm(X[:, None]-self.centroids,axis=2)
            labels=np.argmin(dist,axis=1)

            new_centroids =[]
            for k in range(self.K):
                if np.any(labels==k):
                    new_centroids.append(X[labels==k].mean(axis=0))
                else:
                    new_centroids.append(self.centroids[k])

            new_centroids=np.array(new_centroids)

            if np.allclose(self.centroids,new_centroids):
                break
            
            self.centroids=new_centroids

    def predict(self, X):
        dist = np.linalg.norm(X[:, None]-self.centroids,axis=2)
        return np.argmin(dist,axis=1)