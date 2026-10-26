import os
import pickle
import numpy as np
import warnings
import matplotlib.pyplot as plt
from features import extract_mfcc, KMeans
from gmm import GMM
from hmm import DiscreteHMM
np.random.seed(42)
warnings.filterwarnings("ignore")

DATA_PATH="recordings"
CODEBOOK=48
HMM_STATES=7
GMM_K=12

_models={}
# Load data
def load_data():
    train={i:[] for i in range(10)}
    val  ={i:[] for i in range(10)}
    test ={i:[] for i in range(10)}

    for f in sorted(os.listdir(DATA_PATH)):
        if not f.endswith(".wav"):
            continue
        try:
            d=int(f[0])
            s=int(f.split("_")[1].replace("speaker", ""))
        except:
            continue
        path=os.path.join(DATA_PATH, f)
        (train if s<=4 else val if s==5 else test)[d].append(path)

    return train,val,test

# Normalization

def normalize(x,m,s):
    x=(x-m)/s
    return (x-x.mean(0))/(x.std(0)+1e-8)


# Confusion Matrix
def build_confusion_matrix(true_labels,pred_labels):
    cm=np.zeros((10, 10), dtype=int)
    for t,p in zip(true_labels,pred_labels):
        cm[t][p] += 1
    return cm

def print_confusion_matrix(cm,title):
    print(f"\n{title}")
    print("pred:", *range(10))
    for i in range(10):
        print(f"{i:2d}:", *cm[i])
    print("-" * 40)

def print_per_digit(cm,title):
    print(f"\n{title}")
    print("digit  correct  total  acc")

    for d in range(10):
        tot=cm[d].sum()
        cor=cm[d][d]
        acc=round(cor/tot if tot else 0, 4)
        print(f"{d:2d}     {cor:3d}     {tot:3d}    {acc}")
    print("-" * 40)


# Plot 
def plot_confusion_matrix(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Evaluation

def evaluate(data, mean, std, kmeans, gmm, hmm, name):
    true, gp, hp = [], [], []

    for d in range(10):
        for f in data[d]:
            x = normalize(extract_mfcc(f), mean, std)
            seq = kmeans.predict(x)

            gp.append(np.argmax([gmm[i].score(x) for i in range(10)]))
            hp.append(np.argmax([hmm[i].score(seq) for i in range(10)]))
            true.append(d)
    g_acc=sum(t==p for t, p in zip(true,gp))/len(true)
    h_acc=sum(t==p for t, p in zip(true,hp))/len(true)
    print(f"\n{name}")
    print(f"GMM: {round(g_acc,4)}   HMM: {round(h_acc,4)}")
    cm_g = build_confusion_matrix(true, gp)
    cm_h = build_confusion_matrix(true, hp)
    print_confusion_matrix(cm_g, "GMM Confusion Matrix")
    print_confusion_matrix(cm_h, "HMM Confusion Matrix")
    print_per_digit(cm_g,"GMM per digit")
    print_per_digit(cm_h,"HMM per digit")
    plot_confusion_matrix(cm_g, f"{name}-GMM")
    plot_confusion_matrix(cm_h, f"{name}-HMM")

    return g_acc,h_acc

# Main Pipeline
def main():
    global _models
    print("Loading data...")
    train,val,test=load_data()
    n_train=sum(len(v) for v in train.values())
    n_val  =sum(len(v) for v in val.values())
    n_test =sum(len(v) for v in test.values())
    print(f" Train (speakers 1–4) : {n_train} files")
    print(f" Val   (speaker 5)    : {n_val} files")
    print(f" Test  (speaker 6)    : {n_test} files")
    all_features=[]
    train_feats={i: [] for i in range(10)}

    for d in range(10):
        for file in train[d]:
            f = extract_mfcc(file)
            train_feats[d].append(f)
            all_features.append(f)
    all_features=np.vstack(all_features)

    mean = np.mean(all_features, axis=0)
    std  = np.std(all_features, axis=0) + 1e-8
    for d in range(10):
        train_feats[d] = [normalize(f, mean, std) for f in train_feats[d]]
    all_features = normalize(all_features, mean, std)

    kmeans = KMeans(K=CODEBOOK)
    kmeans.fit(all_features)

    # Train GMM & HMM

    print("\nTraining per digit GMM and HMM models")
    gmm_models={}
    hmm_models={}
    for d in range(10):
        print(f"  Digit- {d}")
        X = np.vstack(train_feats[d])

        # GMM
        gmm=GMM(K=GMM_K)
        idx=np.random.choice(len(kmeans.centroids), gmm.K, replace=False)
        gmm.means = kmeans.centroids[idx].copy()
        gmm.covs  = [np.eye(X.shape[1]) for _ in range(gmm.K)]
        gmm.pi    = np.ones(gmm.K) / gmm.K
        gmm.fit(X)
        gmm_models[d] = gmm

        # HMM
        sequences = [kmeans.predict(f) for f in train_feats[d]]
        hmm=DiscreteHMM(N=HMM_STATES, M=CODEBOOK)
        hmm.fit(sequences, n_iter=20)
        hmm_models[d] = hmm

    _models = {"mean": mean,"std": std,"kmeans": kmeans,"gmm_models": gmm_models,"hmm_models": hmm_models,}
    val_gmm_acc, val_hmm_acc = evaluate(val, mean, std, kmeans, gmm_models, hmm_models,"Validation speaker 5")

    tst_gmm_acc, tst_hmm_acc = evaluate(
        test, mean, std, kmeans, gmm_models, hmm_models,
        "Test for speaker 6")
    # Final Table
    print("\nFinal")
    print("-" * 55)
    print(f"{'Metric':<22}  {'GMM':>10}  {'HMM':>10}")
    print(" " + "─" * 46)
    print(f"{'Validation accuracy':<22}  {val_gmm_acc:>10.4f}  {val_hmm_acc:>10.4f}")
    print(f"{'Test accuracy':<22}        {tst_gmm_acc:>10.4f}  {tst_hmm_acc:>10.4f}")


if __name__ == "__main__":
    main()