#%% Library
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() #for plot styling
import numpy as np
#%% Data
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples = 300, centers=4, cluster_std=0.60, random_state=0)
X=X[:,::-1] # flips axes for better plotting
#%% Plot data with Kmeans labels
from sklearn.cluster import KMeans
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:,0],X[:,1], c=labels, s=40, cmap='viridis');
plt.show()
#%% GMM
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:,0],X[:,1], c=labels, s=40,cmap='viridis');
plt.show()
#%% Probabilistic Cluster Assignments
probs = gmm.predict_proba(X)
print(probs[:5].round(3))
size = 50*probs.max(1)**2 # square emphasizes differences
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=size);
plt.show()
#%% GM ellipses programs
from matplotlib.patches import Ellipse

def drawEllipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principle axis
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2*np.sqrt(s)
    else:
        angle = 0
        width, height = 2*np.sqrt(covariance)

    #Draw an ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig*height, angle, **kwargs))

def plotGMM(gmm, X, label = True, ax = None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        drawEllipse(pos, covar, alpha=w*w_factor)
#%% Plotting the ellipse
gmm = GaussianMixture(n_components=4, random_state=42)
plotGMM(gmm, X)
plt.show()

#%% Moons
from sklearn.datasets import make_moons
Xmoon, Ymoon = make_moons(200, noise=0.05, random_state=0)
plt.scatter(Xmoon[:, 0],Xmoon[:, 1]);
plt.show()

#%% fitting a two component GMM
gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
plotGMM(gmm2, Xmoon)
plt.savefig('moon1.png')
plt.show()

#%% improving fit
gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
plotGMM(gmm16, Xmoon, label=False)
plt.savefig('moon2.png')
plt.show()

#%% AIC and BIC measurements
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Xmoon)
          for n in n_components]

plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.ylabel('measurement');
plt.savefig('aicbic.png')
plt.show()
