from util import *
from sklearn import manifold, decomposition, random_projection
from sklearn.datasets import load_digits
from matplotlib import offsetbox
import time

digits = load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
print ("Dataset consist of %d samples with %d features each" % (n_samples, n_features))


n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        # img[xid, yid]: can be reached as 8x8 block directly
        # where, X[bala] will be the other program random idx to pick
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
_ = plt.title('A selection from the 8*8=64-dimensional digits dataset')


print("I. Random choose two dim as feature space")
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)


def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


start_time = time.time()
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits (time: %.3fs)" % (time.time() - start_time))

print("II. PCA dim reduction")
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
start_time = time.time()
print(X_pca.shape)
plot_embedding(X_pca, "Principal Components projection of the digits (time: %.3fs)" % (time.time() - start_time))

print("III. t-SNE non-linear transformation dim reduction")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
start_time = time.time()
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)
plot_embedding(X_tsne, "t-SNE embedding of the digits (time: %.3fs)" % (time.time() - start_time))

plt.show()
