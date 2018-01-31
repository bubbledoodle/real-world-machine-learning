from util import *

print("Create a new dataset, circles")
X, y = make_circles(n_samples=1000, random_state=2)
plot_learning_curve(LinearSVC(C=0.25), "LinearSVC(C=0.25)",
                    X, y, ylim=(0.5, 1.0),
                    train_sizes=np.linspace(.1, 1.0, 5))

print("X shape:" + str(X.shape))    # X is like 2-dim coordinate
print("y shape:" + str(y.shape))    # while y is like its marker to the inner circle or outer circle
df = DataFrame(np.hstack((X, y[:, None])), columns=list(range(2)) + ["class"])
_ = sns.pairplot(df, vars=[0, 1], hue="class", plot_kws={'alpha': 0.3})


X_extra = np.hstack((X, X[:, [0]]**2 + X[:, [1]]**2))   # way to access python matrix
plot_learning_curve(LinearSVC(C=0.25), "LinearSVC(C=0.25) + distance feature",
                    X_extra, y, ylim=(0.5, 1.0),
                    train_sizes=np.linspace(.1, 1.0, 5))
plt.show()
