from util import *

X, y = make_classification(1000, n_features=20, n_informative=2,
                           n_redundant=2, n_classes=2, random_state=0)
df = DataFrame(np.hstack((X, y[:, None])), columns=list(range(20)) + ["class"])
print(df[:6])   # from 0 ~ 6
print("visualize it")
#_ = sns.pairplot(df[:50], vars=[8, 11, 12, 14, 19], hue="class", size=1.5, diag_kind="kde")

# plt.figure(figsize=(12, 10))
corr = df.corr()
_ = sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)

# 20% data, total 5 training score check points
plot_learning_curve(LinearSVC(C=10.0), "LinearSVC(C=10.0)", X, y, ylim=(0.8, 1.01),
                    train_sizes=np.linspace(.05, 0.2, 5))
# previously under-fitting, increase number of training data
plot_learning_curve(LinearSVC(C=10.0), "LinearSVC(C=10.0)", X, y, ylim=(0.8, 1.1),
                    train_sizes=np.linspace(.1, 1.0, 5))
# only choose X[:, [11, 14]] as input features
plot_learning_curve(LinearSVC(C=10.0), "LinearSVC(C=10.0) Features: 11&14",
                    X[:, [11, 14]], y, ylim=(0.8, 1.0),
                    train_sizes=np.linspace(.05, 0.2, 5))
plt.show()
