from util import *

X, y = make_classification(200000, n_features=200, n_informative=25, n_redundant=0, n_classes=10, class_sep=2, random_state=0)
est = SGDClassifier(penalty="l2", alpha=0.001, max_iter=1000)
progressive_validation_score = []
train_score = []

for datapoint in range(0, 199000, 1000):
    # create batch
    X_batch = X[datapoint:datapoint+1000]
    y_batch = y[datapoint:datapoint+1000]

    # first batch come does not have score
    if datapoint > 0:
        progressive_validation_score.append(est.score(X_batch, y_batch))
    est.partial_fit(X_batch, y_batch, classes=range(10))
    if datapoint > 0:
        train_score.append(est.score(X_batch, y_batch))

plt.plot(train_score, label="train score")
plt.plot(progressive_validation_score, label="progressive validation score")
plt.xlabel("Mini-batch")
plt.ylabel("Score")
plt.legend(loc='best')
plt.show()
