from know_data import *

setting()
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
train_ID = train['Id']
test_ID = test['Id']

y_train = train.SalePrice.values
train, test = feature_eng(train, test)
print('-'*40 + '\nModelling')

n_folds = 5




