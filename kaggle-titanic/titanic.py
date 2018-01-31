import pandas as pd
import numpy as np
import random as rnd

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
combine = [train_df, test_df]
print("I. Get data info")
train_df.info()
print("-"*40)
test_df.info()


print("-"*40)
print("II. Drop Features:")
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


print("-"*40)
print("III. Create and clear title feature...")
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
print("Finish create")
print('-'*40)


print("IV. Convert title feature into ordinal feature...")
title_mapping = {"Mr": 1,
                 "Miss": 2,
                 "Mrs": 3,
                 "Master": 4,
                 "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print("Finish convert")
print('-'*40)


print("V. Dropping name and passengerID")
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df_Id = test_df.PassengerId
test_df = test_df.drop(['Name', 'PassengerId'], axis=1)
combine =[train_df, test_df]
print("After droping shape:")
print(train_df.shape, test_df.shape)

print('-'*40)
print("VI. Convert sex feature into catagorical feature...")
sex_map = {'male': 0,
           'female': 1}
for dataset in combine:
    dataset.Sex = dataset.Sex.map(sex_map).astype(int)
print("Finish converting")

print('-'*40)
print("VII. Complete age numerical continuous feature")
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]
    dataset['Age'] = dataset['Age'].astype(int)

print("Create age band")
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age']

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print("Finish filling ages")

print('-'*40)
print("VIII. Create new feature combining existing features")
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('-'*40)
print("Create IsAlone feature")
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
print('-'*40)
print("Check current features")
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

print('-'*40)
print("Fill embarked")
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False))
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

print('-'*40)
print("Fill fare & change to fare band feature")
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print()
print('-'*40)
print("Final check")
print(train_df.head())
print('-'*40)
print(test_df.head())

print()
print('~'*40)
print("IX. Model, predict and solve")
print("Divide features and result...")
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.copy()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)


print("i. Logistic Regression")
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
# Y_pred to be submitted
# Y_pred = logreg.predict(X_test)
scores = cross_val_score(logreg, X_train, Y_train, cv=5)
print("Cross-validation scores:" + str(scores))
print("Done")
print('-'*40)
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
# because of dataframe, pd.Series will automatically match the column
print(coeff_df.sort_values(by='Correlation', ascending=False))

print("XI. Feature ensemble")
clf = LogisticRegression()
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0,
                               bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X_train, Y_train)

Y_pred = bagging_clf.predict(X_test)
print("X. Create csv")
result = pd.DataFrame({'PassengerId': test_df_Id.as_matrix(), 'Survived': Y_pred.astype(np.int32)})
result.to_csv("./titanic_linear_regression_result.csv", index=False)


