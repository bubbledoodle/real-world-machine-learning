import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("./train.csv")
data_train.info()
print(data_train.describe())


##########################
#    Basic Statistics    #
##########################
fig = plt.figure(1)
fig.set(alpha=0.2)

# overall survive count
plt.subplot(231)
#plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title("survived count")
plt.ylabel("number of people")

# overall Pclass distribution
plt.subplot(232)
#plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("number of people")
plt.title("class distribution")

# survived age distribution
plt.subplot(233)
#plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("age")
plt.grid(b=True, which='major', axis='y')
plt.title("survive based on age")

# age distribution among Pclass
plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("age")
plt.ylabel("density")
plt.title("age distribution among class")
plt.legend(("first", "second", "third"),loc='best')

# embark boarding statistic
plt.subplot(236)
#plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("embark")
plt.ylabel("number of people")
plt.show()

##########################
#    Find Correlation    #
##########################
fig, axes = plt.subplots(nrows=2, ncols=1)
fig.set(alpha=0.2)

# Pclass vs survive result
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({
    "survived": Survived_1,
    "not_survived": Survived_0
})

df.plot(kind="bar", stacked=True, ax=axes[0])
plt.title("result based on class")
plt.xlabel("class")
plt.ylabel("number of people")

# Sex vs survive result
Survived_m = data_train.Survived[data_train.Sex == "male"].value_counts()
Survived_f = data_train.Survived[data_train.Sex == "female"].value_counts()
df = pd.DataFrame({
    "male": Survived_m,
    "female": Survived_f
})
df.plot(kind="bar", stacked=True, ax=axes[1])
plt.title("result based on Sex")
plt.xlabel("survive")
plt.ylabel("number of people")
plt.show()


g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

g = data_train.groupby(['SibSp','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

# trails to show relation
print(data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*20)
print(data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*20)
print(data_train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_'*20)
print(data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))