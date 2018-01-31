from util import *


def feature_eng(train, test):
    print(train.shape)
    train.info()

    print('-' * 40 + '\nI. dropping id & eliminate outliers')
    print('At this section, droped id and some outliers')
    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
    print("The test data size after dropping Id feature is : {} ".format(test.shape))

    fig, ax = plt.subplots()
    ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel('GrLivArea')
    plt.show()
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    print('-' * 40 + '\n II. statics')
    print('At this section, draw two pics. one is target distribution, the other is feature correlation')
    sns.distplot(train['SalePrice'], fit=norm)
    (mu, sigma) = norm.fit(train['SalePrice'])
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    plt.show()

    corrmat = train.corr()
    plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()

    print('-' * 40 + '\nIII. imputing missing data')
    ntrain = train.shape[0]
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    null_percentage(all_data)
    all_data = impute(all_data)
    print('~' * 40)
    print('After imputing:(should show nothing)')
    null_percentage(all_data)

    print('-' * 40 + '\n More feature engineering')
    print('1. label encoding')
    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
    all_data['OverallCond'] = all_data['OverallCond'].astype(str)
    all_data['YrSold'] = all_data['YrSold'].astype(str)
    all_data['MoSold'] = all_data['MoSold'].astype(str)
    all_data = label_encoding(all_data)

    print('2. change some feature')
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    print('3. skewed features')
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
    all_data = trans_skew(all_data, skewness)
    all_data = pd.get_dummies(all_data)
    train = all_data[:ntrain]
    test = all_data[ntrain:]
    return train, test



