# Be sure to pip install: pandas, NumPy, scikit-learn, Seaborn and Matplotlib.
# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

## =======================================================================================
# # print("***** Train_Set *****")
# # print(train.head())
# # print("\n")
# # print("***** Test_Set *****")
# # print(test.head())

# ## You can get some initial statistics of both the train and test DataFrames using pandas'
# ## describe()method.
# # print("***** Train_Set *****")
# # print(train.describe())

# # print("***** Values in train_set *****")
# # print(train.columns.values)

# ## What do we do when we have missing data (NaN)?
# ## For the train set
# train.isna().head()
# ## For the test set
# test.isna().head()
# ## Let's get the total number of missing values in both datasets.
# print("*****In the train set*****")
# print(train.isna().sum())
# print("\n")
# print("*****In the test set*****")
# print(test.isna().sum())

# ## There are a couple of ways to handle missing values:
# ## • Remove rows with missing values
# ## • Add value to missing values (best option)

# ## Now, there are several ways you can perform the imputation:
# ## • A constant value that has meaning within the domain, such as 0, distinct from all other values.
# ## • A value from another randomly selected record.
# ## • A mean, median or mode value for the column.
# ## • A value estimated by another machine learning model.


# ## Mean Imputation (adding missing values)
# ## Fill missing values with mean column values in the train set
# # train.fillna(train.mean(), inplace=True) # Fungerar inte för tillfället..?
# ## Fill missing values with mean column values in the test set
# # test.fillna(test.mean(), inplace=True)
# ## OBS! Values that are non-numeric wont get changed in the code above
# ""
# print("*****Sum is NaN train*****")
# print(train.isna().sum())
# print("*****Sum is NaN test*****")
# print(test.isna().sum())


# print("Survival count with respect to Pclass:")
# ## Survival count with respect to Pclass:
# print(
#     train[["Pclass", "Survived"]]
#     .groupby(["Pclass"], as_index=False)
#     .mean()
#     .sort_values(by="Survived", ascending=False)
# )

# print("Survival count with respect to Sex:")
# print(
#     train[["Sex", "Survived"]]
#     .groupby(["Sex"], as_index=False)
#     .mean()
#     .sort_values(by="Survived", ascending=False)
# )

# print("Survival count with respect to SibSps:")
# # Survival count with respect to SibSp:
# print(
#     train[["SibSp", "Survived"]]
#     .groupby(["SibSp"], as_index=False)
#     .mean()
#     .sort_values(by="Survived", ascending=False)
# )

# g = sns.FacetGrid(train, col="Survived")
# g.map(plt.hist, "Age", bins=20)
# plt.show()

## =======================================================================================
print("\n")
train.info()
print("\n")
print("*****In the train set, nr of NaN*****")
print(train.isna().sum())
print("\n")
print("*****In the test set, nr of NaN*****")
print(test.isna().sum())
print("\n")

train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)

print("\n")
print("*****New*****")
print("*****In the train set, nr of NaN*****")
print(train.isna().sum())
print("\n")
print("*****In the test set, nr of NaN*****")
print(test.isna().sum())
print("\n")

# Features Name, Ticket, Cabin and Embarked can be dropped and they will not have significant impact on the training 
# of the K-Means model (have no impact on the survival status of the passengers).

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

print("\n")
print("*****Engineering of train info*****")
train.info()

## Training of the K-Means model
## Droops the survival data
y = np.array(train['Survived'])
X = np.array(train.drop(['Survived'], axis=1).astype(float)) 

## Build the K-Means model
## Cluster the passenger records into 2: Survived or Not survived
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
KMeans(algorithm='lloyd', copy_x=True, init='k-means++', max_iter=300, n_clusters=2, n_init=10, random_state=None, tol=0.0001, verbose=0)


correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))