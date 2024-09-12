                            # TITANIK SURVIVAL PREDICTION
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
titan_df = pd.read_csv("./titanic_train.csv")
titan_df.head()
# titan_df.columns
# titan_df.head()
titan_df.info()
#                          Explore Data Analysis
titan_df.describe()
sns.countplot(x='Sex', data = titan_df)
plt.show()
sns.countplot(x='Pclass', data = titan_df)
plt.show()
sns.countplot(x='Pclass', data = titan_df, hue='Sex')
plt.show()
# titan_df.head()
sns.displot(titan_df['Age'], kde=False, bins=30)
plt.show()
titan_df['Along'] = titan_df['SibSp'] + titan_df['Parch']
titan_df.tail()

titan_df['Along'].loc[titan_df['Along'] > 0 ] = 1
titan_df.head()
                        # Factor of Survive--------
sns.countplot(x='Pclass', data = titan_df, hue='Survived')
sns.countplot(x='Sex', data = titan_df, hue='Survived')
                     #  Data Preparation
titan_df.head()
titanic_df = titan_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

# titan_df.info()
titanic_df.info()
titanic_df['Sex'] = np.where(titanic_df['Sex'] == 'male', 1, 0)
titanic_df.head()
plt.figure(figsize=(15,10))
sns.heatmap(titanic_df.corr(),annot=True)
                    # Handling Missing Values
titanic_df.isnull().sum()
sns.boxplot(x='Pclass', y='Age', data = titanic_df)
plt.show()
print(titanic_df[titanic_df['Pclass']==1]['Age'].mean())
print(titanic_df[titanic_df['Pclass']==2]['Age'].mean())
print(titanic_df[titanic_df['Pclass']==3]['Age'].mean())
def fill_age(row):
    age = row[0]
    pclass = row[1]
    # print(age, pclass)
    # print()

    if pd.isnull(age):
        if pclass ==1:
            return 38.23
        elif pclass ==2:
            return 29.87
        else :    
            return 25.14
    else:
        return age
        
titanic_df['Age'] = titanic_df[['Age', 'Pclass']].apply(fill_age, axis=1)
titanic_df.isnull().sum()
                    # MODEL BUILDING-------------------
x = titanic_df.drop(columns=['Survived'])
y = titanic_df['Survived']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=.20, random_state=0)
x_train.shape,x_test.shape
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=3)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# y_pred,y_test
model.score(x_train,y_train)
model.score(x_test, y_test)
from sklearn.metrics import classification_report, f1_score
print(classification_report(y_test, y_pred))
f1_score(y_test,y_pred)
#                       Visualize the Tree
from sklearn import tree
features = x.columns
features
plt.figure(figsize=(20,20))
result = tree.plot_tree(model, feature_names=features, class_names=['Dead','Survived'])