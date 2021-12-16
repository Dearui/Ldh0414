import re
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
train_data = pd.read_csv("C:/Users/sb/Desktop/机器学习/泰坦尼克号/train.csv")
test_data = pd.read_csv("C:/Users/sb/Desktop/机器学习/泰坦尼克号/test.csv")
#train_data.info()
#train_data.corr()
"""
#维度相关性热力图
plt.figure(figsize=(20,20))
sns.heatmap(train_data.corr(),annot=True)
plt.show()
"""
"""
年龄与乘客仓等级与Survived的关系
fig,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot('Pclass','Age',hue='Survived',data=train_data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
sns.violinplot('Sex','Age',hue='Survived',data=train_data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
"""
test_data0 = test_data
train_data=train_data.fillna({"Embarked":1})
train_data=train_data.fillna({"Fare":8.05})
train_data=train_data.fillna({"A":5})
test_data=test_data.fillna({"A":5})
train_data=train_data.fillna({"Cabin":0})
test_data=test_data.fillna({"Cabin":0})
train_data.loc[train_data["Cabin"] != 0,"Cabin"] = 1
test_data.loc[test_data["Cabin"] != 0,"Cabin"] = 1
train_data.loc[train_data["Age"] <= 16,"Age"] = 0
train_data.loc[(train_data["Age"] > 16) & (train_data["Age"] <= 32), "Age"] = 1
train_data.loc[(train_data["Age"] > 32) & (train_data["Age"] <= 48), "Age"] = 2
train_data.loc[(train_data["Age"] > 48) & (train_data["Age"] <= 64), "Age"] = 3
train_data.loc[train_data["Age"] > 64, "Age"] = 4
test_data.loc[test_data["Age"] <= 16,"Age"] = 0
test_data.loc[(test_data["Age"] > 16) & (test_data["Age"] <= 32), "Age"] = 1
test_data.loc[(test_data["Age"] > 32) & (test_data["Age"] <= 48), "Age"] = 2
test_data.loc[(test_data["Age"] > 48) & (test_data["Age"] <= 64), "Age"] = 3
test_data.loc[test_data["Age"] > 64, "Age"] = 4


train_data=train_data.fillna(method='ffill')
test_data=test_data.fillna(method='ffill')
"""
train_data.loc[(train_data["Fare"] <= 16.136), "Fare"] = 0
train_data.loc[(train_data["Fare"] >16.136) & (train_data["Fare"] <= 32.102), "Fare"] = 1
train_data.loc[(train_data["Fare"] >32.102)& (train_data["Fare"] <= 48.068) , "Fare"] = 2
train_data.loc[(train_data["Fare"] >48.068)& (train_data["Fare"] <= 64.034) , "Fare"] = 3
train_data.loc[(train_data["Fare"] > 64.034), "Fare"] = 4
test_data.loc[(test_data["Fare"] <= 16.136), "Fare"] = 0
test_data.loc[(test_data["Fare"] >16.136) & (test_data["Fare"] <= 32.102), "Fare"] = 1
test_data.loc[(test_data["Fare"] >32.102)& (test_data["Fare"] <= 48.068) , "Fare"] = 2
test_data.loc[(test_data["Fare"] >48.068)& (test_data["Fare"] <= 64.034) , "Fare"] = 3
test_data.loc[(test_data["Fare"] > 64.034), "Fare"] = 4
"""

def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

titleDf = pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = train_data['Name'].map(getTitle)
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
#map函数：对Series每个数据应用自定义的函数计算

titleDf['Title'] = titleDf['Title'].map(title_mapDict)
df1=pd.DataFrame({'col1':titleDf['Title']})
train_data['N']=df1
train_data.loc[(train_data["N"] == 'Miss') , "N"] = 1
train_data.loc[(train_data["N"] == 'Royalty') , "N"] = 1
train_data.loc[(train_data["N"] == 'Officer') , "N"] = 2
train_data.loc[(train_data["N"] == 'Mrs') , "N"] = 1
train_data.loc[(train_data["N"] == 'Master') , "N"] = 1
train_data.loc[(train_data["N"] == 'Mr') , "N"] = 0

def getTitle2(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

titleDf = pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title1'] = test_data['Name'].map(getTitle2)
title_mapDict2 = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title1'] = titleDf['Title1'].map(title_mapDict2)
df1=pd.DataFrame({'col2':titleDf['Title1']})
test_data['N']=df1

test_data.loc[(test_data["N"] == 'Miss') , "N"] = 1
test_data.loc[(test_data["N"] == 'Royalty') , "N"] = 1
test_data.loc[(test_data["N"] == 'Officer') , "N"] = 2
test_data.loc[(test_data["N"] == 'Mrs') , "N"] = 1
test_data.loc[(test_data["N"] == 'Master') , "N"] = 1
test_data.loc[(test_data["N"] == 'Mr') , "N"] = 0

"""
train_data.loc[train_data["Parch"] ==0,"Parch"]=0
train_data.loc[train_data["Parch"] ==1,"Parch"]=1
train_data.loc[train_data["Parch"] ==2,"Parch"]=1
train_data.loc[train_data["Parch"] ==3,"Parch"]=1
train_data.loc[train_data["Parch"] ==4,"Parch"]=2
train_data.loc[train_data["Parch"] ==5,"Parch"]=0
train_data.loc[train_data["Parch"] ==6,"Parch"]=2

test_data.loc[test_data["Parch"] ==1,"Parch"]=0
test_data.loc[test_data["Parch"] ==1,"Parch"]=1
test_data.loc[test_data["Parch"] ==2,"Parch"]=1
test_data.loc[test_data["Parch"] ==3,"Parch"]=1
test_data.loc[test_data["Parch"] ==4,"Parch"]=2
test_data.loc[test_data["Parch"] ==5,"Parch"]=0
test_data.loc[test_data["Parch"] ==6,"Parch"]=2

g=sns.FacetGrid(train_data,col='Survived')
g.map(plt.hist,'Parch',bins=100)
plt.show()
"""

"""
train_data.loc[train_data["SibSp"] ==0,"SibSp"]=3
train_data.loc[train_data["SibSp"] ==1,"SibSp"]=5
train_data.loc[train_data["SibSp"] ==2,"SibSp"]=4
train_data.loc[train_data["SibSp"] ==3,"SibSp"]=2
train_data.loc[train_data["SibSp"] ==4,"SibSp"]=1
train_data.loc[train_data["SibSp"] ==5,"SibSp"]=0
train_data.loc[train_data["SibSp"] ==8,"SibSp"]=0

test_data.loc[test_data["SibSp"] ==0,"SibSp"]=3
test_data.loc[test_data["SibSp"] ==1,"SibSp"]=5
test_data.loc[test_data["SibSp"] ==2,"SibSp"]=4
test_data.loc[test_data["SibSp"] ==3,"SibSp"]=2
test_data.loc[test_data["SibSp"] ==4,"SibSp"]=1
test_data.loc[test_data["SibSp"] ==5,"SibSp"]=0
test_data.loc[test_data["SibSp"] ==8,"SibSp"]=0
train_data['SP']=train_data['SibSp']+train_data['Parch']+1
test_data['SP']=test_data['SibSp']+test_data['Parch']+1
train_data.loc[train_data["SP"] ==1,"SP"]=0
train_data.loc[train_data["SP"] ==2,"SP"]=1
train_data.loc[train_data["SP"] ==3,"SP"]=1
train_data.loc[train_data["SP"] ==4,"SP"]=1
train_data.loc[train_data["SP"] ==5,"SP"]=1
train_data.loc[train_data["SP"] ==6,"SP"]=1
train_data.loc[train_data["SP"] ==7,"SP"]=1
train_data.loc[train_data["SP"] ==8,"SP"]=1
train_data.loc[train_data["SP"] ==11,"SP"]=1
test_data.loc[test_data["SP"] ==1,"SP"]=0
test_data.loc[test_data["SP"] ==2,"SP"]=1
test_data.loc[test_data["SP"] ==3,"SP"]=1
test_data.loc[test_data["SP"] ==4,"SP"]=1
test_data.loc[test_data["SP"] ==5,"SP"]=1
test_data.loc[test_data["SP"] ==6,"SP"]=1
test_data.loc[test_data["SP"] ==7,"SP"]=1
test_data.loc[test_data["SP"] ==8,"SP"]=1
test_data.loc[test_data["SP"] ==11,"SP"]=1
"""
"""
pd.crosstab(train_data.Sex,train_data.Survived).plot.bar(stacked=True,color=['#AAAA','0000'])
plt.xticks(rotation=0,size='large')
plt.legend(bbox_to_anchor=(0.55,0.9))
plt.show(rotation=0,size='large')
"""
"""
train_survived=train_data[train_data['Embarked'].notnull()]
train_survived['Embarked'].value_counts().plot.pie(autopct='%1.2f%%')

train_survived=train_data[train_data['Fare'].notnull()]
train_survived['Fare'].value_counts().plot.pie(autopct='%1.2f%%')
"""


#性别之间的存活概率关系的条形图
#train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(figsize=(8,6))

for i in range(891):
    train_data['nd']=train_data["Name"].map(lambda i :len(re.split(" ",i)))
for i in range(891):
    test_data['nd']=test_data["Name"].map(lambda i :len(re.split(" ",i)))
print(train_data)
#train_data.info()
#test_data.info()
"""
#数据处理后维度相关性热力图
plt.figure(figsize=(20,20))
sns.heatmap(train_data.corr(),annot=True)
"""
"""
g=sns.FacetGrid(train_data,col='Survived')
g.map(plt.hist,'Fare',bins=100)
"""
"""
fig,ax=plt.subplots(1,2,figsize=(15,4))
train_data['Fare'].hist(bins=70,ax=ax[0])
train_data.boxplot(column='Fare',by='Survived',showfliers=False,ax=ax[1])
"""

plt.show()

tree_train = train_data[['Pclass', 'Sex','Age', 'Fare', 'SibSp','Embarked','N','Cabin']]
x_train = train_data[['Pclass', 'Sex', 'Age','Fare', 'SibSp','Embarked','N','Cabin']]
y_train = train_data['Survived']
#x_train.info()
y_test = pd.read_csv("C:/Users/sb/Desktop/机器学习/泰坦尼克号/gender_submission.csv")
test_data['Survived']=y_test['Survived']
test_data = test_data[['Survived', 'Pclass', 'Sex','Age', 'Fare', 'SibSp','Embarked','N','Cabin']]
x_test = test_data[['Pclass', 'Sex','Age', 'Fare', 'SibSp','Embarked','N','Cabin']]
y_test = test_data['Survived']
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# 使用随机森林模型训练以及预测分析。
rfc = RandomForestClassifier(n_estimators=280, max_depth=5, random_state=1)
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)
# 输出随机森林树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('随机森林树预测：', rfc.score(x_test, y_test))
print(classification_report(rfc_y_pred, y_test))

# 逻辑回归
rfc1 = LogisticRegression()
rfc1.fit(x_train, y_train)
lg_pred = rfc1.predict(x_test)
# 输出支持向量机分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('逻辑回归预测：', rfc1.score(x_test, y_test))
print(classification_report(lg_pred, y_test))

# 使用决策树模型训练以及预测分析。
rfc2 = DecisionTreeClassifier(max_depth=5)
rfc2.fit(x_train, y_train)
rfc_y_pred2 = rfc2.predict(x_test)
# 输出决策树在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('决策树预测：', rfc2.score(x_test, y_test))
print(classification_report(rfc_y_pred2, y_test))
np.random.seed(15)
rfc3 = LinearSVC()
rfc3.fit(x_train, y_train)
svm_pred3 = rfc3.predict(x_test)
# 输出支持向量机分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('支持向量机分类器预测：', rfc3.score(x_test, y_test))
print(classification_report(svm_pred3, y_test))

final = pd.DataFrame({"PassengerId":test_data0["PassengerId"],
                    "Survived":rfc_y_pred})
final.to_csv("C:/Users/sb/Desktop/机器学习/泰坦尼克号/Sample31.csv",index=False)
