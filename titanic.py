# read train.csv

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from missingvalues import missing_ratio

# use csv package to read trian.csv
train_ori = csv.reader(open("train.csv","r"))

# this will only print the object of the file
# print(train_ori)

# this statements can help us print all contents:
# this is not what we want, we want each value can be associated with its feature' name
# thus, we use pandas
# for i in train_ori:
#     print(i)

train_ori = pd.read_csv('train.csv')
# print(train_ori)
#  the output is the same as R's
# print(train_ori.Parch)

# then, we can choose features we want to use

# y-Survived
# Pclass -numeric
# Sex - categorical
# Age - numeric
# SibSp -  numeric
# Fare - numberic
# Parch - numeric
# Embarked - categorical

# deal with missing values
# col 2 and 6 have missing values, age and embarked
train = train_ori.loc[:,["Pclass","Sex","Age","SibSp","Fare","Parch","Embarked","Survived"]]
missing_ratio(train)

# plt.hist(train.Age, normed = True)
# cannot have missing values
# plt.show()

# deal with categorical data Sex, Embarked
# we set female 1 and male 0
train.Sex = train.Sex.replace({"female":1, "male":0})

# set C = Cherbourg as 0, Q = Queenstown as 1, S = Southampton as 2

train.Embarked = train.Embarked.replace({"C":0, "Q": 1,"S":2})

# use mean to replace missing age
train = train.fillna(train.mean())
missing_ratio(train)

# we separate the train_ori to train_new(80%) and train_cv (20%) randomly
train = train.sample(frac = 1)
train_new = train.head(round(train.shape[0] * 0.8))
train_cv = train.head(round(train.shape[0] * 0.2))

# Empty DataFrame
# Columns: [PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked]
# Index: []

# similar to R's summary()
#print(train_ori.describe())

