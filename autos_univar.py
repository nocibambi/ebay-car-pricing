import pandas as pd
from numpy import NaN, random

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as auc_score

# Loading data
train = pd.read_csv("autos_training_final.csv", sep="|")
test = pd.read_csv("autos_testing_final.csv", sep="|")
sample_submission = pd.read_csv("autos_submission.csv", sep=",")

# Defining column groups
numcols = train.dtypes.index[train.dtypes.isin(['float64', 'int64']) == True]
special_columns = ['dateCrawled', 'id', 'label']

# 1. Using only numeric dtypes
train_cl = train.loc[:,['yearOfRegistration', 'powerPS', 'kilometer',
                        'monthOfRegistration', 'nrOfPictures', 'postalCode',
                        'label']].dropna().copy()

# Running the model
model = LogisticRegression()
model.fit(train_cl.nrOfPictures.to_frame(), train_cl.label)

#test_cl = test.loc[:, ['yearOfRegistration',
#                       'powerPS',
#                       'kilometer',
#                       'monthOfRegistration',
#                       'nrOfPictures',
#                       'postalCode']].copy()

model.predict(test.nrOfPictures.to_frame())
confs = model.predict_proba(test.nrOfPictures.to_frame())[:,1]
univar_confs = pd.DataFrame([test.id, confs.reshape(-1,)]).transpose()
univar_confs.rename(columns={"Unnamed 0": 'label'}, inplace=True)

# Writing the result into csv
univar_confs.to_csv("univar_confs.csv", index=False)

# Something like this
#pd.DataFrame([test.id, confs])

# Handling datetime values
# train.yearOfRegistration.replace(to_replace=1000, value=NaN, inplace=True)
# train.monthOfRegistration.replace(to_replace=0, value=random.randint(1, 13), inplace=True)
# train.loc[:,['yearOfRegistration', 'monthOfRegistration']].apply(lambda x: pd.datetime(int(x[0]), int(x[1]), 15))
