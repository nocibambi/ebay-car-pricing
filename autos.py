# Data handling
import pandas as pd
import numpy as np
import time
from datetime import date

# Preprocessing
from sklearn.preprocessing import Normalizer, MinMaxScaler, OneHotEncoder, \
    StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.pipeline import Pipeline as imbPipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    IsolationForest, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, \
                                          LinearDiscriminantAnalysis

# Cross validation
from sklearn.model_selection import train_test_split, KFold, cross_val_score, \
    cross_validate, cross_val_predict, GridSearchCV, ParameterGrid, RepeatedStratifiedKFold

# Feature selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE

# Performance measures
from sklearn.metrics import accuracy_score, auc, roc_curve, \
    roc_auc_score as auc_score, confusion_matrix, classification_report

# Loading data
train = pd.read_csv("autos_training_final.csv", sep="|")
#train = train.sample(frac=0.1)

test = pd.read_csv("autos_testing_final.csv", sep="|")
sample_submission = pd.read_csv("autos_submission.csv", sep=",")

# Defining column groups
spec_cols = ['id', 'label', 'dateCrawled']
redundant_cols = ['seller', 'offerType', 'nrOfPictures']
date_cols = ['dateCreated', 'lastSeen']

num_cols = train.loc[:,~train.columns.isin(spec_cols \
                                          + redundant_cols \
                                          + date_cols)] \
                .select_dtypes(include=[int, float]) \
                .columns

cat_cols = train.select_dtypes(include=object).columns
                #.loc[:,train.columns != 'name'] \

# Attribute specific missing value codes
col_nanval = [['yearOfRegistration', 1000],
              ['powerPS', 0],
              ['monthOfRegistration', 0]]

# Data cleaning functions
# Handling native missing data codes
def rep_nanvals(data, collist, inplace=False):
    for col_nan in collist:
        col = data.loc[:, col_nan[0]]
        nanval = col_nan[1]
        col.replace(to_replace=nanval,
                    value=np.random.choice(col[col != nanval]),
                    inplace=inplace)

# Missing values
#train.isna().mean()[train.isna().any() == True].index
def rep_nas(data, inplace=False):
    for col in data.isna().mean()[data.isna().any() == True].index:
        dcol = data.loc[:,col]
        dcol.fillna(value=np.random.choice(dcol[dcol.isna() == False]),
                               inplace=inplace)

# Registration date column
def yr_mth(data, yrcol, mtcol, inplace=False):
    data.loc[:,'yr_mth'] = data.loc[:,[yrcol, mtcol]] \
                               .apply(lambda x: x[0] + x[1], axis=1)
    data.drop(columns=[yrcol, mtcol], inplace=inplace)

# Date type columns
def split_datetype(data, datcols, inplace=False):
    start = time.time()
    print("Splitting date columns...")

    for col in datcols:
        datcol = data.loc[:,col]
        data.loc[:,col + "_yr"] = datcol.apply(lambda x: int(str(x)[0:4]))
        data.loc[:,col + "_mt"] = datcol.apply(lambda x: int(str(x)[5:7]))
        data.loc[:,col + "_dy"] = datcol.apply(lambda x: int(str(x)[8:10]))
        #data.loc[:,col + "_hr"] = datcol.apply(lambda x: int(str(x)[11:13]))

    def calc_date(row):
        duration = date(row.lastSeen_yr,
                        row.lastSeen_mt,
                        row.lastSeen_dy) - date(row.dateCreated_yr,
                                                row.dateCreated_mt,
                                                row.dateCreated_dy)
        return duration.days

    dur = data.apply(lambda x: calc_date(x), axis=1)
    data.loc[:,'visiblePeriod'] = dur
    data.drop(columns=datcols, inplace=inplace)

    end = time.time()
    print("Duration:{}".format(end - start))

# Model attributes
def cat_dummies(data):
    start = time.time()
    print("\nCreating dummy variables...")
    dum_cols = data.loc[:,cat_cols] \
                   .columns.difference(spec_cols
                                       + redundant_cols
                                       + date_cols).drop('name')

    #print("data.columns:{}".format(data.columns))
    #print("dum_cols:{}".format(dum_cols))
    #print("Columns transformed to dummies: {}".format(dum_cols))

    dummies = pd.get_dummies(data.loc[:,dum_cols])
    #print("New dummy columns: {}".format(dummies.columns))
    w_dummies = pd.concat([data, dummies], axis=1)
    w_dummies = w_dummies.select_dtypes(exclude=[object]) \
                         .dropna(axis=1)

    return w_dummies

    end = time.time()
    print("{} seconds".format(end - start))

# Outliers
def drop_outliers(data):
    start = time.time()
    print("\nHandling outliers...")

    clf = IsolationForest(max_samples='auto',
                          random_state=2425,
                          contamination=0.01,
                          verbose=True,
                          n_jobs=-1)

    clf.fit(data)
    isof = clf.predict(data)

    data.loc[:,'Outlier'] = pd.Series(isof)
    outl_rows = data[data.Outlier == -1].index

    data.drop(outl_rows, inplace=True)
    data.drop(columns='Outlier', inplace=True)

    end = time.time()
    print("{} seconds".format(end-start))

# Correlated attributes
def drop_cors(data):
    start = time.time()
    print("Checking for attribute correlations...")

    #Nices solution from Chris Albon
    corrs = data.corr().abs()
    upper = corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
    print("Correlated attributes to drop:{}".format(to_drop))

    data.drop(data.loc[:,to_drop], axis=1, inplace=True)

    return to_drop

    end = time.time()
    print("{} seconds".format(end-start))

def preprocess(train, test,
               dropcors=True,
               outl_drop=True,
               repnas=False,
               repnanvals=True,
               neregdate=False,
               #oneregdate=False,
               usedatecols=True,
               usecats=True):
    cord_cols = []
    datasets = ['tr', 'te']
    training_rows = []

    X_train = np.array([])
    X_test = np.array([])

    for data in (train, test):
        #print("\nOriginal:{}".format(data.shape))
        #print("columns:\n{}\n".format(data.columns))

        if repnanvals == True:
            rep_nanvals(data, col_nanval, inplace=True)
            #print("nanvals:{}".format(data.shape))
            #print("columns:\n{}\n".format(data.columns))

        if repnas == True:
            rep_nas(data, inplace=True)
            #print("repnas:{}".format(data.shape))
            #print("columns:\n{}\n".format(data.columns))

        #if oneregdate == True:
        #    yr_mth(data, 'yearOfRegistration', 'monthOfRegistration', inplace=True)
        #    #print("yr_mth:{}".format(data.shape))
        #    #print("columns:\n{}\n".format(data.columns))

        if usedatecols == True:
            split_datetype(data, date_cols, inplace=True)
            #print("split_datetype:{}".format(data.shape))
            #print("columns:\n{}\n".format(data.columns))
            #print(data.dtypes)

        if usecats == True:
            data = cat_dummies(data)
            #print("cat_dummies:{}".format(data.shape))
            #print("columns:\n{}\n".format(data.columns))
        else:
            num_cols = data.select_dtypes(include=np.number).columns
            data = data.loc[:,num_cols]

        if datasets.pop(0) == 'tr':
            if outl_drop == True:
                drop_outliers(data)
                #print("drop_outliers:{}".format(data.shape))

            if dropcors == True:
                cord_cols = drop_cors(data)
                #print("drop_cors:{}".format(data.shape))

            train_idx = data.index
            #print("X_train columns:{}".format(data.columns))

            X_train = data

        else:
            if dropcors == True:
                data.drop(data.loc[:,cord_cols], axis=1, inplace=True)
                #print("drop_cors:{}".format(data.shape))
                #print("X_test columns:{}".format(data.columns))

            X_test = data

    X_train.drop(columns=X_train.columns \
                        [X_train.columns.isin(X_test.columns) == False], \
                        inplace=True)
    X_test.drop(columns=X_test.columns \
                        [X_test.columns.isin(X_train.columns) == False], \
                        inplace=True)

    print("X_train.shape:{}".format(X_train.shape))
    print("X_test.shape:{}".format(X_test.shape))
    return X_train, train_idx, X_test

# Feature Selection
def fsel(X, y, test_data, method, model=RandomForestClassifier, k=5):
    if method == 'univar':
        fsel_mod = SelectKBest(score_func=chi2, k=k)
        fsel_test = fsel_mod.fit(X, y)

        #print("Feature selection test scores:{}".format(fsel_test.scores_))
        features = fsel_test.transform(X)
        fnames = pd.DataFrame(data={'attribute': X.columns,
                                    'chi2': fsel_test.scores_}) \
                                    .sort_values(by='chi2', ascending=False) \
                                    .head(k).attribute.values
        print("Selected Features:{}".format(fnames))
        return features, fsel_test.transform(test_data)

    elif method == 'rfe':
        fsel_mod = RFE(model(verbose=1, n_jobs=-1), k)
        fsel_test = fsel_mod.fit(X, y)
        fnames = X.columns[fsel_test.support_]
        print("Selected Features:{}".format(fnames))
        return X.loc[:,fnames].as_matrix(), fsel_test.transform(test_data)

    elif method == 'pca':
        fsel_mod = PCA(n_components=k)
        fsel_test = fsel_mod.fit(X)
        print("Explained Variance:{}".format(fsel_test.explained_variance_ratio_))
        #print("Fit components:{}".format(fsel_test.components_))
        return fsel_test.transform(X), fsel_test.transform(test_data)

    elif method == 'fimp':
        fsel_mod = model()
        fsel_test = fsel_mod.fit(X, y)
        fnames = pd.DataFrame(data={'attribute': X.columns,
                                    'fimp': fsel_test.feature_importances_}) \
                                    .sort_values(by='fimp', ascending=False) \
                                    .head(k).attribute.values
        print("Selected Features:{}".format(fnames))
        return X.loc[:,fnames].as_matrix(), test_data.loc[:,fnames].as_matrix()

    else:
        print("You did not choose an existing method!")

# Standardization
def standardize(train, test):
    scaler = StandardScaler()
    scaler.fit(train)

    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test

# Normalization
def normalize(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)

    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test

models = {"RandomForestClassifier": RandomForestClassifier(max_features=0.25, criterion="entropy"), # n_estimators=100
          "Gradient Boosting Classifier": GradientBoostingClassifier(max_features='log2', n_estimators=500),
          "AdaBoost Classifier": AdaBoostClassifier(),
          "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
          #"LogisticRegression": LogisticRegression(C=1.5, penalty='l1'),
          #"GaussianNB": GaussianNB(),
          #"Decision Tree Cl - Gini": DecisionTreeClassifier(),
          #"Extra Tree Classifier": ExtraTreeClassifier(criterion='entropy', max_features='log2'),
          #"MLPClassifier": MLPClassifier(),
          #"K-NN_3 Classifier ": KNeighborsClassifier(3),
          #"QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(), # very bad performance
          #"Linear SVC": LinearSVC(),
          }

def eval_models(models, X, y):
    results = []
    for model in models:
        #print("Running {}...".format(model))
        #start = time.time()

        result = []
        result.append(model)

        model_score = cross_validate(models[model],
                                    X,
                                    y,
                                    scoring=['accuracy',
                                             'f1_micro',
                                             'f1_macro',
                                             'roc_auc'],
                                    cv=kfold,
                                    n_jobs=-1,
                                    verbose=2,
                                    return_train_score=False)

        acc_mean = model_score['test_accuracy'].mean()
        acc_std = model_score['test_accuracy'].std()
        auc_mean = model_score['test_roc_auc'].mean()
        auc_std = model_score['test_roc_auc'].std()

        print("\n{}:\n\tAccuracy: {} ({})".format(model, \
                                                  acc_mean, \
                                                  auc_std))
        print("\tROC AUC: {} ({})".format(auc_mean, auc_std))

        #if model != "Gradient Boosting Classifier":
        f1_micro_mean = model_score['test_f1_micro'].mean()
        f1_micro_std = model_score['test_f1_micro'].std()
        f1_macro_mean = model_score['test_f1_macro'].mean()
        f1_macro_std = model_score['test_f1_macro'].mean()
        print("\tF1 micro: {} ({})".format(f1_micro_mean, f1_micro_std))
        print("\tF1 macro: {} ({})".format(f1_macro_mean, f1_macro_std))

        #result = result + [acc_mean, acc_std, auc_mean, auc_std]

        dur = model_score['fit_time'].sum() + model_score['score_time'].sum()

        print("\tduration:{}\n".format(dur))
        #result.append(dur)

        #results.append(result)


X_train_df, train_idx, X_test_df = preprocess(train.loc[:,train.columns.difference(['id', 'label'])].copy(),
                                              test.loc[:,test.columns.difference(['id'])].copy(),
                                              dropcors=True,
                                              outl_drop=True,
                                              repnas=False,
                                              repnanvals=True,
                                              #oneregdate=False,
                                              usedatecols=True,
                                              usecats=True)

y_train = train.loc[train_idx, 'label'].copy()

X_train, X_test = fsel(X_train_df.copy(),
                       y_train,
                       test_data=X_test_df.copy(),
                       method='fimp',
                       model=RandomForestClassifier,
                       k=90)

X_train, X_test = standardize(X_train, X_test)
X_train, X_test = normalize(X_train, X_test)

# Modeling
seed = np.random.randint(1000)
print("\nseed:{}".format(seed))

kfold = KFold(n_splits=10, random_state=seed)

#results = eval_models(models, X_train, y_train)
#print("results:{}".format(results))

# Grid search
def param_search(train, target, model, pars):
    start = time.time()

    print("Starting grid search for parameters...")
    mod = model()
    grid_m = GridSearchCV(
        estimator=mod,
        param_grid=pars.param_grid,
        scoring=['accuracy',
                 'f1_micro',
                 'f1_macro',
                 'roc_auc'],
        n_jobs=-1,
        verbose=2,
        refit='roc_auc',
        iid=False)
    grid_m.fit(train, target)
    print(grid_m.cv_results_)

    return grid_m
    print("Duration:{}".format(time.time() - start))

rf_params = ParameterGrid({
    #'criterion': ['gini', 'entropy'],
    #'max_depth': [None], # , 5, 4, 3
    'max_features': [0.35, 0.4, 0.3], #'sqrt', 'log2'
    'min_samples_split': [3, 5, 4], #0.01
    'n_estimators': [100, 200], # 1000
    # 'min_samples_leaf': [0.01],
    #'min_weight_fraction_leaf': [0.01],
    #'min_impurtity_decrease': [0, 0.1, 0.2, 0.3],
    'n_jobs': [-1],
    'random_state': [seed],
    'verbose': [0],
    #'class_weight': [],
    })

rf_gridCV_2 = param_search(X_train, y_train, RandomForestClassifier, rf_params)
print(rf_gridCV_2.best_score_)

pd.DataFrame(rf_gridCV_2.cv_results_).to_csv("last_results.csv")

# Repeated cross validation
print("Running model...")
start = time.time()
model = rf_gridCV_2.best_estimator_
kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
cv_results = cross_validate(model,
                            X_train,
                            y_train,
                            scoring=['accuracy', 'roc_auc'],
                            cv=kfold,
                            n_jobs=-1,
                            verbose=2,
                            return_train_score=False)
print("Test accuracy:{}".format(cv_results['test_accuracy'].mean()))
print("Test ROC AUC:{}".format(cv_results['test_roc_auc'].mean()))
end = time.time()
print("Duration:{}".format(end - start))

pd.DataFrame(cv_results).to_csv("rep_cv_res.csv")

#Getting the 1 confindencies
print("X_test.shape:{}".format(X_test.shape))
model = rf_gridCV_2.best_estimator_
model.fit(X_train, y_train)
confs = model.predict_proba(X_test)[:,1]
confs = pd.DataFrame([test.id, confs.reshape(-1,)]).transpose()
confs.rename(columns={"Unnamed 0": 'label'}, inplace=True)
confs.to_csv("late.csv", index=False)

print("\a")
