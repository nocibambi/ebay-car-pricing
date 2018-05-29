# new tasks
* using feature importance on test data too

Read about ROC AUC
Examine the coefficient distributions

- read about performance measures and metrics:
    - response chart
    - lift curve
    - micro, macro, f1, etc?
    Verifying test performance: use multiple runs of k-fold cross validation with statistical significance tests
    - Create a Confusion matrix
        - Example
        ```py
        from sklearn.metrics import confusion_matrix
        expected = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
        predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
        results = confusion_matrix(expected, predicted)
        ```
    - Test against possible imbalanced data

- Predict missing values
    - http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values
    - http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html

~~Tune model parameters~~
    * Random Forest
        - "The number of features that can be searched at each split point (m) must be specified as a parameter to the algorithm. You can try different values and tune it using cross validation."
            - For classification a good default is: m = sqrt(p)
            - For regression a good default is: m = p/3
            Where m is the number of randomly selected features that can be searched at a split point and p is the number of input variables.
    ~~- Read about models~~

Read about how to handle hierarchical attributes
express links and relations between columns

Examine where the coefficient is low

* Other algorithms
    - try XGBoost (from machinelearningmastery)
    - Gaussian SVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
        - this is slow, perhaps try with BaggingClassifier

* Read about ensemble methods
    * https://machinelearningmastery.com/faq/single-faq/can-you-help-me-with-ensembles
    * This may be achieved through many ways. Three ensemble strategies you can explore are:
        "* Bagging: Known more formally as Bootstrapped Aggregation is where the same algorithm has different perspectives on the problem by being trained on different subsets of the training data.
        * Boosting: Different algorithms are trained on the same training data.
        * Blending: Known more formally as Stacked Aggregation or Stacking is where a variety of models whose predictions are taken as input to a new model that learns how to combine the predictions into an overall prediction."

~~Control excess weight created by dummy categoricals~~
Manual weighting attributes (weight of evidence?)
Feature engineering

- Think through correct order of cleaning steps:
    * Clustering
    * Dimensionality reduction
    * Feature selection
    * Feature engineering
    * Modeling

Read about imbalanced data
~~Read about how to make KNN, SVC, LinearSVC, GaussianProcessC, MLPC etc faster~~
Read about reindex() and how should I use it instead of pasing lists in .loc

~~pipelining with scikit learn?~~

~~* check how MinMaxScaler is different from Normalize in sklearn (http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)~~
* Check whether I should binarize data? http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html


Try spark or amazon aws for faster execution


* Read about LeaveOneOut
    ```py
    from sklearn.model_selection import LeaveOneOut
    loocv = LeaveOneOut()
    results = model_selection.cross_val_score(model, X, Y, cv=loocv)
    ```
Create a parameter logger

Clean the name attribute and create new attributes from it
Examine for random variables
* Process the test and training data together
    1. Flagging each rows in the two data sets as traingin and testing
    2. Concatenating the two data sets
    3. Doing the following steps together:
        1. Normalization
        2. ...?

Do RepeatedStratifiedKFold before final submission

## Done
~~include date columns~~
~~convert date columns into integer~~

~~read about algorithm selection~~
~~read about normalization and standardization~~

~~Try MinMaxScaler~~
~~Spot-check algorithms~~
    ~~- might try on just a sample of the data~~
    ~~- this can help to overcome my old habits~~

~~Take notes about the data~~
~~Examine data for outliers~~
~~Examine correlated featur~~es
~~Read Jason's new posts~~
~~Convert categorical data: `pd.get_dummies()`~~
~~Read about handling date variables~~
~~Read about handling categorical attributes in python~~
~~Examine other models~~

~~take a respresentative/stratified sample~~
~~examine variables relationship to the label~~
~~examine attributes for~~
- ~~values themselves~~
- ~~distribution~~
- ~~missing values~~
~~examine the redundant variables (if they are good predictors)~~
~~Create registration date column~~
~~Examine data distribution~~
~~* Recognize and create variables for special time periods:~~
    - https://www.salford-systems.com/blog/dan-steinberg/using-dates-in-data-mining-models
    * Calculate number of days

~~Create a data cleaner function~~
~~Read about car powerPS~~
~~Read about algorithm selection methods~~
~~Apply ensemble methods~~
~~Convert date into categoricals~~
~~Handle categorical values with scikit learn~~
~~gradient boosting classifier~~
~~Select features automatically~~
- ~~run the top 5 algorithms with the advised parameters~~
- ~~* calculate duration of being advertized~~
~~* standardization?~~
~~Read about Grid search~~
~~Run grid selector~~
