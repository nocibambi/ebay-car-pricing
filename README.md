# ebay-car-pricing
Using scikit-learn to predict whether a specific car is going to be overpriced within its own category. Based on ebay data.

You can read the actual script in the [this notebook](autos_sold_above_avg.ipynb).

# Context
We have a training data set showing car adverts also telling us whether the specific car has been sold on a price above its category average (no further definition available). Our task was to predict this value also for a test data set and to upload our prediction coefficients (for the '1' value) to a Kaggle competition space. After each upload, the system returned a ROC AUC metric for 30% of the test data. We received our results for the whole data only after the end of the competition.

# Data
The files are in `autos_data.zip`. You need to download and compress them into the same directory where you store the notebook.

Source: Ebay Kleinanzeigen
* autos_training_final.csv - training data
* autos_testing_final.csv - test data
* autos_submission.csv - an example file showing the format of the data to submit

## Attributes
* dateCrawled: The saving date. Every other value is based on this one.
* name : The car's name.
* seller : Whether the seller is
* offerType
* vehicleType: The car type.
* yearOfRegistration: The car's registration year.
* gearbox
* powerPS : PowerPS
* model: Car model
* kilometer: Kilometers in the car.
* monthOfRegistration: The car's registration month.
* fuelType: Fuel type.
* brand: Car brand.
* notRepairedDamage: Whether there is any unrepaired damage on the car.
* dateCreated: The ad's creation date.
* nrOfPictures: Number of pictures attached to the ad.
* postalCode: Postal code.
* label: The target value. Tells whether the car's price is higher than its category's average price.
* lastSeenOnline: The last time the ad has been seen online.

# Analysis steps
## Data cleaning and preprocessing
I summarized the whole process into a function so, when needed, we could also create new data sets with different preprocessing parameters.

The covered steps:
- Drop correlated values
- Drop outliers
- Replace attribute's own missing value codes
- Replace missing values
- Splitting DateTime columns into separate year, months and day attributes
- Transforming categorical values into dummy attributes
- Automatic feature selection based on one of the following methods:
    - Univariate feature selection based on chi-squared test
    - Recursive Feature Elimination
    - Principal Component Analysis
    - Feature importance
- Standardization
- Normalization

## Model evaluation
I run trial predictions with the following models on a limited size of the data:
- RandomForestClassifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- LinearDiscriminantAnalysis
- LogisticRegression
- GaussianNB
- Decision Tree Cl - Gini
- Extra Tree Classifier
- MLPClassifier
- K-NN_3 Classifier
- QuadraticDiscriminantAnalysis
- Linear SVC

While we are looking for ROC AUC, I also calculated Accuracy and F1 metrics. Based on the results I kept the top one third models based on their AUC scores:
- RandomForestClassifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- LinearDiscriminantAnalysis

I run trial models with them on the whole dataset.

## Grid search
Because the best results came from the RandomForestClassifier, I also run a grid search along the following parameters:
- Max features within a split: [0.35, 0.3]
- Minimum sample split: [3, 4]
- Number of estimators: [10, 100, 200]

This took a long time to run on my laptop.

## Final validation
After finding the parameters with the best results, I also run a Repeated Stratified K-Fold validation.
