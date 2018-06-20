# ebay-car-pricing
Predicting whether a specific car is going to be overpriced within its own category. Based on ebay data.

# Context
We have a training data set showing car adverts also telling us whether the specific car has been sold on a price above its category average (no further definition available). Our task was to also predict this also for a test data set and to upload our prediction coefficients (for the '1' value) to a Kaggle competition space. After each upload, the system returned a ROC AUC metric for 30% of the test data. We received our results for the whole data only after the end of the competition.

# Data available
Source: Ebay Kleinanzeigen

* autos_training_final.csv - training data
* autos_testing_final.csv - test data
* autos_submission.csv - an example file showing the format of the data to submit

# Attributes
* dateCrawled: The saving date. Every other value is based on this one.
* name : The car's name.
* seller : Whether the seller is "private" or a "dealer".
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
