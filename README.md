This is a tool for working with the [Community Health Status Indicators](http://www.healthdata.gov/dataset/community-health-status-indicators-chsi-combat-obesity-heart-disease-and-cancer) dataset.

Most of the functionality is in [data_handler.py](data_handler.py), which defines the `CSHIDataHandler` class, which is  initialized with the following parameters:

  + `data_dir` - the path to the CHSI dataset
  + `dependent` - the name of the dependent variable
  + `exclude_cols` - a list of columns to be excluded
  + `threshold` - the proportion of values which must be present to include a predictor
  
Once initialized, the `training_data()` method returns a tuple `X,Y` , a Pandas DataFrame and Series which hold the predictors and dependent variable with various cleaning already completed.

Other useful methods include:

  + `data_element` - looks up an indicator in the DATAELEMENTDESCRIPTION.csv file.
  + `export_data` - exports indicator data file to CSV. Extra columns (e.g. predicted values of the dependent variable) can be included by specifying a Pandas DataFrame for the `extra_columns` parameter.
  + `all_county_data` - returns all county-level data in a single DataFrame