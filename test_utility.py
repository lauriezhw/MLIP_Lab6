import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    return_tuple = data_split(*feature_target_sample)
    # TODO test if the length of return_tuple is 4
    assert len(return_tuple) == 4, "The data_split function should return 4 items"

    # Unpack the return tuple
    X_train, X_test, y_train, y_test = return_tuple

    # Test that all returned objects are not empty
    assert len(X_train) > 0, "X_train should not be empty"
    assert len(X_test) > 0, "X_test should not be empty"
    assert len(y_train) > 0, "y_train should not be empty"
    assert len(y_test) > 0, "y_test should not be empty"

    # Test that the sizes are correct (using the actual 67-33 split)
    total_samples = len(feature_target_sample[0])
    expected_train_size = int(total_samples * 0.67)  # Round down for small samples
    expected_test_size = total_samples - expected_train_size

    assert len(X_train) == expected_train_size, f"X_train should be approximately 67% of the data ({expected_train_size} samples)"
    assert len(X_test) == expected_test_size, f"X_test should be approximately 33% of the data ({expected_test_size} samples)"
    assert len(y_train) == len(X_train), "y_train should have the same length as X_train"
    assert len(y_test) == len(X_test), "y_test should have the same length as X_test"

    # Test that X_train and X_test are mutually exclusive
    assert set(X_train.index).isdisjoint(set(X_test.index)), "X_train and X_test should not overlap"