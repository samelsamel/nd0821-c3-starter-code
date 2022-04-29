'''
Author: Amel Sellami 
Date: 28-04-2022
Goal: Testing for starter functions
'''
import numpy as np

def test_columns_names(data):
    expected_columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income"
    ]

    curr_columns = data.columns.values
    assert list(expected_columns) == list(curr_columns), \
        print("Column names in the input data doesn't match")


def test_age_range(data, min_age=0, max_age=100):
    """To check if price values lies within reasonable range."""

    idx = data['age'].between(min_age, max_age)

    assert np.sum(~idx) == 0, \
        print("Age column has outliers...")


def test_relationship_category(data):
    known_relationship_values = [
        'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']

    relationships = set(data['relationship'].unique())
    print(f'known relations: {relationships}')
    # Unordered check
    assert set(known_relationship_values) == set(relationships), \
        print("relationships values in the input data doesn't match")