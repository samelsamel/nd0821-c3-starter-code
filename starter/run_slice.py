'''
Author: Amel Sellami
Date: 15-05-2022
Goal: running the slice function on slices of
the education column and output recall, precision and fbet scores.
'''
# Script to train machine learning model.
import pandas as pd
# Add the necessary imports for the starter code.
from starter.train_model import batch_inference


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# EDUCATION VALUES COUNTS:
education_income = [
    "HS-grad",
    "Some-college",
    "Bachelors",
    "Masters",
    "Assoc-voc",
    "11th",
    "Assoc-acdm",
    "10th",
    "7th-8th",
    "Prof-school",
    "9th",
    "12th",
    "Doctorate",
    "5th-6th",
    "1st-4th",
    "Preschool"
    ]


def create_data_slice(data_path, col_to_slice, value_to_replace=None):
    input_df = pd.read_csv(data_path)
    # Add code to load in the data.
    if value_to_replace:
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: str(value_to_replace)
        )

    else:
        input_df[col_to_slice] = input_df[col_to_slice].apply(
            lambda x: input_df[col_to_slice][0]
        )

    return input_df


if __name__ == '__main__':
    col_to_slice = 'education'

    for col in education_income:
        value_to_replace = col
        print("performance on sliced column\t", col_to_slice, value_to_replace)
        sliced_data = create_data_slice('data/census_clean.csv',
                                        col_to_slice,
                                        value_to_replace)

        precision, recall, fbeta = batch_inference(sliced_data,
                                                   "model/model.pkl",
                                                   CAT_FEATURES)

        with open('slice_output.txt', 'a') as f:
            result = f"""\n{'-'*50}\nperformance on sliced column -- \
            {col_to_slice} -- {value_to_replace}\n{'-'*50} \
                \nPrecision:\t{precision}\nRecall:\t{recall}\n \
                F-beta score:\t{fbeta}\n"""
            f.write(result)
