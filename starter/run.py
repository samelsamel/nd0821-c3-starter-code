"""
Author: Amel Sellami
Date: 28-04-2022
Goal: loading data, run the train command and save the model.
"""
from starter.train_model import trainer, load_data

if __name__ == '__main__':
    data_path = 'data/census_clean.csv'
    model_path = "path/to/file/model/random_forest_model_encoder_lb.pkl"

    # Get the splitted data
    train_data, test_data = load_data(data_path)
    # Training the model on the train data
    trainer(train_data, model_path)
    # evaluating the model on the test data
    # Test model and show metrics
