from starter.train_model  import trainer, load_data

if __name__ == '__main__':
    data_path = 'data/census_clean.csv'
    model_path = "/home/amel/work/udacity/nd0821-c3-starter-code/model/random_forest_model_encoder_lb.pkl"

    # Get the splitted data
    train_data, test_data = load_data(data_path)
    # Training the model on the train data
    trainer(train_data, model_path)
    # evaluating the model on the test data
    # Test model and show metrics