from PyCrowlingo import Client

crow_client = Client('[YOUR_API_KEY]')

DATA_PATH = './data.csv'
MODEL_NAME = 'CrowlingoTutorialIntent'

# create the model
crow_client.model.create(MODEL_NAME, "clf")

# upload the data
crow_client.classifier.upload_csv(MODEL_NAME, DATA_PATH, fieldnames=["text", "class"])

# train the model
crow_client.model.train(MODEL_NAME, model_type="deep")

# wait for the training to finish
metrics = crow_client.model.wait_training(MODEL_NAME)

# test the model

TEST_WITH_TRAD = [
('How do I cancel my insurance?', "Comment résilier mon assurance ?"),
('How do we say cow in german?', "Come si dice mucca in tedesco?"),('What is the meaning of life?', "В чем смысл жизни?"),
('Where were you born?', "あなたはどこで生まれましたか？"),
('tell me something interesting about crows', "dime algo interesante sobre los cuervos"),
('how do I set the language setting to german?', "Wie stelle ich die Spracheinstellung auf Deutsch ein?")]

for test in TEST_WITH_TRAD:
    trad, text = test
    res = crow_client.classifier.classify(MODEL_NAME, text)
    print(f'{trad} : {res.classes[0].class_id}')