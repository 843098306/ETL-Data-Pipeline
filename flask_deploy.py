import flask 
import requests
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn import preprocessing

app = flask.Flask(__name__)

# Ordial Encode of Categorical Value for input
def prepare_input(X):
	oe = preprocessing.OrdinalEncoder()
	oe.fit(X)
	Output = oe.transform(X)
	return Output
    
def prepare_target(Y):
	le = preprocessing.LabelEncoder()
	le.fit(Y)
	Output = le.transform(Y)
	return Output

def get_model():
    print("Loading Model!")
    global model
    model = load_model("Titanic.h5")
    print("Model Loaded!")

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    message = flask.request.get_json(force = True)
    url = message['api']
    temp = requests.get(url)
    data = temp.json()
    # column_train_feature_name = message['label']
    df = pd.DataFrame.from_dict(data['Rent'], orient="columns")
    print(df)
    x = df.drop(['rooms'], axis=1)
    x = prepare_input(x)
    y = df['rooms']
    y = prepare_target(y)
    scores = model.evaluate(x,y)
    
    Dict = {}
    for i in range(len(model.metrics_names)):
        print(model.metrics_names[i],scores[i])
        Dict[model.metrics_names[i]] = scores[i]
        
    response = {"prediction":Dict}
    return response
    

if __name__ == '__main__':
    get_model()
    app.run()