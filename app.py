import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('info.html')

@app.route('/predict',methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    pred = model.predict(final_features) # making prediction
    
    return render_template('prediction.html', prediction_text='Prediction: Number of sales after the advertisement with different sources is {}' .format(pred)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)
