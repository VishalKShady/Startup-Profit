from flask import Flask
import numpy as np
import pickle
from flask import render_template, jsonify, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Hello" + render_template('index.html')

@app.route('/pred', methods=['POST'])
def predict():
    '''The prediction page
       for profit of startup'''

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    return render_template('index.html', prediction_text = 'Profit should be $ {}'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    prediction = model.predict([np.array(list(data.values()))])
 
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(port='9000')
