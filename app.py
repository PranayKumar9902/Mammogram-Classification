import numpy as np
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)
model = joblib.load(open('RF.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':
        text1 = request.form['Age']
        text2 = request.form['Shape']
        text3 = request.form['Margin']
        text4 = request.form['Density']
    calls=[text1,text2,text3,text4]
    


    #    text=np.array(text)
    #    data = [text]
    prediction = model.predict([calls])
    if prediction[0]:
        return render_template('index.html', prediction_text='The Mammogram is malignant')
    else:
        return render_template('index.html', prediction_text='The Mammogram is benign')
    
if __name__ == "__main__":
    app.run(debug=True)