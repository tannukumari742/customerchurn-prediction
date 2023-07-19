
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)  
rfc = joblib.load("Customer_churn.pkl")  
df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('avi.html')

@app.route('/predict' ,methods=['POST'])
def predict():
    global df

    # Get the values from the form and convert them to integers
    independent_var = [int(x) for x in request.form.values()]
    features_value = np.array(independent_var)

    # Perform the prediction using the model (rfc in this case)
    output = rfc.predict([features_value]).round(2)

    df = pd.concat([df, pd.DataFrame({'credit_score':independent_var[0], 'age':independent_var[1],'tenure':independent_var[2] , 'balance':independent_var[3], 'products_number':independent_var[4],'credit_card':independent_var[5],'estimated_salary':independent_var[6] ,'Predicted Sales':[output]})], ignore_index=True)

    df.to_csv("resultant.csv")


    return render_template('avi.html', prediction_text='{} CHURN'.format(output, int(features_value[0])))


if __name__== "__main__" :
    app.run()


    