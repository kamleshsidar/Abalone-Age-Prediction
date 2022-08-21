from flask import Flask, render_template, request
import pickle
import numpy as np

f = open('rf_model.pkl', 'rb')
model = pickle.load(f)
f.close()
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    le = float(request.form['length'])
    
    dia = float(request.form['diameter'])
    
    he = float(request.form['height'])
    
    whole_wt = float(request.form['whole_weight'])
    
    shucked_wt = float(request.form['shucked_weight'])
    
    viscera_wt = float(request.form['viscera_weight'])
    
    shell_wt = float(request.form['shell_weight'])
    
    gender = request.form['gender']
    
    sex_i = 1 if gender=='infant' else 0
    
    sex_m = 1 if gender=='male' else 0

    arr = [[le, dia, he, whole_wt, shucked_wt,viscera_wt,shell_wt,sex_i, sex_m]]
    
    pred = np.round(model.predict(arr)[0],1)
    
    return render_template('home.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















