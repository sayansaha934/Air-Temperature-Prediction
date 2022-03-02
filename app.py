from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

df=pd.read_csv('ai4i2020.csv')
df=df.drop(columns=['UDI','Product ID','Type'])
x=df.drop(columns=['Air temperature [K]'])
scaler=StandardScaler()
x=scaler.fit_transform(x)

@app.route('/', methods=['POST','GET'])
def index():
    try:
        if request.method=='POST':
            ptemp = float(request.form['ptemp'])
            rspeed = int(request.form['rspeed'])
            torq = float(request.form['torq'])
            toolwr = int(request.form['toolwr'])
            mfail= int(request.form['mfail'])
            twf = int(request.form['twf'])
            hdf = int(request.form['hdf'])
            pwf = int(request.form['pwf'])
            osf = int(request.form['osf'])
            rnf = int(request.form['rnf'])
            scaled_data=scaler.transform([[ptemp,rspeed,torq,toolwr,mfail,twf,hdf,pwf,osf,rnf]])
            model=pickle.load(open('temp_analysis.pickle','rb'))
            result=model.predict(scaled_data)[0][0]
            return render_template("index.html", result='Air Temp:'+' '+str(result)+' '+'K')

        else:
            return render_template("index.html")
    except Exception as e:
        return "Something went wrong"
@app.route('/analysis', methods=['POST','GET'])
def analysis():
    return render_template("analysis.html")
if __name__ == "__main__":
    app.run(port=8000)