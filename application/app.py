from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict_food():
    type = request.form.get('Type')
    palmoil = str(request.form.get('PalmOil'))
    addedsugar = float(request.form.get('AddedSugar'))
    foodcolor = int(request.form.get('FoodColor'))
    fat = float(request.form.get('Fat'))
    sodium = float(request.form.get('Sodium'))
    protein = float(request.form.get('Protein'))


    #predict

    result = model.predict(np.array([type,palmoil,addedsugar,foodcolor,fat,sodium,protein]).reshape(1,7))

    if result[0] == 1 :
        result =  "Good Food"
    else:
        result =  "Not Good For Health"

    return render_template('index.html',result = result)


if __name__ =='__main__':
    app.run(host= '0.0.0.0',port=8080)