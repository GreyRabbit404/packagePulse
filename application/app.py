from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl','rb'))

Rmodel = pickle.load(open('cosine_sim.pkl','rb'))
filter = pd.read_csv('food.csv')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendui')
def indexr():
    return render_template('recommend.html')

@app.route('/predictui')
def indexp():
    return render_template('predict.html')

# Define a route for receiving food_name and returning recommendations
@app.route('/recommendations', methods=['POST'])
# Load the recommendation function
def get_recommendations( cosine_sim=Rmodel, df=filter):
    food_name = request.form.get('Name')

    idx = df[df['name'] == food_name].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the first item (itself)

    food_indices = [i[0] for i in sim_scores]
    res = df['name'].iloc[food_indices].values.tolist()

    print(res)
    return render_template('recommend.html',res = res)


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

    return render_template('predict.html',result = result)


if __name__ =='__main__':
    app.run(host= '0.0.0.0',port=8080)