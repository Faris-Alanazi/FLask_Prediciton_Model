from markupsafe import escape
from flask import Flask,url_for ,render_template,request
import numpy as np
import pickle

Stroke_Model = pickle.load(open('Stroke-Modelv-RandomForest.pickle', "rb")) # Trained NN Model
# these are the Featuers the model trained on
# ['age','hypertension','heart_disease','smoking_status_never smoked','smoking_status_smokes']

app = Flask(__name__)
@app.route('/')
def page():

    return render_template('form.html')

@app.route('/Predicting',methods=['POST','GET'])
def model():

    form_data = request.form # gathering the form input
    data = [int(x) for _,x in form_data.items()] # extracting the values and leaving the keys
    # we will only receive 4 inputs and we depending on the smoking status we fill the last one either (0,1)
    if(data[3]==1): # to fill the last dummy varible
        data.append(0)  
    else:
        data.append(1)

    predict_data = np.array([data[0],data[1],data[2],data[3],data[4]]).reshape(1,5)
    pred = Stroke_Model.predict(predict_data)
    print(data)
    print(predict_data)
    print(pred[0])
    output = ''
    if pred[0]==1:
        output = "The Model Predicted That You Might Have a Stroke"
    else:
        output = "The Model Predicted That You Wouldn't Have a Stroke"

    return render_template('show.html',pred=output)


app.run(host='localhost', port=5000)


                                  