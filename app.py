from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pickle

app = Flask(__name__)




def get_categories(data):
    c_f = {
        'Club' : ['Club Class C', 'Club Class B'],
    }

    cate_df = pd.DataFrame()
    for col in c_f.keys():
        category = data[col].values
        if category in c_f[col]:
            for cat in c_f[col]:
                if cat == category:
                    cate_df[col + f'_{cat}'] = [1]
                else:
                    cate_df[col + f'_{cat}'] = [0]
        else:
            for cat in c_f[col]:
                cate_df[col + f'_{cat}'] = [0]
    return cate_df


def transform_data(data):
    num_fea = ['Overall', 'Potential', 'Wage', 'International Reputation', 
        'Skill Moves', 'Release Clause', 'Contract Expire Year']
    num_data = data[num_fea]
    for col in num_data.columns:
        num_data[col] = num_data[col].astype('int64')
    cate_data = get_categories(data)
    final_data = pd.concat([num_data, cate_data], axis=1)
    rearrange = ['Overall', 'Potential', 'Wage', 'International Reputation', 
        'Skill Moves', 'Release Clause', 'Club_Club Class C', 'Club_Club Class B', 'Contract Expire Year']
    final_data = final_data[rearrange]
    min_scale = pickle.load(open('scale_object_inputs.pkl', 'rb'))
    final_data = pd.DataFrame(min_scale.transform(final_data), columns=rearrange)
    
    return final_data


def make_predictions(data):

    final_model = pickle.load(open("fifa_2019_players_value_model.pkl", "rb"))
    prediction = final_model.predict(data)
    scale = pickle.load(open('scale_object_target.pkl', 'rb'))
    prediction = scale.inverse_transform(np.array(prediction).reshape(-1,1))
    return round(prediction[0][0],2)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clubs')
def countries():
    return render_template('clubs.html')

@app.route('/prediction', methods=['POST'])
def prediction():

    data = pd.DataFrame()
    features = [ 'Overall', 'Potential', 'Wage', 'International Reputation', 'Skill Moves',
        'Release Clause', 'Club', 'Contract Expire Year' ]

    for index in range(len(features)):
           value = request.form[features[index]]
           data[features[index]] = [value]

    final_data = transform_data(data)
    #print(final_data)
    return render_template('predictions.html', predictions= make_predictions(final_data))

if __name__ == "__main__":
    app.run(debug=True)