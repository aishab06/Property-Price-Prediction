import numpy as np
import pandas as pd
import pickle
from datetime import date
from flask import Flask, request, render_template

with open('House_Price_Prediction_Model.pkl', 'rb') as t:
    reg = pickle.load(t)

app = Flask(__name__)

FEATURE_NAMES = list(reg.feature_names_in_)

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    form = request.form

    input_df = pd.DataFrame(np.zeros((1, len(FEATURE_NAMES))), columns=FEATURE_NAMES)

    today = date.today()
    if 'country' in input_df.columns:
        input_df['country'] = 1
    if 'year' in input_df.columns:
        input_df['year'] = today.year
    if 'month' in input_df.columns:
        input_df['month'] = today.month
    if 'day' in input_df.columns:
        input_df['day'] = today.day

    numeric_fields = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                       'waterfront', 'view', 'condition', 'sqft_basement', 'yr_built']

    for field in numeric_fields:
        if field in input_df.columns and form.get(field):
            input_df[field] = float(form.get(field))

    selected_city = form.get('city')
    if selected_city:
        city_col = f'city_{selected_city}'
        if city_col in input_df.columns:
            input_df[city_col] = True
        # else: selected_city is the baseline city drop_first=True dropped —
        # all city_* columns staying 0 already represents it correctly

    sol = reg.predict(input_df)[0]
    return render_template('index.html', prediction_text=round(sol, 2))


if __name__ == '__main__':
    app.run(debug=True)