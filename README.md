🏡 Property Price Prediction Using Multiple Linear Regression
📊 End-to-End Machine Learning Web Application

Predicting property prices based on bedrooms, bathrooms, living area, lot size, floors, waterfront, view, condition, basement area, year built, and city using Python • Scikit-learn • Flask • Gunicorn

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?logo=scikitlearn&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-Web%20App-black?logo=flask&logoColor=white) ![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)

Data Analysis • Feature Engineering • Machine Learning • Model Evaluation • Flask • Cloud Deployment

---

## 📌 Project Overview

This project demonstrates the complete end-to-end Machine Learning lifecycle, starting from a raw housing dataset and finishing with a web application that predicts property prices in real time.

The goal of the project is to build a **Multiple Linear Regression** model that predicts a property's price based on multiple features of the property — not just one, as in a simple linear regression problem.

The project covers:

    📂 Data Collection
    🔍 Exploratory Data Analysis
    🧹 Data Cleaning & Feature Engineering
    🏷️ Categorical Encoding (One-Hot Encoding)
    ✂️ Train-Test Split
    🧠 Machine Learning Model Training
    📊 Model Evaluation
    🔮 Custom Predictions
    💾 Model Serialization using Pickle
    🌐 Flask Web Application Development
    🦄 Gunicorn Production Server
    🐙 GitHub Version Control
    ☁️ Cloud Deployment

---

## 🎯 Problem Statement

A property's price is influenced by several factors at once — its size, condition, location, age, and amenities. A single-variable model cannot capture this.

The objective of this project is to understand the relationship between multiple property features and price:

```
Bedrooms, Bathrooms, Living Area, Lot Size,
Floors, Waterfront, View, Condition,
Basement Area, Year Built, City
        │
        ▼
   Property Price
```

and build a Machine Learning model capable of predicting price for previously unseen property combinations.

---

## 📂 Dataset

The dataset used in this project is:

```
data.csv
```

| Property | Value |
|---|---|
| Target Variable | `price` |
| Numeric Features | `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_basement`, `yr_built` |
| Categorical Features | `city` (one-hot encoded), `country` |
| Date Feature | `date` (decomposed into `year`, `month`, `day`) |

> Replace this table with your actual row/column counts once you've inspected `data.csv` — run `dataset.shape` and `dataset.info()` and paste the real numbers here for an accurate README.

### Sample Dataset

```
date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_basement,yr_built,city,country
2014-05-02,313000,3,1.5,1340,7912,1.5,0,0,3,0,1955,Shoreline,USA
2014-05-02,2384000,5,2.5,3650,9050,2,0,4,5,0,1921,Seattle,USA
2014-05-02,342000,3,2.0,1930,11947,1,0,0,4,0,1966,Kent,USA
```

---

## 🧠 Machine Learning Algorithm

### Multiple Linear Regression

This project uses **Multiple Linear Regression**, a supervised Machine Learning algorithm used to model the relationship between:

- Several independent variables (features)
- One dependent variable (price)

The mathematical equation is:

```
y = b0 + b1·x1 + b2·x2 + b3·x3 + ... + bn·xn
```

Where:

- `y` = Predicted Price
- `x1, x2, ... xn` = Property features (bedrooms, bathrooms, sqft_living, city_Redmond, ...)
- `b1, b2, ... bn` = Coefficients learned for each feature
- `b0` = Intercept

The algorithm finds the best-fitting hyperplane through the available data, rather than a single line, since multiple features are involved.

---

## 🔄 Complete Project Workflow

```
📂 data.csv
    │
    ▼
🔍 Exploratory Data Analysis
    │
    ▼
🧹 Clean & Encode (country, city one-hot, date → year/month/day)
    │
    ▼
🧮 Prepare Features (X) & Target (y)
    │
    ▼
✂️ Train-Test Split
   80%       20%
    │         │
    ▼         ▼
🏋️ Train    🧪 Test
    │
    ▼
🧠 Multiple Linear Regression
    │
    ▼
📊 Model Evaluation (R², RMSE)
    │
    ▼
🔮 Custom Predictions
    │
    ▼
💾 Save Model.pkl
    │
    ▼
🌐 Flask Web Application
    │
    ▼
🦄 Gunicorn
    │
    ▼
🐙 GitHub
    │
    ▼
☁️ Cloud Deployment
    │
    ▼
🌍 Public ML Application
```

---

## 🛠️ Tech Stack

Python • Pandas • NumPy • Matplotlib • Scikit-learn • Flask • HTML/CSS • Git • GitHub • Gunicorn

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🐼 Pandas | Dataset loading, cleaning, and manipulation |
| 🔢 NumPy | Numerical operations |
| 📊 Matplotlib | Data visualization and EDA |
| 🤖 Scikit-learn | Machine Learning model, train/test split, metrics |
| 💾 Pickle | Saving the trained model |
| 🌶️ Flask | Backend web framework |
| 🌐 HTML/CSS | Frontend user interface |
| 🦄 Gunicorn | Production WSGI server |
| 🐙 Git/GitHub | Version control and source hosting |
| ☁️ Render (or similar) | Cloud deployment |

---

## 🔍 Exploratory Data Analysis & Feature Engineering

Before training, the raw dataset needs several transformations:

```python
import pandas as pd

df = pd.read_csv("data.csv")

# Country is constant (only USA present) — map to numeric
df['country'] = df['country'].map({'USA': 1})

# City is categorical — one-hot encode it
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Break the listing date into separate numeric parts
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df = df.drop(columns=['date'])
```

A correlation check between the date-derived columns and price is useful before deciding whether to keep them:

```python
print(df[['year', 'month', 'day', 'price']].corr()['price'])
```

Because the dataset only covers a narrow window of listing dates, `year`/`month`/`day` add little predictive signal and can distort predictions when a user later queries the model with a date far outside the training range. See **Key Learnings** below.

---

## 📥 Loading the Dataset

```python
import pandas as pd
dataset = pd.read_csv("data.csv")
```

Inspect it with:

```python
dataset.head()
dataset.info()
dataset.describe()
```

---

## 🔢 Independent and Dependent Variables

```python
y = df['price']
X = df.drop(columns=['price'])
```

**Independent Variables (X):** every remaining column — `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_basement`, `yr_built`, `country`, one-hot `city_*` columns.

**Dependent Variable (y):** `price` — the value the model attempts to predict.

---

## ✂️ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Why Split the Dataset?

The training set teaches the model. The testing set is held back so the trained model can be evaluated on data it has never seen — this reveals whether the model generalizes or has simply memorized the training rows.

---

## 🏋️ Model Training

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)
```

During training, the model learns the best coefficient for every feature simultaneously:

```
price = b0 + b1·bedrooms + b2·bathrooms + b3·sqft_living + ... + bn·city_Redmond
```

---

## ⚙️ Why Gradient Descent Was Not Required

Gradient Descent is an iterative optimization technique commonly used to minimize error while training a model:

```
Initialize Parameters
        ↓
Calculate Predictions
        ↓
Calculate Error
        ↓
Calculate Gradient
        ↓
Update Parameters
        ↓
Repeat
```

For ordinary least-squares linear regression, though, Scikit-learn's `LinearRegression` solves for the optimal coefficients directly using a closed-form solution (the normal equation / least-squares decomposition) rather than iterating. For a dataset of this size and dimensionality, this is fast and exact, so manually implementing Gradient Descent was not necessary — `LinearRegression().fit()` already produces the best-fitting coefficients in one step.

---

## 📊 Training Performance

```python
train_predict_data = reg.predict(X_train)

from sklearn.metrics import r2_score, root_mean_squared_error

print(f'Train R² Score : {r2_score(y_train, train_predict_data)}')
print(f'Train RMSE     : {root_mean_squared_error(y_train, train_predict_data)}')
```

> Fill in your actual values after running `MLR.py`:
> **Training R² Score ≈ `<your value>`**
> **Training RMSE ≈ `<your value>`**

---

## 📈 Training Data Visualization

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))
plt.title("Train performance")
plt.scatter(y_train, train_predict_data, color='r', marker='*')
plt.plot([y_train.min(), y_train.max()],
         [y_train.min(), y_train.max()], color='b')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
```

Points sitting close to the diagonal reference line indicate the model is fitting the training data well.

---

## 🧪 Model Testing

```python
test_predict_data = reg.predict(X_test)

print(f'Test R² Score : {r2_score(y_test, test_predict_data)}')
print(f'Test RMSE     : {root_mean_squared_error(y_test, test_predict_data)}')
```

> Fill in your actual values:
> **Testing R² Score ≈ `<your value>`**
> **Testing RMSE ≈ `<your value>`**

---

## 📉 Testing Data Visualization

```python
plt.figure(figsize=(5, 3))
plt.title("Test performance")
plt.scatter(y_test, test_predict_data, color='r', marker='*')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], color='b')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
```

---

## 📊 Model Performance Summary

| 📌 Metric | 🏋️ Training | 🧪 Testing |
|---|---|---|
| Dataset Size | `<rows>` | `<rows>` |
| R² Score | `<value>` | `<value>` |
| RMSE | `<value>` | `<value>` |
| Overall Performance | ✅ / ⚠️ | ✅ / ⚠️ |

> Note: For regression problems, **R² Score**, **MAE**, **MSE**, and **RMSE** are the appropriate evaluation metrics — not classification "accuracy".

---

## 📏 Regression Evaluation Metrics

**R² Score** — how much variance in price is explained by the model. Closer to `1.0` = better fit.

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, test_predict_data)
```

**Mean Absolute Error (MAE)** — average absolute difference between actual and predicted price.

```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, test_predict_data)
```

**Mean Squared Error (MSE)** — average squared difference between actual and predicted price.

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, test_predict_data)
```

**Root Mean Squared Error (RMSE)** — same units as price, easier to interpret than MSE.

```python
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, test_predict_data)
```

A lower MAE / MSE / RMSE and an R² closer to 1.0 indicate better predictive performance.

---

## 🔮 Custom Predictions

Once the model is trained, arbitrary property combinations can be scored directly:

```python
obj = MULTIPLE_L_R("data.csv")
obj.training_data_performance()
obj.test_data_performance()

obj.custom_input_predict_price(
    bedrooms=3,
    bathrooms=2,
    sqft_living=1800,
    sqft_lot=5000,
    floors=1,
    waterfront=0,
    view=0,
    condition=3,
    sqft_basement=0,
    yr_built=1995,
    city_Redmond=True   # exact one-hot column name
)
```

The prediction flow:

```
Property Features
   (bedrooms, bathrooms, sqft_living,
    city, waterfront, ...)
        │
        ▼
🧠 Trained ML Model
        │
        ▼
💰 Predicted Price
```

---

## 🧾 Key Learnings

- **One-hot city columns** must be zero-filled for every city except the one being predicted for — that's the correct behavior for categorical encoding.
- **`country`** was constant (`USA` → `1`) across every training row. A zero-variance feature carries no real predictive signal and can distort a linear model's coefficients if left inconsistent between training and inference.
- **`year`/`month`/`day`**, derived from the listing date, only make sense within the date range the training data actually covers. Feeding the model a date far outside that range (e.g. today's date, months or years later) causes the model to extrapolate, which linear regression handles poorly — predictions can swing by large, unrealistic amounts. Dropping these columns (or supplying a value consistent with the training range) keeps predictions stable.
- Always build the prediction input `DataFrame` using the exact column names and order the model was trained on (`model.feature_names_in_` is a reliable source of truth) rather than assuming an ordered list of values.

---

## ✅ Model Finalization

The model was finalized after observing:

    ✅ Reasonable training performance
    ✅ Reasonable testing performance
    ✅ Small difference between training and testing scores
    ✅ Actual and predicted values remained close
    ✅ Successful custom predictions
    ✅ Stable predictions across realistic input ranges

---

## 💾 Saving the Model Using Pickle

```python
import pickle

def save_model(self, filename='House_Price_Prediction_Model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(self.reg, f)
```

This creates `House_Price_Prediction_Model.pkl`, which the Flask application loads at runtime.

---

## 📥 Loading the Saved Model

```python
def load_model(self, filename='House_Price_Prediction_Model.pkl'):
    with open(filename, 'rb') as f:
        self.m = pickle.load(f)
```

This eliminates the need to retrain the model every time the web application starts.

```
Train Model Once
      │
      ▼
💾 House_Price_Prediction_Model.pkl
      │
      ▼
🌐 Flask Application
      │
      ▼
🔮 Predictions
```

---

## 🐍 Python Virtual Environment

A virtual environment keeps this project's dependencies isolated from other Python projects on your machine.

```
Computer
│
├── Property-Price-Prediction
    └── Virtual Environment (.venv)
        ├── Flask
        ├── Pandas
        ├── NumPy
        ├── Scikit-learn
        └── Other Dependencies
```

### Benefits

- Prevents dependency conflicts
- Keeps project libraries isolated
- Makes dependency management easier
- Allows different projects to use different package versions
- Improves project portability
- Makes cloud deployment more reliable

### Creating a Virtual Environment

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Then install the required libraries:

```bash
pip install flask pandas numpy matplotlib scikit-learn gunicorn
```

---

## 📦 requirements.txt

All project dependencies should be stored inside `requirements.txt`. Generate it with:

```bash
pip freeze > requirements.txt
```

Example:

```
Flask
gunicorn
numpy
pandas
scikit-learn
matplotlib
```

This file is essential for cloud deployment — the hosting platform reads it and installs every required dependency automatically.

---

## 🌐 Flask Web Application

Once the model is trained and saved, it's wired into a web application using Flask:

```
Frontend
   ↕
Backend
   ↕
Machine Learning Model
```

A user enters property details on a web page. Flask receives the input, converts it into the exact feature format the model expects (including one-hot city encoding), passes it to the trained model, and returns the predicted price.

---

## 🎨 Frontend — `templates/index.html`

The frontend is a responsive HTML/CSS form (no framework) covering every feature the model needs: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_basement, yr_built, and city.

```
👤 User
   │
   ▼
Enter Property Details
   │
   ▼
Click "Predict House Price"
   │
   ▼
Flask Backend
   │
   ▼
Machine Learning Model
   │
   ▼
💰 Predicted Price
```

---

## ⚙️ Backend — `app.py`

```python
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

    if 'country' in input_df.columns:
        input_df['country'] = 1

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

    sol = reg.predict(input_df)[0]
    return render_template('index.html', prediction_text=round(sol, 2))


if __name__ == '__main__':
    app.run(debug=True)
```

---

## 🖥️ Running the Application Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```bash
cd Property-Price-Prediction
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

The application should now be running locally.

---

## 🦄 Gunicorn

Gunicorn stands for **Green Unicorn** — a production-ready Python WSGI HTTP server.

Flask's built-in development server is fine for local testing, but a production server like Gunicorn is more appropriate for a publicly deployed application.

Install it:

```bash
pip install gunicorn
```

Run the application with it:

```bash
gunicorn app:app
```

Here, `app:app` means:

```
app.py
  :
Flask application object named app
```

---

## 📄 Procfile

Include a `Procfile` in the project root for platforms like Render or Heroku:

```
web: gunicorn app:app
```

This tells the deployment platform how to start the web application.

---

## 📁 Project Structure

```
Property-Price-Prediction/
│
├── 🐍 MLR.py                              # Model class: data prep, training, evaluation, save/load, predict
├── 🐍 app.py                              # Flask app serving the prediction web form
├── 📊 data.csv                            # Housing dataset used for training
├── 🧠 House_Price_Prediction_Model.pkl    # Trained & serialized model
├── 📦 requirements.txt
├── ⚙️ Procfile
├── 📖 README.md
├── 🚫 .gitignore
│
└── 📁 templates/
    └── 🌐 index.html
```

---

## 🏗️ Application Architecture

```
                    👤 USER
                       │
                       ▼
                 🌐 WEB BROWSER
                       │
                       ▼
                  ☁️ CLOUD HOST
                       │
                       ▼
                 🦄 GUNICORN
                       │
                       ▼
                🌶️ FLASK API
                    app.py
                       │
                       ▼
        💾 House_Price_Prediction_Model.pkl
                       │
                       ▼
          🧠 MULTIPLE LINEAR REGRESSION
                       │
                       ▼
                💰 PREDICTED PRICE
                       │
                       ▼
                  🌐 index.html
                       │
                       ▼
                    👤 USER
```

---

## 🔗 Project Links

- 🌐 **Live Application:** `<add your deployed URL here>`
- 💼 **LinkedIn:** `<add your LinkedIn URL here>`
- 🐙 **GitHub:** `<add your GitHub profile URL here>`

---

## ⭐ Conclusion

The **Property Price Prediction Using Multiple Linear Regression** project demonstrates how a Machine Learning model can be developed from a raw, multi-feature dataset and transformed into a publicly accessible web application.

The project started with property records containing:

```
Bedrooms + Bathrooms + Living Area + Lot Size +
Floors + Waterfront + View + Condition +
Basement Area + Year Built + City
```

The data was analyzed through Exploratory Data Analysis, cleaned and encoded (one-hot city, mapped country, decomposed dates), then split into training and testing sets.

A **Multiple Linear Regression** model was trained on 80% of the data and evaluated on the remaining 20%, achieving:

```
🏋️ Training R² Score : <your value>
🧪 Testing R² Score  : <your value>
```

After evaluation, the model was tested with custom property inputs, then saved with Pickle, wired into a Flask web application, prepared to run under Gunicorn, and version-controlled with Git/GitHub — ready for cloud deployment.

The complete lifecycle:

```
📂 DATA
   ↓
🔍 ANALYSIS
   ↓
🧹 FEATURE ENGINEERING
   ↓
🧠 MACHINE LEARNING
   ↓
📊 MODEL EVALUATION
   ↓
💾 MODEL SERIALIZATION
   ↓
🌐 WEB APPLICATION
   ↓
🐙 VERSION CONTROL
   ↓
☁️ CLOUD DEPLOYMENT
   ↓
🌍 REAL-WORLD ML APPLICATION
```

---

## 👩‍💻 Author

**Aisha**
Python • Machine Learning • Data Science • Flask • Cloud Deployment

⭐ If you found this project useful, please consider giving the repository a star!
Feedback, suggestions, and contributions are welcome.
