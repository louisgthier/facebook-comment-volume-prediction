from urllib import request
from altair import repeat
from flask import Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/predictions', methods=['GET'])
def predict_comments():

    page_likes = float(request.args.get('page_likes'))
    page_checkins = float(request.args.get('page_checkins'))
    page_talking_about = float(request.args.get('page_talking_about'))
    page_category = float(request.args.get('page_category'))
    derived_5 = float(request.args.get('derived_5'))
    derived_6 = float(request.args.get('derived_6'))
    derived_7 = float(request.args.get('derived_7'))
    derived_8 = float(request.args.get('derived_8'))
    derived_9 = float(request.args.get('derived_9'))
    derived_10 = float(request.args.get('derived_10'))
    derived_11 = float(request.args.get('derived_11'))
    derived_12 = float(request.args.get('derived_12'))
    derived_13 = float(request.args.get('derived_13'))
    derived_14 = float(request.args.get('derived_14'))
    derived_15 = float(request.args.get('derived_15'))
    derived_16 = float(request.args.get('derived_16'))
    derived_17 = float(request.args.get('derived_17'))
    derived_18 = float(request.args.get('derived_18'))
    derived_19 = float(request.args.get('derived_19'))
    derived_20 = float(request.args.get('derived_20'))
    derived_21 = float(request.args.get('derived_21'))
    derived_22 = float(request.args.get('derived_22'))
    derived_23 = float(request.args.get('derived_23'))
    derived_24 = float(request.args.get('derived_24'))
    derived_25 = float(request.args.get('derived_25'))
    derived_26 = float(request.args.get('derived_26'))
    derived_27 = float(request.args.get('derived_27'))
    derived_28 = float(request.args.get('derived_28'))
    derived_29 = float(request.args.get('derived_29'))
    cc1 = float(request.args.get('cc1'))
    cc2 = float(request.args.get('cc2'))
    cc3 = float(request.args.get('cc3'))
    cc4 = float(request.args.get('cc4'))
    cc5 = float(request.args.get('cc5'))
    base_time = float(request.args.get('base_time'))
    post_length = float(request.args.get('post_length'))
    post_share_count = float(request.args.get('post_share_count'))
    post_promotion_status = float(request.args.get('post_promotion_status'))
    h_local = float(request.args.get('h_local'))

    data = {
        'Page Popularity/likes': [page_likes],
        'Page Checkins': [page_checkins],
        'Page talking about': [page_talking_about],
        'Page Category': [page_category],
        'Derived_5': [derived_5],
        'Derived_6': [derived_6],
        'Derived_7': [derived_7],
        'Derived_8': [derived_8],
        'Derived_9': [derived_9],
        'Derived_10': [derived_10],
        'Derived_11': [derived_11],
        'Derived_12': [derived_12],
        'Derived_13': [derived_13],
        'Derived_14': [derived_14],
        'Derived_15': [derived_15],
        'Derived_16': [derived_16],
        'Derived_17': [derived_17],
        'Derived_18': [derived_18],
        'Derived_19': [derived_19],
        'Derived_20': [derived_20],
        'Derived_21': [derived_21],
        'Derived_22': [derived_22],
        'Derived_23': [derived_23],
        'Derived_24': [derived_24],
        'Derived_25': [derived_25],
        'Derived_26': [derived_26],
        'Derived_27': [derived_27],
        'Derived_28': [derived_28],
        'Derived_29': [derived_29],
        'CC1': [cc1],
        'CC2': [cc2],
        'CC3': [cc3],
        'CC4': [cc4],
        'CC5': [cc5],
        'Base time': [base_time],
        'Post Length': [post_length],
        'Post Share Count': [post_share_count],
        'Post Promotion Status': [post_promotion_status],
        'H Local': [h_local]
    }
    df = pd.DataFrame(data)

    model = LinearRegression()
    model.load_model("chemin_vers_le_modele_entrene.pkl")  # Remplacez par le chemin vers votre modèle pré-entraîné

    prediction = model.predict(df)[0]

    response = {'prediction': prediction}
    return jsonify(response)


if __name__ == '__main__':
    data = pd.read_csv("Dataset/Training/Features_Variant_5.csv") 
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_predicted))
    print(rms)

    app.run(debug=True)
    



