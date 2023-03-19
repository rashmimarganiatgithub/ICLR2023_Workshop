import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import fasttext
import fasttext
import texthero
from nltk.corpus import stopwords
import lightgbm as lgbm
import pandas as pd
from tqdm.auto import tqdm


app = Flask(__name__)


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():


    model = pickle.load(open('model.pkl','rb'))
    fttm=fasttext.load_model('transformer.pkl')
    inputQuery1 = request.form['querry1']
    df          = pd.DataFrame({'text': [inputQuery1]})
    all_texts   = df['text'].str.lower()
    stopwords_list = stopwords.words('english') + stopwords.words('french')
    all_texts   = texthero.remove_stopwords(all_texts, stopwords_list)
    all_texts   = texthero.remove_whitespace(all_texts)
    
    all_features = [fttm.get_sentence_vector(text) for text in tqdm(all_texts)]
    all_features = np.vstack(all_features)
    
    prediction = model.predict(all_features)
    print(prediction)

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Arabic Text Label {}".format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)
