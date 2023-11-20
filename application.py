from flask import Flask, request, render_template
from preprocessor import predict_emotion
import pickle
import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from sklearn import preprocessing
import re

voc_size=10000 # Vocabulary size
embed_size=100 #word vector size
ps = PorterStemmer()
sent_length = 1

with open('test_model.pkl', 'rb') as file:
    model = pickle.load(file)

application = Flask(__name__, template_folder='Templates')
app = application

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict_emotion():
    ps =PorterStemmer()
    review = request.form.get('user_text')
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review,voc_size)] 
    embed = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    predicti = model.predict(embed)
    return label_encoder.classes_[np.argmax(predicti)]

if __name__ == '__main__':
    app.run()
