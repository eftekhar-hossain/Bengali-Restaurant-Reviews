from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import SGDClassifier
import re

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	stopword_list = open('rr_stopwords.pkl', 'rb')
	stp = pickle.load(stopword_list)

	def process_reviews(review):
		review = review.replace('\n', '')  # removing new line
		# removing unnecessary punctuation
		review = re.sub('[^\u0980-\u09FF]', ' ', str(review))
		result = review.split()
		review = [word.strip() for word in result if word not in stp]
		review = " ".join(review)
		return review

	# load the pickle file of the cleaned data
	cleaned_data = open('rr_review_data.pkl', 'rb')
	data = pickle.load(cleaned_data)

	# Extract TF-IDF for Trigram feature
	tfidf = TfidfVectorizer(ngram_range=(1, 3), use_idf=True, tokenizer=lambda x: x.split())
	X = tfidf.fit_transform(data.cleaned)

	# load the Stochastic Gradient Descent model
	model = open('rr_review_sgd.pkl', 'rb')
	sgd = pickle.load(model)
	# sentiment = nb.predict(feature)

	if request.method == 'POST':
		comment = request.form['comment']
		review = process_reviews(comment)
		vect = tfidf.transform([review]).toarray()
		my_prediction = sgd.predict(vect)
		prediction_score = sgd.predict_proba(vect)
		score = round(max(prediction_score.reshape(-1)), 2) * 100

	return render_template('predict.html', value=comment, sentiment=my_prediction, prob=score)



if __name__ == '__main__':
	app.run(debug=True)