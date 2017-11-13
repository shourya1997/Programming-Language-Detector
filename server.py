from flask import Flask , render_template , request , redirect , url_for
app = Flask(__name__)

_code = ""
@app.route("/", methods=["GET","POST"])
def main():
	if(request.method == "POST"):
		_code = request.form['code']
		return redirect(url_for('getcode' , code = _code))
		# return render_template('index.html')
	else:
		return render_template('index.html' , var = "")

@app.route("/code")
def getcode():
	_code = request.args['code']
	

	css_code = open("code/css.txt",encoding="utf8").read()
	html_code = open("code/html.txt",encoding="utf8").read()
	java_code = open("code/java.txt",encoding="utf8").read()
	python_code = open("code/python.txt",encoding="utf8").read()
	js_code = open("code/js.txt",encoding="utf8").read()

	docs = []
	labels = []

	for r in css_code.split('\n'):
	    docs.append(r)
	    labels.append('css')

	for r in html_code.split('\n'):
	    docs.append(r)
	    labels.append('html')

	for r in java_code.split('\n'):
	    docs.append(r)
	    labels.append('java')

	for r in python_code.split('\n'):
	    docs.append(r)
	    labels.append('python')

	# Splitting data into training and test sets
	from sklearn.model_selection import train_test_split

	docs_train, docs_test, labels_train, labels_test = train_test_split(docs,
	                                                                    labels,
	                                                                    test_size = 0.25,
	                                                                    random_state = 42)


	# Tokenizing text with scikit-learn
	from sklearn.feature_extraction.text import CountVectorizer

	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(docs_train)
	print("Shape CountVectorizer on date: ",X_train_counts.shape)

	from sklearn.feature_extraction.text import TfidfTransformer

	tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
	X_train_tfidf = tf_transformer.transform(X_train_counts)
	print("Shape of TfidfTransformer on count_vect: ",X_train_tfidf.shape)

	from sklearn.svm import SVC
	from sklearn.model_selection import GridSearchCV

	# parameters = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 0.00001], 'class_weight':['balanced', None], 'kernel':['linear', 'rbf']}
	# classifier_lsvc =  SVC()
	# clf_LSVC = GridSearchCV(classifier_lsvc, parameters).fit(X_train_tfidf, labels_train)
	# print("Best parameters set found on development set:", clf_LSVC.best_estimator_)

	# Saving classifier in pickle
	import pickle
	# Saving
	# save_classifier = open("svm.pickle","wb")
	# pickle.dump(clf_LSVC, save_classifier)
	# save_classifier.close()
	# # loading
	classifier_f = open("svm.pickle", "rb")
	clf_LSVC = pickle.load(classifier_f)
	classifier_f.close()

	# Data transformation
	X_new_counts = count_vect.transform(docs_test)
	X_new_tfidf = tf_transformer.transform(X_new_counts)

	# testing1_code = ['<h1>']
	# testing2_code = ['import java.util.']
	testing3_code = _code.split('\n')
	# x_java_testing_counts = count_vect.transform(testing1_code)
	# x_java_testing_tfidf = tf_transformer.transform(x_java_testing_counts)

	# x_html_testing_counts = count_vect.transform(testing2_code)
	# x_html_testing_tfidf = tf_transformer.transform(x_html_testing_counts)

	x_code_testing_counts = count_vect.transform(testing3_code)
	x_code_testing_tfidf = tf_transformer.transform(x_code_testing_counts)


	# Testing and Accuracy
	from sklearn.metrics import accuracy_score

	# pred = clf_LSVC.predict(X_new_tfidf)
	# acc = accuracy_score(pred, labels_test)
	# print("Accuracy LinearSVC: ", acc*100)

	# pred_html = clf_LSVC.predict(x_java_testing_tfidf)
	pred_java = clf_LSVC.predict(x_code_testing_tfidf)
	# _code = pred_java[0]
	# print("True pred shud be html, but:",pred_html)
	# print("True pred shud be java, but:",pred_java)
	return redirect(url_for('showinfo' , language = pred_java[0].upper() ))



@app.route("/output")
def showinfo():
	language = request.args['language']
	return render_template('index.html' , var= language)		



	# return render_template('index.html' , var= pred_java[0].upper())
		
if __name__ == "__main__":	
	app.run(host='127.0.0.1', debug=True, port=5000)
