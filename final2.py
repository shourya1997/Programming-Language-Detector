css_code = open("code/css.txt","r").read()
java_code = open("code/java.txt","r").read()
python_code = open("code/python.txt","r").read()
js_code = open("code/js.txt","r").read()
c_code = open("code/c.txt","r").read()
cpp_code = open("code/cpp.txt").read()

docs = []
labels = []

for r in css_code.split('\n'):
    docs.append(r)
    labels.append('css/html')

for r in java_code.split('\n'):
    docs.append(r)
    labels.append('java')

for r in python_code.split('\n'):
    docs.append(r)
    labels.append('python')

for r in js_code.split('\n'):
    docs.append(r)
    labels.append('javascript')

for r in c_code.split('\n'):
    docs.append(r)
    labels.append('c')

for r in cpp_code.split('\n'):
    docs.append(r)
    labels.append('c++')

# Tokenizing text with scikit-learn
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs)
print("Shape CountVectorizer on date: ",X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts)
print("Shape of TfidfTransformer on count_vect: ",X_train_tfidf.shape)

# Splitting data into training and test sets
from sklearn.model_selection import train_test_split

docs_train, docs_test, labels_train, labels_test = train_test_split(X_train_tfidf,
                                                                    labels,
                                                                    test_size = 0.15,
                                                                    random_state = 42)

from sklearn.svm import LinearSVC
# clf_SVC = SVC(kernel="rbf", decision_function_shape='ovo').fit(X_train_tfidf, labels_train)

clf_LSVC = LinearSVC().fit(docs_train, labels_train)

# Saving classifier in pickle
import pickle
# # Saving
# save_classifier = open("linearsvc.pickle","wb")
# pickle.dump(clf_LSVC, save_classifier)
# save_classifier.close()
# loading
classifier_f = open("linearsvc.pickle", "rb")
clf_LSVC = pickle.load(classifier_f)
classifier_f.close()

# Testing and Accuracy
from sklearn.metrics import accuracy_score

pred = clf_LSVC.predict(docs_test)
acc = accuracy_score(pred, labels_test)
print("Accuracy LinearSVC: ", acc*100)


# testing1_code = ['<h1>']
# testing2_code = ['import java.util.']
#
# x_java_testing_counts = count_vect.transform(testing1_code)
# x_java_testing_tfidf = tf_transformer.transform(x_java_testing_counts)
#
# x_html_testing_counts = count_vect.transform(testing2_code)
# x_html_testing_tfidf = tf_transformer.transform(x_html_testing_counts)

# pred_html = clf_SVC.predict(x_java_testing_tfidf)
# pred_java = clf_SVC.predict(x_html_testing_tfidf)
# print("True pred shud be html, but:",pred_html)
# print("True pred shud be java, but:",pred_java)
