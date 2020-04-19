# Importing the libraries
import os
import numpy as np  
import re

# 1. Acquiring, preprocessing, and analyzing the data
# Importing training data
neg_Path = r'E:\ML-miniprojects\P2\aclImdb\train\neg'
os.chdir(r'E:\ML-miniprojects\P2\aclImdb\train\neg')
files_train = os.listdir(neg_Path)
reviews_train=[]
rate_train=[]
for i in range(len(files_train)):
    f=files_train[i]
    my_file_handle=open(f, encoding="utf8")
    x=my_file_handle.read()
    reviews_train.append(x)
    y=files_train[i]
    rate_train.append('0')
      
pos_Path = r'E:\ML-miniprojects\P2\aclImdb\train\pos'
os.chdir(r'E:\ML-miniprojects\P2\aclImdb\train\pos')
files = os.listdir(pos_Path)
for i in range(len(files)):
    f=files[i]
    my_file_handle=open(f, encoding="utf8")
    x=my_file_handle.read()
    reviews_train.append(x)
    y=files[i]
    rate_train.append('1')
## labels = {'pos': 1, 'neg': 0}

# Cleaning train data
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)

# Tokenizing text
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(reviews_train_clean)
X_train_counts.shape

# Obtaining frequencies from occurancies. First we obtain term frequencies (TF)...
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf = False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

#.. and then term frequencies times inverse document frequency (TF-IDF):
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# 2. Training and testing the classifiers
# Before we dive into training and testing, we need to define how do we do cross-validation. Cross-validation is essential for validation of many models and this case is not an exception. It was decided to consider  k=5  folds, just like in Mini-Project 1.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def cross_validation(model, X, y, folds):
    pipeline_tfidf = Pipeline([
        ('tfidf',
         TfidfVectorizer(sublinear_tf = True,
                         smooth_idf = True,
                         norm = "l2",
                         lowercase = True,
                         max_features = 50000,
                         use_idf = True,
                         encoding = "utf-8",
                         decode_error = 'ignore',
                         strip_accents = 'unicode',
                         analyzer = "word")),
          ('clf', model)],
          verbose = True)

    scores = cross_val_score(pipeline_tfidf,
                             X,
                             y,
                             cv = folds,
                             scoring = "accuracy")

    print("Cross-validation scores:", scores)
    print("Cross-validation mean score:", scores.mean())

    return scores, scores.mean()

#Now we can actually start training the classifiers.
    
# 2.1. Fitting Logistic regression to the Training set
from sklearn.linear_model import LogisticRegression
# 2.1.1. Training the logistic regression classifier
clf = LogisticRegression().fit(X_train_tfidf, rate_train)

# The model is tested on a couple of custom target values to check whether the trained model can correctly predict them:
docs_new = ['Korea, Russia, Iran']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, rate_train))

# 2.1.2. Hyperparameter tuning for logistic regression
C_par = [0.01, 0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

mu_max = 0
C_max = C_par[0]

for c in C_par:
    scores, mu = cross_validation(model = LogisticRegression(C = c, max_iter = 500), X = reviews_train_clean, y = rate_train, folds = 5)
    
    if (mu > mu_max):
        mu_max = mu
        C_max = c

    print("-----------------------------------------------------------------")

print("Best hyperparameter C:", C_max)
print("Best mean accuracy:", mu_max)   

# Tuning the loss function:
penalties = ['l1', 'l2', 'elasticnet']

mu_max = 0
penalty_best = penalties[0]

for p in penalties:
  scores, mu = cross_validation(model = LogisticRegression(C = 2, max_iter = 500, penalty = p), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    penalty_best = p

  print("-----------------------------------------------------------------")

print("Best loss function:", penalty_best)
print("Best accuracy:", mu_max)

# 2.1.3 Building a pipeline for logistic regression
# A pipeline is built to make the model to be easier to work with:
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(C = 2, max_iter = 500, penalty = 'l2')),
])
    
# The model can be trained much simplier now:
text_clf.fit(reviews_train_clean, rate_train)

# 2.1.4. Evaluation of the performance of logistic regression on the test set 
# Importing test data
neg_Path_test = r'E:\ML-miniprojects\P2\aclImdb\test\neg'
os.chdir(r'E:\ML-miniprojects\P2\aclImdb\test\neg')
files_test = os.listdir(neg_Path_test)
reviews_test=[]
rate_test=[]
for j in range(len(files_test)):
    f_test=files_test[j]
    my_file_handle_test=open(f_test, encoding="utf8")
    x_test=my_file_handle_test.read()
    reviews_test.append(x_test)
    y_test=files_test[j]
    rate_test.append('0')
       
pos_Path_test = r'E:\ML-miniprojects\P2\aclImdb\test\pos'
os.chdir(r'E:\ML-miniprojects\P2\aclImdb\test\pos')
files_test = os.listdir(pos_Path_test)
for j in range(len(files_test)):
    f_test=files_test[j]
    my_file_handle_test=open(f_test, encoding="utf8")
    x_test=my_file_handle_test.read()
    reviews_test.append(x_test)
    y_test=files_test[j]
    rate_test.append('1')

# Cleaning test dataset
reviews_test_clean = preprocess_reviews(reviews_test)

# Predicting the Test set results
predicted = text_clf.predict(reviews_test_clean)
print("Average accuracy:", np.mean(predicted == rate_test)) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rate_test, predicted)

from sklearn import metrics
print(metrics.classification_report(rate_test, predicted))

# 2.2 Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
# 2.2.1. Training the decision tree classifier
clf = DecisionTreeClassifier().fit(X_train_tfidf, rate_train)

# The model is tested on a couple of custom target values to check whether the trained model can correctly predict them:
docs_new = ['Korea, Russia, Iran']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, rate_train))

# 2.2.2. Hyperparameter tuning for decision trees
# First we tune min_samples_split
min_samples_splits = np.linspace(0.01, 0.1, 10, endpoint = True)

mu_max = 0
min_samples_split_max = min_samples_splits[0]

for min_samples_split in min_samples_splits:
  scores, mu = cross_validation(model = DecisionTreeClassifier(min_samples_split = min_samples_split), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    min_samples_split_max = min_samples_split

  print("-----------------------------------------------------------------")

print("Best hyperparameter min_samples_splits:", min_samples_split_max)
print("Best accuracy:", mu_max)    

# Tuning min_samples_leaf:
min_samples_leafs = np.linspace(0.001, 0.01, 10, endpoint=True)

mu_max = 0
min_samples_leaf_max = min_samples_leafs[0]

for min_samples_leaf in min_samples_leafs:
  scores, mu = cross_validation(model = DecisionTreeClassifier(min_samples_leaf = min_samples_leaf, min_samples_split = 0.06000000000000001), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    min_samples_leaf_max = min_samples_leaf

  print("-----------------------------------------------------------------")

print("Best hyperparameter min_samples_leaf:", min_samples_leaf_max)
print("Best accuracy:", mu_max)

# 2.2.3. Building a pipeline for decision tree
# A pipeline is built to make the model to be easier to work with:
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier(min_samples_leaf = 0.001, min_samples_split = 0.06000000000000001)),
])
    
# The model can be trained much simplier now:
text_clf.fit(reviews_train_clean, rate_train)

# 2.2.4. Evaluation of the performance of decision tree on the test set
# Predicting the Test set results
predicted = text_clf.predict(reviews_test_clean)
print("Average accuracy:", np.mean(predicted == rate_test)) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rate_test, predicted)

from sklearn import metrics
print(metrics.classification_report(rate_test, predicted))

# 2.3. Fitting SVM to the Training set
from sklearn.svm import LinearSVC
# 2.3.1. Training the SVM classifier
clf = LinearSVC().fit(X_train_tfidf, rate_train)

# 2.3.2. Hyperparameter tuning for SVM
# Tuning the penalty parameter  C :
C_par = [0.25, 0.5, 0.75, 1, 1.25, 1.5]

mu_max = 0
C_max = C_par[0]

for c in C_par:
  scores, mu = cross_validation(model = LinearSVC(C = c), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    C_max = c

  print("-----------------------------------------------------------------")

print("Best hyperparameter C:", C_max)
print("Best accuracy:", mu_max)

#Tuning the loss function:
penalties = ['l1', 'l2']

mu_max = 0
penalty_best = penalties[0]

for p in penalties:
    scores, mu = cross_validation(model = LinearSVC(C = 0.25, penalty = p), X = reviews_train_clean, y = rate_train, folds = 5)
    
    if (mu > mu_max):
        mu_max = mu
        penalty_best = p
        
    print("-----------------------------------------------------------------")

print("Best loss function:", penalty_best)
print("Best accuracy:", mu_max)

# 2.3.3. Building a pipeline for SVM
# A pipeline is built to make the model to be easier to work with:

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(C = 0.25, penalty = 'l2')),
])
    
# The model can be trained much simplier now:
text_clf.fit(reviews_train_clean, rate_train)

# 2.3.4. Evaluation of the performance of SVM on the test set
# Predicting the Test set results
predicted = text_clf.predict(reviews_test_clean)
print("Average accuracy:", np.mean(predicted == rate_test))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rate_test, predicted)

from sklearn import metrics
print(metrics.classification_report(rate_test, predicted))   

# 2.4 Fitting Ada boost classifier to the Training set
from sklearn.ensemble import AdaBoostClassifier
# 2.4.1. Training Ada boost classifier
clf = AdaBoostClassifier().fit(X_train_tfidf, rate_train) 

# 2.4.2. Hyperparameter tuning for Ada boost
# First we tune the maximum number of estmators:
ns = [50, 100, 150, 200]

mu_max = 0
n_best = ns[0]

for n in ns:
  scores, mu = cross_validation(model = AdaBoostClassifier(n_estimators = n), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    n_best = n

  print("-----------------------------------------------------------------")

print("Best hyperparameter n_estimators:", n_best)
print("Best accuracy:", mu_max)

# Tuning the learning rate:
lrs = [0.01, 0.1, 0.5, 1, 1.5, 10]

mu_max = 0
lr_best = ns[0]

for lr in lrs:
  scores, mu = cross_validation(model = AdaBoostClassifier(n_estimators = 
                                                           200, learning_rate = lr), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    lr_best = lr

  print("-----------------------------------------------------------------")

print("Best learning rate:", lr_best)
print("Best accuracy:", mu_max)

# 2.4.3. Building a pipeline for Ada boost
# A pipeline is built to make the model to be easier to work with:
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', AdaBoostClassifier(n_estimators = 200, learning_rate = 0.5)),
])

# The model can be trained much simplier now:
text_clf.fit(reviews_train_clean,  rate_train)

# 2.4.4. Evaluation of the performance of Ada boost on the test set
# Predicting the Test set results
predicted = text_clf.predict(reviews_test_clean)
print("Average accuracy:", np.mean(predicted == rate_test)) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rate_test, predicted)

from sklearn import metrics
print(metrics.classification_report(rate_test, predicted))     

# 2.5 Fitting Random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
# 2.5.1. Training Random forest classifier
clf = RandomForestClassifier().fit(X_train_tfidf, rate_train)    

# 2.5.2. Hyperparameter tuning for random forest
# First we tune the maximum number of estmators:
ns = [50, 100, 150, 200]

mu_max = 0
n_best = ns[0]

for n in ns:
  scores, mu = cross_validation(model = RandomForestClassifier(n_estimators = n), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    n_best = n

  print("-----------------------------------------------------------------")

print("Best hyperparameter n_estimators:", n_best)
print("Best accuracy:", mu_max)

# Increasing the number of estimators affects the time required for training while accuracy does not increase much after 150, so it was decided to stop at 150.
# Next we tune min_samples_split:
min_samples_splits = np.linspace(0.001, 0.01, 10, endpoint=True)

mu_max = 0
min_samples_split_max = min_samples_splits[0]

for min_samples_split in min_samples_splits:
  scores, mu = cross_validation(model = RandomForestClassifier(n_estimators = 200, min_samples_split = min_samples_split), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    min_samples_split_max = min_samples_split

  print("-----------------------------------------------------------------")

print("Best hyperparameter min_samples_splits:", min_samples_split_max)
print("Best accuracy:", mu_max)
# Tuning min_samples_leaf:
min_samples_leafs = np.linspace(0.001, 0.01, 10, endpoint=True)

mu_max = 0
min_samples_leaf_max = min_samples_leafs[0]

for min_samples_leaf in min_samples_leafs:
  scores, mu = cross_validation(model = RandomForestClassifier(n_estimators = 200, min_samples_split = 0.003, min_samples_leaf = min_samples_leaf), X = reviews_train_clean, y = rate_train, folds = 5)

  if (mu > mu_max):
    mu_max = mu
    min_samples_leaf_max = min_samples_leaf

  print("-----------------------------------------------------------------")

print("Best hyperparameter min_samples_leaf:", min_samples_leaf_max)
print("Best accuracy:", mu_max)

# 2.5.3. Building a pipeline for random forest
# A pipeline is built to make the model to be easier to work with:
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier(n_estimators = 200, min_samples_split = 0.003, min_samples_leaf = 0.001)),
])
    
# The model can be trained much simplier now:
text_clf.fit(reviews_train_clean,  rate_train)

# 2.5.4. Evaluation of the performance of Ada boost on the test set
# Predicting the Test set results
predicted = text_clf.predict(reviews_test_clean)
print("Average accuracy:", np.mean(predicted == rate_test)) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rate_test, predicted)

from sklearn import metrics
print(metrics.classification_report(rate_test, predicted))     