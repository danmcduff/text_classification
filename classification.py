#################################################
# Comparison of text classification techniques using
# python libraries.  
#
# Depdendencies are: 
#    spacy - NLP toolbox.
#    sklearn - ML toolbox.
#    pandas - Data analysis library.
#    numpy - Math library.
# 
# Daniel McDuff
#################################################

# SPACY:
from spacy.en import English
parser = English()
# SKLEARN:
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from nltk.corpus import stopwords
import string
import re
# PANDAS:
import pandas
# NUMPY:
import numpy
from statistics import mode

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "'ve"]

# Every step in a pipeline needs to be a "transformer". 
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
    
# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    
    # replace twitter @mentions
    mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    text = mentionFinder.sub("@MENTION", text)
    
    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<")
    
    # lowercase
    text = text.lower()

    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens

def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)


# Initialize the text vectorizor (this code uses a bag of words set of features):
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))

# Load the data (pulled directly using the query that Kareem sent):
df = pandas.read_csv('../Data/get_appropriate_posts.csv', encoding = "ISO-8859-1")

# Identify the header of the labels in the data file:
label_col = 'app_label'

# Identify which class is the +ve class (for the purproses of calcualting precision and recall):
pos_lab = 'notappropriate'

# Append the situation and thoughts columns to form a single string:
df['train'] = df['situation'] + " " + df['thoughts']

# Convert the labels into strings (if necessary):
df['label'] = df[label_col].astype(str)

# Define a list of classifiers to test:
names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Decision Tree",
         "Random Forest", 
         "AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier()]
    
    
# Loop through classifiers and do a K-Fold validation using a 10% (random) hold-out for testing.
for name, clf in zip(names, classifiers):

    print("----------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------")
    print name

    # Initialize a dataframe to store the results:
    df_results = pandas.DataFrame(columns=('accuracy','precision','recall'))

    # Iterate for 9 folds of train and test:
    for i in range(0,9):
        
        # Hold out a random 10% of the data for testing and use the rest for training:
        df_train, df_test = train_test_split(df, test_size = 0.1)

        train = df_train['train'].tolist()
        labelsTrain = df_train['label'].tolist()
        
        test = df_test['train'].tolist()
        labelsTest = df_test['label'].tolist()
        
        ## VALIDATION:
        # Where necessary perform a set of validation on the training set to identify parameters
        # For now this is just a coarse and not rigourous search. 
        # I can easily do a much more thorough validation of parameters for a selected model:
        if(name=='RBF SVM'):
            scores=numpy.zeros((3,3))
            gs = [0.001, 0.01, 0.1]
            cs = [1, 10, 100]
            
            for i in range(0,len(gs)):
                for j in range(0,len(cs)):
                    g=gs[i]
                    c=cs[j]
                    clf = SVC(gamma=g, C=c)
                    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
                    pipe.fit(train, labelsTrain)
                    preds = pipe.predict(test)
                    pre_score = precision_score(labelsTest, preds, pos_label=pos_lab)
                    rec_score = recall_score(labelsTest, preds, pos_label=pos_lab)
                    scores[i,j] = (pre_score+rec_score)/2
            
            print scores
            
            j=numpy.argmax(numpy.max(scores, axis=0))
            i=numpy.argmax(numpy.max(scores, axis=1))
            g=gs[i]
            c=cs[j]
            print g
            print c
            clf = SVC(gamma=g, C=c)
        elif(name=='Random Forest'):
            scores=numpy.zeros((4,1))
            ns = [10,100,500,1000]
            
            for i in range(0,len(ns)):
                n=ns[i]
                clf = RandomForestClassifier(n_estimators=n)
                pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
                pipe.fit(train, labelsTrain)
                preds = pipe.predict(test)
                pre_score = precision_score(labelsTest, preds, pos_label=pos_lab)
                rec_score = recall_score(labelsTest, preds, pos_label=pos_lab)
                scores[i] = (pre_score+rec_score)/2
            
            print scores
            
            i=numpy.argmax(scores, axis=0)
            n=ns[i]
            print n
            clf = RandomForestClassifier(n_estimators=n)
        elif(name=='Linear SVM'):
            scores=numpy.zeros((5,1))
            cs = [0.01,0.1,1,10,100]
            
            for i in range(0,len(cs)):
                c=cs[i]
                clf = SVC(kernel="linear", C=c)
                pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
                pipe.fit(train, labelsTrain)
                preds = pipe.predict(test)
                pre_score = precision_score(labelsTest, preds, pos_label=pos_lab)
                rec_score = recall_score(labelsTest, preds, pos_label=pos_lab)
                scores[i] = (pre_score+rec_score)/2
            
            print scores
            
            i=numpy.argmax(scores, axis=0)
            c=cs[i]
            print c
            clf = SVC(kernel="linear", C=c)  
        
            
        ## Pipeline to clean, tokenize, vectorize, and classify:
        pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
        pipe.fit(train, labelsTrain)
        preds = pipe.predict(test)
        

        ## Evaluate the trained model on the testing set:
        print("----------------------------------------------------------------------------------------------")
        print("results:")
        
        # Accuracy:
        acc_score = accuracy_score(labelsTest, preds)
        print("accuracy:", acc_score)
        # Precision:
        pre_score = precision_score(labelsTest, preds, pos_label=pos_lab)
        print("precision:", pre_score)
        # Recall:
        rec_score = recall_score(labelsTest, preds, pos_label=pos_lab)
        print("recall:", rec_score)

        df_results = df_results.set_value(len(df_results), 'accuracy', acc_score)
        df_results = df_results.set_value(len(df_results)-1, 'precision', pre_score)
        df_results = df_results.set_value(len(df_results)-1, 'recall', rec_score)
        
    # Store the results in a file:
    df_results.to_csv(name+'.csv',index=False)
        
        
# Test an ensemble of classifiers to see whether this is better than a single classifier:    
print("----------------------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------------------")
print "Ensemble classifiers:"

df_results = pandas.DataFrame(columns=('accuracy','precision','recall'))
        
for i in range(0,9):
    
    df_train, df_test = train_test_split(df, test_size = 0.1)

    # TRAIN DATA:
    train = df_train['train'].tolist()
    labelsTrain = df_train['label'].tolist()
        
    # TEST DATA:
    test = df_test['train'].tolist()
    labelsTest = df_test['label'].tolist()
    
    clf = KNeighborsClassifier(3)
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds7 = pipe.predict(test)

    # the pipeline to clean, tokenize, vectorize, and classify
    clf = MultinomialNB()
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds1 = pipe.predict(test)

    clf = SVC(kernel="linear", C=0.1)
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds2 = pipe.predict(test)

    clf = SVC(gamma=0.01, C=100)
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds3 = pipe.predict(test)

    clf = DecisionTreeClassifier() #max_depth=5
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds8 = pipe.predict(test)

    clf = RandomForestClassifier(n_estimators=100) #max_depth=5, , max_features=1
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds6 = pipe.predict(test)

    clf = AdaBoostClassifier()
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])
    pipe.fit(train, labelsTrain)
    preds5 = pipe.predict(test)



    print("----------------------------------------------------------------------------------------------")
    print("results:")
    final_preds = []
    for (sample, pred1, pred2, pred3, pred5, pred6, pred7, pred8) in zip(test, preds1, preds2, preds3, preds5, preds6, preds7, preds8):
        #print(sample, ":", pred, ":", pred2)
        x = [pred1, pred2, pred3, pred5, pred6, pred7, pred8]
        y = mode(x)
        final_preds.append(y)

    pos_lab = 'notappropriate'
    
    acc_score = accuracy_score(labelsTest, final_preds)
    print("accuracy:", acc_score)

    pre_score = precision_score(labelsTest, final_preds, pos_label=pos_lab)
    print("precision:", pre_score)
    rec_score = recall_score(labelsTest, final_preds, pos_label=pos_lab)
    print("recall:", rec_score)

    df_results = df_results.set_value(len(df_results), 'accuracy', acc_score)
    df_results = df_results.set_value(len(df_results)-1, 'precision', pre_score)
    df_results = df_results.set_value(len(df_results)-1, 'recall', rec_score)
    
df_results.to_csv("Ensemble.csv",index=False)
        