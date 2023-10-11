# Import necessary libraries
import sys
import os
import json
import csv
from collections import defaultdict, Counter
import spacy
import nltk
from nltk.corpus import words, stopwords
import pandas as pd
import re
import matplotlib.pyplot as plt
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize

#Json read and write functions
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

nlp = spacy.load("en_core_web_lg")
nltk.download("stopwords") 
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')
stopwords_fr = nltk.corpus.stopwords.words('french')
stopwords_it = nltk.corpus.stopwords.words('italian')
stopwords = stopwords_en + stopwords_sp + stopwords_fr + stopwords_it

nltk.download('words')
hyphenated_words = set(word for word in words.words() if '-' in word)
all_words = nltk.corpus.words.words('en')
#all_words = set(word for word in words.words())

#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder
Model_Folder = config.texts

database_data = load_data(database_file)

#Load Model Parameters
Body = config.Body
Model = config.Model
accepted_types = config.accepted_types
Model_Subfolder = f'/{Body} Texts/{Model}'
Model_Folder = Model_Folder + Model_Subfolder


#Fed Speeches 
if Body == "Fed":
    files = pd.read_csv(f"/Users/kylenabors/Documents/Database/Training Data/Fed/Speeches/fed_speeches_2006_2023.csv", encoding='UTF-8')

#ECB Speeches 
if Body == "ECB":
    files = pd.read_csv('/Users/kylenabors/Documents/Database/Training Data/ECB/Speeches/all_ECB_speeches.csv', sep='|')

files = files.dropna()
print(files.columns)

#files = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv")  
#files = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv", sep=";", encoding='cp1252')  
#files = pd.read_csv('/Users/kylenabors/Documents/Database/Training Data/Fed/Press Confrences/press_conference_transcript.csv')
#files = pd.DataFrame(files, columns=['document_kind', 'meeting_date', 'release_date', 'text', 'url'])

# Specify the year and month you want to start and end processing files from
start_year_month_day = '2006-12-31'
end_year_month_day = '2023-12-31'

if Body == "Fed":
    files['date'] = files['date'].astype(int)
    files['date'] = pd.to_datetime(files['date'], format='%Y%m%d')

files = files[files['date'] >= start_year_month_day]
files = files[files['date'] <= end_year_month_day]

files['date'] = files['date'].astype(str)

final = []
remove_words = ["report", "appendix", "table", 'contents', 'overview', 'reserve', 'congress', 'board', 'federal', 'senate', 
                'seventh', 'document', 'beige', 'book', 'commentary', 'summary', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelveth',
                "chase", "citi", 'fi', 'par', 'committee', 'chairman', 
                'janurary', 'feburary', 'march', 'april', 'may', 'june', 
                'july', 'august', 'september', 'october', 'november', 'december',
                'year', 'month',
                'noted', 'since', 'said', 'also', 'one', 'first', 'six', 'third', 
                'united', 'city', 'district', 'york', 'new york', 'boston', 'san', 'francisco', 'atlanta', 'chicago', 'virgin', 'northern', 'montana',
                'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

csv_out = []

pattern = re.compile(r'[^a-zA-z.,!?/:;\"\'\w\s]')

# Global Counter object to keep track of word frequencies
global_word_counter = Counter()

keyword_freq_ts = defaultdict(lambda: defaultdict(Counter))
keyword_freq_ts2 = defaultdict(lambda: defaultdict(Counter))

for index, row in files.iterrows():
    
    if Body == "Fed":
        year_month_day = row['date']
        doc_type = row['speaker']
        text = row['text']
    
    if Body == "ECB":
        year_month_day = row['date']
        doc_type = row['speakers']
        text = row['contents']
    
    text = text.casefold()
    text = re.sub(r"[^A-Za-z0-9.,!?/:;\s]+", "", text)
    
    segments = sent_tokenize(text, language='english')
    
    for segment in segments:
        segment_words = word_tokenize(segment, language='english')
        segment = ' '.join(segment_words)
        
        if 1 < len(segment) < 10000:
            final.append(segment)
            csv_out.append([year_month_day, doc_type, segment])
            keyword_freq_ts[year_month_day][doc_type].update(segment_words)
            global_word_counter.update(segment_words)
            
print(len(final))

# Sort words by frequency
sorted_word_freq = sorted(global_word_counter.items(), key=lambda x: x[1], reverse=True)

# Write the processed data to a JSON file
write_data(f"{Model_Folder}/{Model}_texts.json", final)
write_data(f"{Model_Folder}/keyword_freq_ts_{Model}.json", keyword_freq_ts)

df_csv_out = pd.DataFrame(csv_out, columns=["date", "type", "segment"])
df_csv_out.to_csv(f"{Model_Folder}/{Model}_texts.csv", index=True)

# Write sorted words and their frequencies to a file
with open(f"{Model_Folder}/{Model}_word_freq.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Word', 'Frequency'])
    for word, freq in sorted_word_freq:
        writer.writerow([word, freq])

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import torch 
import nltk
import spacy

# NLTK English stopwords
nlp = spacy.load("en_core_web_lg")
nltk.download('stopwords') 
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')
stopwords_fr = nltk.corpus.stopwords.words('french')
stopwords_it = nltk.corpus.stopwords.words('italian')
stopwords = stopwords_en + stopwords_sp + stopwords_fr + stopwords_it

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder
bert_models = config.bert_models
bert_models_local = config.bert_models_local
keywords = config.keywords

Body = config.Body
Model = config.Model
Model_Subfolder = f'/{Body} Texts/{Model}'
Model_Folder = config.texts
Model_Folder = Model_Folder + Model_Subfolder

df = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv")  
docs = df["segment"].to_list()
timestamps = df['date'].to_list()
type = df['type'].to_list()

# Embedding
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = embedding_model.encode(docs, 
                                    batch_size=32, 
                                    show_progress_bar=True)


#Reduce Dimensionality
umap_model = UMAP(n_neighbors=5, 
                  n_components=2, 
                  metric='cosine', 
                  n_epochs=500,
                  min_dist=0.0, 
                  target_metric_kwds=keywords, 
                  target_weight=0.95, 
                  verbose=True)

# Clustering model
cluster_model = HDBSCAN(min_cluster_size = 10, 
                        min_samples=10,
                        metric = 'euclidean', 
                        cluster_selection_method = 'eom', 
                        prediction_data = True)

#Representation model
representation_model = MaximalMarginalRelevance(diversity=0.4)

#Create UMAP model
vectorizer_model = CountVectorizer(stop_words=stopwords, ngram_range=(1, 3))

ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)

print("Done Preprocessing Data")
# BERTopic model
topic_model = BERTopic(language= 'english',
                       min_topic_size=15,
                       n_gram_range=(1, 3),
                       nr_topics = 64,
                       embedding_model=embedding_model,
                       umap_model=umap_model,
                       hdbscan_model=cluster_model,
                       vectorizer_model=vectorizer_model,
                       ctfidf_model=ctfidf_model,
                       representation_model=representation_model,
                       verbose=True
                       ).fit(docs, embeddings = embeddings)

print("Done Creating BERTopic Model")

torch.save(topic_model, f"{bert_models_local}/{Body}/{Model}/topic_model_{Model}.pt")
print("Done Saving BERTopic Model")

# Save topic-terms barcharts as HTML file
topic_model.visualize_barchart(top_n_topics = 32, n_words=8).write_html(f"{bert_models}/barchart.html")

topics_per_class = topic_model.topics_per_class(docs, classes=type)
topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=32).write_html(f"{bert_models}/topics_per_class.html")

topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=200)
#save topics over time graph as HTML file
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=32).write_html(f"{bert_models}/topics_over_time.html")

# Save intertopic distance map as HTML file
topic_model.visualize_topics().write_html(f"{bert_models}/intertopic_dist_map.html")

# Save documents projection as HTML file
topic_model.visualize_documents(docs).write_html(f"{bert_models}/projections.html")

# Save topics dendrogram as HTML file
topic_model.visualize_hierarchy().write_html(f"{bert_models}/hieararchy.html")

print("All Visuals Done")



import os 
import sys
import pandas as pd
import numpy as np
from umap import UMAP
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import matplotlib.pyplot as plt
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

# NLTK English stopwords
nlp = spacy.load("en_core_web_lg")
nltk.download('stopwords') 
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')
stopwords_fr = nltk.corpus.stopwords.words('french')
stopwords_it = nltk.corpus.stopwords.words('italian')
stopwords = stopwords_en + stopwords_sp + stopwords_fr + stopwords_it

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder
bert_models = config.bert_models
bert_models_local = config.bert_models_local
keywords = config.keywords
Sentiment_models = config.Sentiment_models

Body = config.Body
Model = config.Model
Model_Subfolder = f'/{Body} Texts/{Model}'
Model_Folder = config.texts
Model_Folder = Model_Folder + Model_Subfolder

df = pd.read_csv(f"{Model_Folder}/{Model}_texts.csv")  

Body_2 = config.Body_2
Model_2 = config.Model_2
Model_Subfolder_2 = f'/{Body_2} Texts/{Model_2}'
Model_Folder_2 = config.texts
Model_Folder_2 = Model_Folder_2 + Model_Subfolder_2

df_2 = pd.read_csv(f"{Model_Folder_2}/{Model_2}_texts.csv")  

df = pd.concat([df, df_2], axis=0)
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.columns)
out = []

sia = SentimentIntensityAnalyzer()

# #VADER Analysis 
for index, row in df.iterrows():
    docs = row["segment"]
    timestamps = row['date']
    type = row['type']
    docs = str(docs)
    
    good = sia.polarity_scores(docs)['pos']
    bad = sia.polarity_scores(docs)['neg']
    neutral = sia.polarity_scores(docs)['neu']
    compound = sia.polarity_scores(docs)['compound']
    out.append([timestamps, type, docs, good, bad, neutral, compound])
    
df_out = pd.DataFrame(out, columns=["date", "type", "segment","good", "bad", "neutral", "compound"])

df_out.to_csv(f"{Model_Folder}/{Model}_vader_sentiment_texts.csv")  
print(df_out.head())

#Adding Custom Sentiment Analysis 
increase_words = ['increase', 'raise', 'increasing', 'raising', 'higher', 'hike', 'hiking', 'increases', 'raises', 'hikes']
decrease_words = ['decrease', 'lower', 'decline', 'declining', 'cut', 'cutting', 'reducing', 'reduction', 'reduce']
funds_words = ['interest', 'rate', 'funds', 'federal funds', 'points']

pos_train = []
neg_train = []
neutral_data = []
other_data = []

for index, row in df.iterrows():

    docs = row["segment"]
    timestamps = row['date']
    type = row['type']
    docs = str(docs)

    interest = 0
    increase = 0
    decrease = 0
    good = 0
    bad = 0
    neutral = 0
    
    for word in docs.split():
        word = word.casefold()
        if word in funds_words:
            interest = 1
            
        if word in increase_words:
            increase = 1
 
        if word in decrease_words:
            decrease = 1
        
        if interest == 1 and increase == 1:
            bad = 1
            
        if interest == 1 and decrease == 1:
            good = 1  
        
        if interest == 1 and increase == 0 and decrease == 0:
            neutral = 1
    
    if good == 1:
        pos_train.append([timestamps, type, docs, interest, increase, decrease, good, bad, neutral])
        
    if bad == 1:
        neg_train.append([timestamps, type, docs, interest, increase, decrease, good, bad, neutral])
        
    if neutral == 1:
        neutral_data.append([timestamps, type, docs, interest, increase, decrease, good, bad, neutral])
        
    if good == 0 and bad == 0:
        other_data.append([timestamps, type, docs, interest, increase, decrease, good, bad, neutral])
    
    out.append([timestamps, type, docs, interest, increase, decrease, good, bad, neutral])

all_words = []
documents = []

#  j is adject, r is adverb, and v is verb
allowed_word_types = ["J","R","V"]
#allowed_word_types = ["J"]

pos_train = pd.DataFrame(pos_train, columns=["date", "type", "segment", "interest", "increase", "decrease", "good", "bad", "neutral"])
for index, row in pos_train.iterrows():
    
    docs = row["segment"]
    timestamps = row['date']
    type = row['type']
    docs = str(docs)
    
    documents.append( (docs, "pos") )
    
    #Tokenize 
    tokenized = word_tokenize(docs)

    #Remove Stopwords 
    cleaned = [w for w in tokenized if not w in stopwords]
    
    # parts of speech tagging for each word 
    pos = nltk.pos_tag(cleaned)
    
    # make a list of  all adjectives identified by the allowed word types list above
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

neg_train = pd.DataFrame(neg_train, columns=["date", "type", "segment", "interest", "increase", "decrease", "good", "bad", "neutral"])

for index, row in neg_train.iterrows():

    docs = row["segment"]
    timestamps = row['date']
    type = row['type']
    docs = str(docs)

    documents.append( (docs, "neg") )
    
    #Tokenize
    tokenized = word_tokenize(docs)
    
    #Remove Stopwords
    cleaned = [w for w in tokenized if not w in stopwords]
    
    # parts of speech tagging for each word 
    neg = nltk.pos_tag(cleaned)
    
    #Parts of the Speech Tagging for each word 
    for w in neg:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

pos_A = []
for w in pos:
    if w[1][0] in allowed_word_types:
        pos_A.append(w[0].lower())
        
pos_N = []
for w in neg:
    if w[1][0] in allowed_word_types:
        pos_N.append(w[0].lower())


from wordcloud import WordCloud
text = ' '.join(pos_A)
wordcloud = WordCloud().generate(text)

plt.figure(figsize = (15, 9))
# Display the generated image:
plt.imshow(wordcloud, interpolation= "bilinear")
plt.axis("off")
plt.show()

# pickling the list documents to save future recalculations 
save_documents = open(f'{Sentiment_models}/pickled_algos/documents.pickle', "wb")
pickle.dump(documents, save_documents)
save_documents.close()

BOW = nltk.FreqDist(all_words)
BOW

word_features = list(BOW.keys())[:5000]
word_features[0], word_features[-1]

save_word_features = open(f'{Sentiment_models}/pickled_algos/word_features5k.pickle', "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Creating features for each review
featuresets = [(find_features(doc), category) for (doc, category) in documents]
fs_scale = len(featuresets)

# Shuffling the documents 
random.shuffle(featuresets)

ts_var = fs_scale * 0.85
ts_var = int(ts_var)

training_set = featuresets[:ts_var]
testing_set = featuresets[ts_var:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

accuracy = nltk.classify.accuracy(classifier, testing_set) * 100
accuracy = str(round(accuracy, 2))
print(f"Baysian Machine Learning Classifier Accuracy:{accuracy}%")

classifier.show_most_informative_features(15)

mif = classifier.most_informative_features()
mif = [a for a,b in mif]
print(mif)

ground_truth = [r[1] for r in testing_set]

preds = [classifier.classify(r[0]) for r in testing_set]

f1_score(ground_truth, preds, labels = ['neg', 'pos'], average = 'micro')

y_test = ground_truth
y_pred = preds
class_names = ['neg', 'pos']

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

# training various models by passing in the sklearn models into the SklearnClassifier from NLTK 

MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_clf, testing_set))*100)

BNB_clf = SklearnClassifier(BernoulliNB())
BNB_clf.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BNB_clf, testing_set))*100)

LogReg_clf = SklearnClassifier(LogisticRegression())
LogReg_clf.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogReg_clf, testing_set))*100)

SGD_clf = SklearnClassifier(SGDClassifier())
SGD_clf.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGD_clf, testing_set))*100)

SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_clf, testing_set))*100)

def create_pickle(c, file_name): 
    save_classifier = open(file_name, 'wb')
    pickle.dump(c, save_classifier)
    save_classifier.close()

classifiers_dict = {'ONB': [classifier, f'{Sentiment_models}/pickled_algos/ONB_clf.pickle'],
                    'MNB': [MNB_clf, f'{Sentiment_models}/pickled_algos/MNB_clf.pickle'],
                    'BNB': [BNB_clf, f'{Sentiment_models}/pickled_algos/BNB_clf.pickle'],
                    'LogReg': [LogReg_clf, f'{Sentiment_models}/pickled_algos/LogReg_clf.pickle'],
                    'SGD': [SGD_clf, f'{Sentiment_models}/pickled_algos/SGD_clf.pickle'], 
                    'SVC': [SVC_clf, f'{Sentiment_models}/pickled_algos/SVC_clf.pickle']}

for clf, listy in classifiers_dict.items(): 
    create_pickle(listy[0], listy[1])


ground_truth = [r[1] for r in testing_set]
predictions = {}
f1_scores = {}
acc_scores = {}

for clf, listy in classifiers_dict.items(): 
    predictions[clf] = [listy[0].classify(r[0]) for r in testing_set]
    f1_scores[clf] = f1_score(ground_truth, predictions[clf], labels = ['neg', 'pos'], average = 'micro')
    print(f'f1_score {clf}: {f1_scores[clf]}')
    
for clf, listy in classifiers_dict.items(): 
    acc_scores[clf] = accuracy_score(ground_truth, predictions[clf])
    print(f'Accuracy_score {clf}: {acc_scores[clf]}')
    
from nltk.classify import ClassifierI

# Defininig the ensemble model class 
class EnsembleClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    # returns the classification based on majority of votes
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    # a simple measurement the degree of confidence in the classification 
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
# function to load models given filepath
def load_model(file_path): 
    classifier_f = open(file_path, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier

# Original Naive Bayes Classifier
ONB_Clf = load_model(f'{Sentiment_models}/pickled_algos/ONB_clf.pickle')

# Multinomial Naive Bayes Classifier 
MNB_Clf = load_model(f'{Sentiment_models}/pickled_algos/MNB_clf.pickle')

# Bernoulli  Naive Bayes Classifier 
BNB_Clf = load_model(f'{Sentiment_models}/pickled_algos/BNB_clf.pickle')

# Logistic Regression Classifier 
LogReg_Clf = load_model(f'{Sentiment_models}/pickled_algos/LogReg_clf.pickle')

# Stochastic Gradient Descent Classifier
SGD_Clf = load_model(f'{Sentiment_models}/pickled_algos/SGD_clf.pickle')

# Initializing the ensemble classifier 
ensemble_clf = EnsembleClassifier(ONB_Clf, MNB_Clf, BNB_Clf, LogReg_Clf, SGD_Clf)

# List of only feature dictionary from the featureset list of tuples 
feature_list = [f[0] for f in testing_set]

# Looping over each to classify each review
ensemble_preds = [ensemble_clf.classify(features) for features in feature_list]
    
f1_score(ground_truth, ensemble_preds, average = 'micro')

def sentiment(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats), ensemble_clf.confidence(feats)   

def sentiment_out_class(text):
    feats = find_features(text)
    return ensemble_clf.classify(feats)

def sentiment_out_conf(text):
    feats = find_features(text)
    return ensemble_clf.confidence(feats)   

other_data = pd.DataFrame(other_data, columns=["date", "type", "segment", "interest", "increase", "decrease", "good", "bad", "neutral"])

advanced_sent = []

print('Start Advanced Sentiment Analysis')
for index, row in df.iterrows():
    date = row["date"]
    type = row["type"]
    segment = row["segment"]
    feats = find_features(segment)
    seg_class = ensemble_clf.classify(feats)
    seg_conf = ensemble_clf.confidence(feats)  
    
    advanced_sent.append([date, type, segment, seg_class, seg_conf])
    
out_advanced_sent = pd.DataFrame(advanced_sent, columns=["date", "type", "segment", "sentiment", "confidence"])
print(out_advanced_sent.head())
out_advanced_sent.to_csv(f"{Model_Folder}/{Model}_advanced_sentiment_texts.csv")

df_out = pd.DataFrame(out, columns=["date", "type", "segment", "interest", "increase", "decrease", "good", "bad", "neutral"])
df_out.to_csv(f"{Model_Folder}/{Model}_sentiment_texts.csv")  