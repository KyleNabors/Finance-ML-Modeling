# Import necessary libraries
import json
import nltk
import gensim
import gensim.corpora as corpora
import spacy
from gensim.models import TfidfModel
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
import pyLDAvis.gensim

# Download stopwords from nltk
nltk.download("stopwords") 

# Load and write JSON data
def load_data(file):
    with open(file) as f:
        data = json.load(f)
    return data
    
def write_data(file, data):
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Load stopwords
stopwords = stopwords.words('english')

# Load data
data = load_data("/Users/kylenabors/Documents/GitHub/MS-Thesis/Training Data/ushmm_dn.json")["texts"]

# Lemmatization
def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for sent in texts:
        doc = nlp(sent)
        new_text = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        texts_out.append(" ".join(new_text))
    return texts_out

# Apply lemmatization
lemmatized_texts = lemmatization(data)

# Generate word tokens
def gen_words(texts):
    return [gensim.utils.simple_preprocess(text, deacc=True) for text in texts]

data_words = gen_words(lemmatized_texts)

# Create Bigrams and Trigrams
bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigrams_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=100)

bigrams = gensim.models.phrases.Phraser(bigrams_phrases)
trigrams = gensim.models.phrases.Phraser(trigrams_phrases)

data_bigrams = [bigrams[doc] for doc in data_words]
data_bigrams_trigrams = [trigrams[bigrams[doc]] for doc in data_bigrams]

# TF-IDF Removal
id2word = corpora.Dictionary(data_bigrams_trigrams)
corpus = [id2word.doc2bow(text) for text in data_bigrams_trigrams]
tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
for i, bow in enumerate(corpus):
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    words_missing_in_tfidf = [id for id, _ in bow if id not in [id for id, _ in tfidf[bow]]]
    drops = low_value_words + words_missing_in_tfidf
    corpus[i] = [b for b in bow if b[0] not in drops]

# Build LDA Model
lda_model = LdaModel(corpus=corpus[:-1],
                     id2word=id2word,
                     num_topics=30,
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha="auto")   

# Test LDA Model
test_doc = corpus[-1]
vector = sorted(lda_model[test_doc], key=lambda x: x[1], reverse=True)
print(vector)

# Save and load LDA Model
lda_model.save("/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/test_model.model")
new_model = LdaModel.load("/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/test_model.model")

# Test new model
test_doc = corpus[-1]
vector = sorted(new_model[test_doc], key=lambda x: x[1], reverse=True)
print(vector)

# Visualize Topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=30)
pyLDAvis.save_html(vis, "/Users/kylenabors/Documents/GitHub/MS-Thesis/Models/USHMM DN.html")
