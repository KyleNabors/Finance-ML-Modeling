from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

# Prepare data
data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
docs = data["data"]
categories = [data["target_names"][category] for category in data["target"]]

# Train model on all data
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

import pandas as pd
documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics,
                          "Category": categories})

# We slice the data and extract the topics from that subset
# In other words, instead of "sci.electronics", you would select "Author A"
subset = documents.loc[documents.Category == "sci.electronics", :]
subset_labels = sorted(list(subset.Topic.unique()))

# First, we group the documents per topic
documents_per_topic = subset.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})

# Then we calculate the c-TF-IDF representation but we do not fit this method 
# as it was already fitted on the entire dataset
topic_model.c_tf_idf, words = topic_model._c_tf_idf(documents_per_topic, fit=False)

# Lastly, we extract the words per topic based on the subset_labels,
# and we update the topic size to correspond with the subset
topic_model.topics = topic_model._extract_words_per_topic(words, labels=subset_labels)
topic_model._create_topic_vectors()
topic_model.topic_names = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                    for key, values in
                    topic_model.topics.items()}
topic_model._update_topic_size(subset)