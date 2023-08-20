# Import necessary libraries
import sys
import os
import json
import pdfplumber
import csv
from collections import defaultdict, Counter
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import spacy
import nltk
from nltk.corpus import words, stopwords
import pandas as pd

nlp = spacy.load("en_core_web_lg")

nltk.download("stopwords") 
stopwords = stopwords.words('english')

nltk.download('words')
hyphenated_words = set(word for word in words.words() if '-' in word)

#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder

# Load data from the JSON database
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Write data to a JSON file
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Handle hyphenation in text
def handle_hyphenation(text):
    for word in text.split():
        if '-' in word and word not in hyphenated_words:
            text = text.replace(word, word.replace('-', ''))
    return text
#Handle stopwords in text
def handle_stopwords(text):
    for word in text.split():
        if word in stopwords:
            text = text.replace(word, '')
    return text

# Load the data from the JSON database
database_data = load_data(database_file)

# Define accepted types
accepted_types = ['Beige Book']  # replace these with actual types

# Extract file paths from the database data
files = [(entry["path"], entry["date"][:10], entry["type"]) for entry in database_data if "path" in entry and entry["type"] in accepted_types] 

# Sort files by date
files.sort(key=lambda x: x[1])  # sort by year_month_day

# Specify the percentage of files you want to process
percentage_to_process = .1
files_to_process = files[::int(1 / percentage_to_process)]

# Define the ranges
year_ranges = config.year_ranges

# Create a dictionary to store the processed segments for each range
segments_by_range = {range_: [] for range_ in year_ranges}

# List to hold processed segments from the PDF files
final = []
remove_words = ["report"]

# Process each PDF file
keyword_freq_ts = defaultdict(lambda: defaultdict(Counter))
for file, year_month_day, doc_type in files_to_process:
    year = int(year_month_day[:4])  # extract the year from the year_month_day string
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            text = handle_hyphenation(text)
            doc = nlp(text)  # pass the text into the Spacy NLP model
            text = text.replace("\n", " ").lower()  # Convert to lower case once
            segments = text.split('. ')
            for segment in segments:
                # Remove unwanted words and extra spaces
                segment = ' '.join([word.strip() for word in segment.split() if word not in remove_words])
                if 5 < len(segment) < 350:
                    for range_ in year_ranges:
                        if range_[0] <= year_month_day <= range_[1]:
                            segments_by_range[range_].append(segment)
                            #for word in segment.split():
                                #keyword_freq_ts[range_][doc_type][word] += 1

# Save the processed data for each range to a JSON file
four_models_datapath = config.four_models_datapath
for range_, segments in segments_by_range.items():
    write_data(f"{four_models_datapath}/fed_data_blocks_{range_[0]}_{range_[1]}.json", segments)
    #write_data(f"{four_models_datapath}/keyword_freq_ts_{range_[0]}_{range_[1]}.json", keyword_freq_ts)

