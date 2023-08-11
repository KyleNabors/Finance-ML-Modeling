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
from nltk.corpus import words
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_lg")

nltk.download("stopwords") 
stopwords = stopwords.words('english')

nltk.download('words')
hyphenated_words = set(word for word in words.words() if '-' in word)

#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

print(config_path)

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database

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
percentage_to_process = 1
files_to_process = files[::int(1 / percentage_to_process)]

# Define the ranges
year_ranges = config.year_ranges

# Create a dictionary to store the processed segments for each range
segments_by_range = {range_: [] for range_ in year_ranges}

# Specify the year and month you want to start and end processing files from
start_year_month_day = '2007-12-01'
end_year_month_day = '2023-12-31'

# Only process files from the selected year and month range
#files_to_process = [file for file in files_to_process if start_year_month_day <= file[1] <= end_year_month_day]

# List to hold processed segments from the PDF files
final = []
compare = []
remove_words = ["report", 'cid', 'i', 'v', '(cid', '(cid)', '(cid:', '(cid: ',
                'reported', 'reporting', 'district', 'federal', 'reserve', 'districts']

# Process each PDF file
keyword_freq_ts = defaultdict(lambda: defaultdict(Counter))
for file, year_month_day, doc_type in files_to_process:
    year = int(year_month_day[:4]) # extract the year from the year_month string
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            text = handle_hyphenation(text)
            #text = handle_stopwords(text)
            doc = nlp(text)  # pass the text into the Spacy NLP model
            text = text.replace("\n", " ")
            segments = text.split('. ') 
            for segment in segments:
                segment = segment.lower()
                for word in segment.split():
                    word = word.lower()
                    word = word.strip()
                    if word in remove_words:
                        segment = segment.replace(word, '')
                        segment = segment.replace('  ', ' ')
                if len(segment) > 5 and len(segment) < 350:
                    final.append(segment)
                    
print(len(final))

# Write the processed data to a JSON file
write_data("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/fed_data_blocks.json", final)
write_data("/Users/kylenabors/Documents/MS-Thesis Data/Database/Fed Data/keyword_freq_ts_blocks.json", keyword_freq_ts)