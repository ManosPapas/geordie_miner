import os
import re
import PyPDF2
import csv
import shutil
import numpy as np
import pandas as pd
from gensim import matutils
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from datetime import datetime
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from wordcloud import WordCloud
import _functions


###############################################################################################
# Initialise
###############################################################################################
# Clear the screen (this may differ based on the operating system)
# os.system('cls' if os.name == 'nt' else 'clear')

# Load settings, save logs, and initialise directories
config = _functions.load_config()

language = _functions.language
directory_analysis = _functions.directory_analysis
directory_data = _functions.directory_data
directory_text = _functions.directory_text
directory_processed = _functions.directory_processed

#logging function
def logger(message):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{current_time} - {message}"
    print(log_entry)
    log_file_path = os.path.join(directory_analysis, "_log_output.log")
    with open(log_file_path, "a", encoding='utf-8') as log_file:
        log_file.write(log_entry+ "\n")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

logger("")
logger("")    
logger("")    
logger("START")
logger("")    
logger("=========================================================")
logger("Geordie Miner")
logger("Last updated: 11/24")    
logger("=========================================================")
logger("")
logger(f"Runtime: {now}")    
logger("")    
logger("Configuration")
logger(f"Language: {language}")
logger(f"Directory Data: {directory_data}")
logger(f"Directory Analysis: {directory_analysis}")


# Set the NLTK data path to a _nltk_data folder in the current working directory
nltk_data_dir = os.path.join(os.getcwd(), "_nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
logger(f"NLTK setup:")

# Check and download the necessary resources only if they are not present
resources = {"stopwords": "corpora","punkt": "tokenizers","wordnet": "corpora"}
for resource in resources:
    try:
        nltk.data.find(f"corpora/{resource}")  # For corpora like stopwords and wordnet
        logger(f" Exists: {resource}")
    except LookupError:
        logger(f" Downloading: {resource}...")
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)

logger("NLTK setup complete.")
logger("")    
logger("")    

###############################################################################################
# PDF to text
###############################################################################################

# Check if the "data" directory exists
if not os.path.exists(directory_data):
    logger(f"Directory '{directory_data}' does not exist. Exiting.")
    exit()

    
# Remove the "analysis" directory if it exists
if os.path.exists(directory_analysis):
    shutil.rmtree(directory_analysis)

# Create the "text" directory
os.makedirs(directory_text)

# List all files in the "data" directory
all_files = sorted(os.listdir(directory_data))


# List all files in the "data" directory and sort alphabetically
all_files = sorted(os.listdir(directory_data))

# Process each file, with index tracking
for index, file_name in enumerate(tqdm(all_files, desc="Processing files")):
    file_path = os.path.join(directory_data, file_name)
    
    # Format the target file name with an index prefix
    numbered_file_name = f"{index + 1:03d}__{file_name}"  # e.g., "001__filename.txt"
    target_file_path = os.path.join(directory_text, numbered_file_name)

    # Check if the file is a text file and copy directly
    if file_name.lower().endswith('.txt'):
        lmsg="Text copying completed"
        try:
            shutil.copy(file_path, target_file_path)
        except Exception as e:
            logger(f"Failed to copy '{file_name}': {e}")
        continue

    # Check if the file is a PDF file and export as txt
    elif file_name.lower().endswith('.pdf'):
        lmsg="PDF to text conversion completed"
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []

                # Extract text from each page
                for page in range(len(reader.pages)):
                    text.append(reader.pages[page].extract_text())

                # Save the extracted text to a numbered text file
                with open(target_file_path.replace('.pdf', '.txt'), 'w', encoding='utf-8') as text_file:
                    text_file.write('\n'.join(text))

        except Exception as e:
            logger(f"Failed to process '{file_name}': {e}")

logger(lmsg)



# Check if the "text" directory exists
if not os.path.exists(directory_text):
    logger(f"Directory '{directory_text}' does not exist.")
    exit()
    
# Remove the "processed" directory if it exists
if os.path.exists(directory_processed):
    shutil.rmtree(directory_processed)

# Create the "processed" directory
os.makedirs(directory_processed)

###############################################################################################
# Save corpus descriptive data
###############################################################################################
_functions.descriptive_stats(directory_text)


###############################################################################################
# Prepreocessing
###############################################################################################
# Load stopwords preserving structure
custom_stopwords = []
if (stopwords_file := config.get('preprocessing', 'stopwords_file', fallback='stopwords.txt')) is not None and os.path.exists(stopwords_file):
    with open(stopwords_file, "r", encoding="utf-8") as f:
        custom_stopwords = [line.strip().lower() for line in f if line.strip()]

nltk_stopwords = stopwords.words(language)
combined_stopwords = custom_stopwords + nltk_stopwords
with open(os.path.join(directory_analysis,"_log_stopwords.txt"), "w", encoding="utf-8") as log:
    log.write("\n".join(combined_stopwords))

def remove_stopwords(text, stopwords):
    pattern = r"\b(" + "|".join(map(re.escape, stopwords)) + r")\b"
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()
    
# Load substitutions
substitutions = {}
substitutions_file = config.get('preprocessing', 'substitutions_file', fallback='substitutions.txt')
with open(substitutions_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip().lower()
        if line:
            original, replacement = line.split(",", 1)
            substitutions[original.strip()] = replacement.strip()

with open(os.path.join(directory_analysis,"_log_substitutions.txt"), "w", encoding="utf-8") as log:
    for key, val in substitutions.items():
        log.write(f"{key} -> {val}\n")

def apply_substitutions(text, substitutions):
    for original, replacement in substitutions.items():
        text = re.sub(rf"\b{re.escape(original)}\b[\s.,]*", replacement+"  ", text, flags=re.IGNORECASE)
    return text

# Min frequency
min_frequency = config.getint('preprocessing', 'min_frequency', fallback=5)

def remove_low_frequency_terms(text, min_frequency):
    # Step 1: Count all words in the entire text
    all_words = text.split()
    word_counts = Counter(all_words)

    # Step 2: Process line by line, filtering low-frequency words
    filtered_lines = []
    for line in text.splitlines():
        words = line.split()
        filtered = [word for word in words if word_counts[word] >= min_frequency]
        filtered_lines.append(" ".join(filtered))  # always append, even if result is empty

    # Step 3: Join the lines back together with preserved structure
    return "\n".join(filtered_lines)

###########################################################################
# Process files
corpus = []  # To store the processed text for analysis
for filename in tqdm(os.listdir(directory_text), desc="Processing files"):
    if filename.endswith(".txt"):
        with open(os.path.join(directory_text, filename), "r", encoding="utf-8") as f:
            text = f.read().lower()
            text = re.sub(r'-\s*\n\s*', "", text)                                   # Join split terms with - over lines breaks
            
            text = apply_substitutions(text, substitutions)                         # Apply substitutions from the provided list
            text = remove_stopwords(text, combined_stopwords)                       # Remove stopwords
            
            text = re.sub(r"http\S+|www\.\S+", " ", text)                           # Remove web links
            text = re.sub(r"\(.*?\)", " ", text)                                    # Remove parentheses and their contents
            text = re.sub(r"[^a-zA-Z\s]", " ", text)                                # Remove all non-alphabetic characters (keeping whitespace)
            text = re.sub(r"\b\w{1}\b", " ", text)                                  # Remove single-character terms
            #text = re.sub(r"\s+", " ", text).strip()                                # Replace multiple spaces with a single space
            
            text = apply_substitutions(text, substitutions)                         # Apply substitutions from the provided list -- NOTE: this will only replace full words, ensure that plurals are treated appropriately
            text = remove_stopwords(text, combined_stopwords)                       # Remove stopwords 
            text = remove_low_frequency_terms(text, min_frequency=min_frequency)    # Remove low-frequency terms
            
        # Save processed text
        with open(os.path.join(directory_processed, filename), "w", encoding="utf-8") as f:
            f.write(text)
        
        corpus.append(text.split())  # Append processed text to corpus for term analysis
logger("Preprocessing completed")

###############################################################################################
# Save corpus descriptive data
###############################################################################################
_functions.descriptive_stats(directory_processed)



###############################################################################################
# Term Analysis 
###############################################################################################

# Instantiate the lemmatizer
lemmatiser = WordNetLemmatizer()

# Function to lemmatise and clean text
def lemmatise_text(text):
    tokens = word_tokenize(text)  # Tokenise the text
    lemmatised_tokens = [lemmatiser.lemmatize(token.lower()) for token in tokens if token.isalpha()]  # Lowercase and remove punctuation
    return lemmatised_tokens  # Return lemmatised tokens for further processing

# Lemmatise each document in the corpus
lemmatised_corpus = [lemmatise_text(" ".join(doc)) for doc in corpus]

# Term Analysis 
top_n_terms = int(config.get('term_analysis', 'top_n_terms', fallback=200))
wordcloud_width = int(config.get('term_analysis', 'wordcloud_width', fallback=800))
wordcloud_height = int(config.get('term_analysis', 'wordcloud_height', fallback=400))
wordcloud_background_color = config.get('term_analysis', 'wordcloud_background_color', fallback='white')
output_wordcloud = config.get('term_analysis', 'output_wordcloud', fallback='True') 

# Frequency analysis for raw terms
all_terms_raw = [word for doc in corpus for word in doc]
term_counter_raw = Counter(all_terms_raw)
total_docs_with_term_raw = {term: sum(1 for doc in corpus if term in doc) for term in term_counter_raw}
total_docs_count_raw = len(corpus)

# Frequency analysis for lemmatised terms
all_terms_lemmatised = [word for doc in lemmatised_corpus for word in doc]
term_counter_lemmatised = Counter(all_terms_lemmatised)
total_docs_with_term_lemmatised = {term: sum(1 for doc in lemmatised_corpus if term in doc) for term in term_counter_lemmatised}
total_docs_count_lemmatised = len(lemmatised_corpus)

# Prepare for TF-IDF analysis on raw terms
tfidf_vectorizer_raw = TfidfVectorizer()
# Fit the vectorizer and transform the raw corpus into a TF-IDF matrix
tfidf_matrix_raw = tfidf_vectorizer_raw.fit_transform([' '.join(doc) for doc in corpus])

# Prepare for TF-IDF analysis on lemmatised terms
tfidf_vectorizer_lemmatised = TfidfVectorizer()
# Fit the vectorizer and transform the lemmatised corpus into a TF-IDF matrix
tfidf_matrix_lemmatised = tfidf_vectorizer_lemmatised.fit_transform([' '.join(doc) for doc in lemmatised_corpus])

# Prepare top N terms analysis for raw terms
top_terms_raw = term_counter_raw.most_common(top_n_terms)
analysis_terms_raw = []

for term, count in top_terms_raw:
    doc_freq = total_docs_with_term_raw[term]
    percentage_docs = (doc_freq / len(corpus)) * 100 if corpus else 0

    # Check if the term exists in the vocabulary to avoid KeyError
    if term in tfidf_vectorizer_raw.vocabulary_:
        term_index = tfidf_vectorizer_raw.vocabulary_[term]
        tfidf_value = tfidf_matrix_raw[:, term_index].sum()  # Get the sum of TF-IDF values for the term
    else:
        tfidf_value = 0  # Default to 0 if the term is not found

    # Append as a tuple instead of a formatted string
    analysis_terms_raw.append((term, count, percentage_docs, tfidf_value))

# Prepare top N terms analysis for lemmatised terms
top_terms_lemmatised = term_counter_lemmatised.most_common(top_n_terms)
analysis_terms_lemmatised = []

for term, count in top_terms_lemmatised:
    doc_freq = total_docs_with_term_lemmatised[term]
    percentage_docs = (doc_freq / len(lemmatised_corpus)) * 100 if lemmatised_corpus else 0

    # Check if the term exists in the vocabulary to avoid KeyError
    if term in tfidf_vectorizer_lemmatised.vocabulary_:
        term_index = tfidf_vectorizer_lemmatised.vocabulary_[term]
        tfidf_value = tfidf_matrix_lemmatised[:, term_index].sum()  # Get the sum of TF-IDF values for the term
    else:
        tfidf_value = 0  # Default to 0 if the term is not found

    # Append as a tuple instead of a formatted string
    analysis_terms_lemmatised.append((term, count, percentage_docs, tfidf_value))

# Create DataFrames with proper column names for raw and lemmatised terms
df_analysis_terms_raw = pd.DataFrame(analysis_terms_raw, columns=["Term", "Count", "% of Docs", "TF-IDF"])
df_analysis_terms_lemmatised = pd.DataFrame(analysis_terms_lemmatised, columns=["Term", "Count", "% of Docs", "TF-IDF"])

# Save to Excel if the option is enabled
df_analysis_terms_raw.to_excel(os.path.join(directory_analysis, "analysis_terms_single_raw.xlsx"), index=False)
df_analysis_terms_lemmatised.to_excel(os.path.join(directory_analysis, "analysis_terms_single_lemmatised.xlsx"), index=False)

logger("Term analysis completed for raw and lemmatised terms")



###############################################################################################
# Generate word cloud if the option is enabled
###############################################################################################
if output_wordcloud:
    wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height, background_color=wordcloud_background_color).generate_from_frequencies(term_counter_raw)
    wordcloud.to_file(os.path.join(directory_analysis, "analysis_descriptive_raw_wordcloud.jpg"))
    logger("Wordcloud (raw terms) exported")
    
    wordcloud = WordCloud(width=wordcloud_width, height=wordcloud_height, background_color=wordcloud_background_color).generate_from_frequencies(term_counter_raw)
    wordcloud.to_file(os.path.join(directory_analysis, "analysis_descriptive_lemmatised_wordcloud.jpg"))
    logger("Wordcloud (lemmatised terms) exported")    
    
    

    
    
###############################################################################################
# Phrases
###############################################################################################

# Instantiate the lemmatiser
lemmatiser = WordNetLemmatizer()

# Function to lemmatise and clean text
def lemmatise_text(text):
    tokens = word_tokenize(text)  # Tokenise the text
    lemmatised_tokens = [lemmatiser.lemmatize(token.lower()) for token in tokens if token.isalpha()]  # Lowercase and remove punctuation
    return lemmatised_tokens  # Return lemmatised tokens for further processing

bigram_threshold = config.getint('phrase_analysis', 'bigram_threshold', fallback=10)
trigram_threshold = config.getint('phrase_analysis', 'trigram_threshold', fallback=10)
bigram_export_count = config.getint('phrase_analysis', 'bigram_export_count', fallback=100)
trigram_export_count = config.getint('phrase_analysis', 'trigram_export_count', fallback=100)
window_size = config.getint('phrase_analysis', 'window_size', fallback=5)
cooccurrence_threshold = config.getint('phrase_analysis', 'cooccurrence_threshold', fallback=3)
clustering_metric = config.get('phrase_analysis', 'clustering_metric', fallback='jaccard')
linkage_method = config.get('phrase_analysis', 'linkage_method', fallback='ward')
dendrogram_figsize = tuple(map(int, config.get('phrase_analysis', 'dendrogram_figsize', fallback='10,7').split(',')))

# Read files and create a corpus with lemmatisation
corpus = []
file_names = []

for filename in os.listdir(directory_processed):
    if filename.endswith(".txt"):
        try:
            with open(os.path.join(directory_processed, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                lemmatised_content = lemmatise_text(content)
                corpus.extend(lemmatised_content)
                file_names.append(filename)
                # logger(f"Successfully read and lemmatised {filename}. Number of tokens: {len(lemmatised_content)}")

        except Exception as e:
            logger(f"Failed to read {filename}: {e}")

            
###########################################################################
# Frequency Count with Simple Tokenisation
#word_count = Counter(corpus)
#with open(os.path.join(directory_analysis,"analysis_term.txt"), "w", encoding='utf-8') as f:
#    for word, count in word_count.most_common():
#        f.write(f"{word}: {count}\n")
#logger("Frequency count with tokenisation")

###########################################################################
# Term Frequency with Word Pairs
word_pairs = Counter(ngrams(corpus, 2))

# Write bigram term frequencies to a CSV file
#with open(os.path.join(directory_analysis,"analysis_terms_pairs.txt"), "w", encoding='utf-8', newline='') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerow(["ngram", "frequency"])
#    for pair, count in word_pairs.most_common():
#        writer.writerow([f"{pair[0]} {pair[1]}", count])
#logger("Term frequency with word pairs")

###########################################################################
# Get a list of 2 terms n-grams with frequency >= bigram_threshold
bigram_freq = word_pairs.most_common()
bigram_freq_filtered = [(f"{pair[0]} {pair[1]}", count) for pair, count in bigram_freq if count >= bigram_threshold]
bigram_freq_filtered_sorted = sorted(bigram_freq_filtered, key=lambda x: x[1], reverse=True)[:bigram_export_count]

with open(os.path.join(directory_analysis,"analysis_terms_ngram2.csv"), "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ngram", "frequency"])
    writer.writerows(bigram_freq_filtered_sorted)
logger(f"Get a list of 2 terms n-grams with frequency >= {bigram_threshold}")


###########################################################################
# Get a list of 3 terms n-grams with frequency >= trigram_threshold
trigram_freq = Counter(ngrams(corpus, 3))
trigram_freq_filtered = [(f"{triple[0]} {triple[1]} {triple[2]}", count) for triple, count in trigram_freq.items() if count >= trigram_threshold]
trigram_freq_filtered_sorted = sorted(trigram_freq_filtered, key=lambda x: x[1], reverse=True)[:trigram_export_count]

with open(os.path.join(directory_analysis,"analysis_terms_ngram3.csv"), "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ngram", "frequency"])
    writer.writerows(trigram_freq_filtered_sorted)
logger(f"Get a list of 3 terms n-grams with frequency >= {trigram_threshold}")

###############################################################################################
# Co-occurrence Analysis
###############################################################################################
cooccurrence_counts = defaultdict(Counter)

for sentence in sent_tokenize(" ".join(corpus)):
    sentence_tokens = lemmatise_text(sentence)
    for i, token in enumerate(sentence_tokens):
        for j in range(i+1, min(i+1+window_size, len(sentence_tokens))):
            cooccurrence_counts[token][sentence_tokens[j]] += 1
            cooccurrence_counts[sentence_tokens[j]][token] += 1

with open(os.path.join(directory_analysis,"analysis_cooccurrence.csv"), "w", encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["term1", "term2", "cooccurrence_count"])
    for term1, neighbours in cooccurrence_counts.items():
        for term2, count in neighbours.items():
            writer.writerow([term1, term2, count])
logger("Co-occurrence analysis completed")


###########################################################################
# Co-occurrence Network Analysis
###########################################################################
G = nx.Graph()

# Add edges to the network
for term1, neighbours in cooccurrence_counts.items():
    for term2, count in neighbours.items():
        if count >= cooccurrence_threshold:  # Filter threshold for meaningful co-occurrences
            G.add_edge(term1, term2, weight=count)

# Export to GEXF for visualisation in tools like Gephi if enabled
nx.write_gexf(G, os.path.join(directory_analysis,"analysis_cooccurrence_network.gexf"))
logger("Co-occurrence network saved")

###########################################################################
# Agglomerative Hierarchical Clustering (AHC) with Jaccard Distance
###########################################################################
terms = list(cooccurrence_counts.keys())
matrix_size = len(terms)

# Create a binary co-occurrence matrix
binary_cooccurrence_matrix = np.zeros((matrix_size, matrix_size))

for i, term1 in enumerate(terms):
    for j, term2 in enumerate(terms):
        binary_cooccurrence_matrix[i, j] = 1 if cooccurrence_counts[term1][term2] > 0 else 0

# Compute the Jaccard distance and ensure we use the correct matrix format
distance_matrix = pdist(binary_cooccurrence_matrix, metric=clustering_metric)  # Get condensed distance matrix

# Proceed with the linkage method on the condensed distance matrix
linkage_matrix = linkage(distance_matrix, method=linkage_method)

plt.figure(figsize=dendrogram_figsize)
plt.title("Agglomerative Hierarchical Clustering Dendrogram (Jaccard Distance)")
dendrogram(linkage_matrix, labels=terms, orientation='right')
plt.savefig( os.path.join(directory_analysis,"analysis_ahc_dendrogram_jaccard.png") )
#plt.show()
logger(f"Agglomerative Hierarchical Clustering Completed with Jaccard Distance and dendrogram saved.")




###############################################################################################
# Topic modelling
###############################################################################################
topic_modelling_multi1 = config.getint('topic_modelling', 'topic_modelling_multi1', fallback=0)
topic_modelling_multi2 = config.getint('topic_modelling', 'topic_modelling_multi2', fallback=0)
topic_modelling_multi3 = config.getint('topic_modelling', 'topic_modelling_multi3', fallback=0)

# Read files and create a corpus with lemmatisation
corpus = []
file_names = []

for filename in os.listdir(directory_processed):
    if filename.endswith(".txt"):
        with open(os.path.join(directory_processed, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            lemmatised_content = lemmatise_text(content)
            corpus.append(lemmatised_content)
            file_names.append(filename)
corpus = [" ".join(doc) if isinstance(doc, list) else doc for doc in corpus]


###########################################################################
# 1. KMeans clustering
kmeans_topics = config.getint('topic_kmeans', 'kmeans_topics', fallback=10)

def kmeans_clustering(corpus, num_clusters):
    # Vectorize the corpus using CountVectorizer
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(corpus)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(doc_term_matrix)

    # Get cluster assignments for each document
    cluster_assignments = kmeans.labels_

    # Get the cluster centres (centroids) of the clusters
    cluster_centroids = kmeans.cluster_centers_

    # Save the cluster assignments to a file
    with open(os.path.join(directory_analysis, f"analysis_topicmodel_KMeans_{num_clusters}_doc2topic_assignments.txt"), "w", encoding='utf-8') as f:
        for idx, cluster in enumerate(cluster_assignments):
            file_id = file_names[idx].split('__')[0]  # Extract file_id as in previous functions
            f.write(f"{file_id}, {cluster + 1}\n")  # Write file_id and cluster assignment

    # Save the cluster centroids to a file
    with open(os.path.join(directory_analysis, f"analysis_topicmodel_KMEANS_cluster_centroids_{num_clusters}.txt"), "w", encoding='utf-8') as f:
        for i, centroid in enumerate(cluster_centroids):
            # Get the top terms closest to the cluster centroid
            top_terms_idx = centroid.argsort()[-10:][::-1]  # Get top 10 terms closest to centroid
            top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_idx]
            f.write(f"Cluster {i + 1}: {', '.join(top_terms)}\n")

    return cluster_assignments, doc_term_matrix


# Run the KMeans topic modeling
cluster_assignments, doc_term_matrix = kmeans_clustering(corpus, kmeans_topics)
logger(f"KMeans completed: {kmeans_topics} topics")
if topic_modelling_multi1 > 0:
    cluster_assignments, doc_term_matrix = kmeans_clustering(corpus, kmeans_topics*topic_modelling_multi1)
    logger(f"KMeans completed: {kmeans_topics*topic_modelling_multi1} topics")
if topic_modelling_multi2 > 0:
    cluster_assignments, doc_term_matrix = kmeans_clustering(corpus, kmeans_topics*topic_modelling_multi2)
    logger(f"KMeans completed: {kmeans_topics*topic_modelling_multi2} topics")
    
if topic_modelling_multi3 > 0:
    cluster_assignments, doc_term_matrix = kmeans_clustering(corpus, kmeans_topics*topic_modelling_multi3)
    logger(f"KMeans completed: {kmeans_topics*topic_modelling_multi3} topics")


###########################################################################
# 2. Topic Modelling using LDA
lda_topics = config.getint('topic_lda', 'lda_topics', fallback=10)
lda_terms_per_topic = config.getint('topic_lda', 'lda_terms_per_topic', fallback=10)
lda_passes = config.getint('topic_lda', 'lda_passes', fallback=25)
lda_per_word_topics = config.getint('topic_lda', 'lda_per_word_topics', fallback=0)

def get_lda_top_terms(model, num_words):
    topics = model.show_topics(num_words=num_words, formatted=False)
    top_terms = {}
    for idx, topic in topics:
        top_terms[f"Topic {idx + 1}"] = [(word, round(weight, 4)) for word, weight in topic]
    return top_terms

def lda_topic_modeling(corpus, num_topics, passes, terms_per_topic, per_word_topics):
    # Ensure each document in the corpus is a single string, not a list of tokens
    corpus = [" ".join(doc) if isinstance(doc, list) else doc for doc in corpus]
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(corpus)

    #logger(f"Document-Term Matrix Shape: {doc_term_matrix.shape}")
    id2word = dict(enumerate(vectorizer.get_feature_names_out()))
    bow_corpus = matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)

    # Initialize the LDA model with per_word_topics parameter
    lda = LdaModel(corpus=bow_corpus, num_topics=num_topics, id2word=id2word, passes=passes, iterations=1000, alpha='auto', eta='auto', random_state=42, per_word_topics=per_word_topics)

    lda_results = get_lda_top_terms(lda, num_words=terms_per_topic)
    topic_doc_counts = {f"Topic {i + 1}": 0 for i in range(num_topics)}

    document_topics = []
    for doc in bow_corpus:
        topic_probs = lda.get_document_topics(doc)
        assigned_topic = max(topic_probs, key=lambda x: x[1])[0]
        document_topics.append(assigned_topic)
        topic_doc_counts[f"Topic {assigned_topic + 1}"] += 1

    # Save the LDA topic modeling results to a file
    with open(os.path.join(directory_analysis, f"analysis_topicmodel_LDA_{num_topics}.txt"), "w", encoding='utf-8') as f:
        for i in range(num_topics):
            topic_label = f"Topic {i + 1}"
            terms = lda.show_topic(i, topn=terms_per_topic)  # Fetch terms for each topic
            f.write(f"{topic_label} ({topic_doc_counts[topic_label]} docs): ")
            f.write(", ".join([f"{term} ({weight:.4f})" for term, weight in terms]) + "\n")


    # Save document-topic assignments to a file
    with open(os.path.join(directory_analysis, f"analysis_topicmodel_LDA_{num_topics}_doc2topic_assignments.txt"), "w", encoding='utf-8') as f:
        for idx, topic in enumerate(document_topics):
            if idx < len(file_names):
                file_id = file_names[idx].split('__')[0]  # Extract file_id as in HDP function
                f.write(f"{file_id}, {topic + 1 if topic != -1 else 'No Topic'}\n")

    return document_topics

# Run the LDA topic modeling
lda_topic_modeling(corpus, num_topics=lda_topics, passes=lda_passes, terms_per_topic=lda_terms_per_topic, per_word_topics=lda_per_word_topics)
logger(f"LDA completed: {lda_topics} topics")
if topic_modelling_multi1 > 0:
    lda_topic_modeling(corpus, num_topics=lda_topics*topic_modelling_multi1, passes=lda_passes, terms_per_topic=lda_terms_per_topic, per_word_topics=lda_per_word_topics)
    logger(f"LDA completed: {lda_topics*topic_modelling_multi1} topics")
if topic_modelling_multi2 > 0:
    lda_topic_modeling(corpus, num_topics=lda_topics*topic_modelling_multi2, passes=lda_passes, terms_per_topic=lda_terms_per_topic, per_word_topics=lda_per_word_topics)
    logger(f"LDA completed: {lda_topics*topic_modelling_multi2} topics")
if topic_modelling_multi3 > 0:
    lda_topic_modeling(corpus, num_topics=lda_topics*topic_modelling_multi3, passes=lda_passes, terms_per_topic=lda_terms_per_topic, per_word_topics=lda_per_word_topics)
    logger(f"LDA completed: {lda_topics*topic_modelling_multi3} topics")   
     
###########################################################################
# 3. Topic Modelling with NMF
nmf_topics = config.getint('topic_nmf', 'nmf_topics', fallback=10)
nmf_terms_per_topic = config.getint('topic_nmf', 'nmf_terms_per_topic', fallback=10)
nmf_max_iter = config.getint('topic_nmf', 'nmf_max_iter', fallback=500)

def nmf_topic_modeling(corpus, num_topics, max_iter, terms_per_topic):
    vectorizer = TfidfVectorizer(lowercase=True)
    doc_term_matrix = vectorizer.fit_transform(corpus)
    nmf = NMF(n_components=num_topics, random_state=42, max_iter=max_iter)
    nmf.fit(doc_term_matrix)

    topic_word_matrix = nmf.components_
    document_topic_matrix = nmf.transform(doc_term_matrix)
    terms = vectorizer.get_feature_names_out()
    document_topics = document_topic_matrix.argmax(axis=1)
    topic_doc_counts = {f"Topic {i + 1}": sum(document_topics == i) for i in range(num_topics)}

    with open(os.path.join(directory_analysis, f"analysis_topicmodel_NMF_{num_topics}.txt"), "w", encoding='utf-8') as f:
        for topic_idx, topic in enumerate(topic_word_matrix):
            f.write(f"Topic {topic_idx + 1} ({topic_doc_counts[f'Topic {topic_idx + 1}']} docs): ")
            f.write(", ".join([f"{terms[i]} ({topic[i]:.4f})" for i in topic.argsort()[:-terms_per_topic - 1:-1]]) + "\n")
    
    with open(os.path.join(directory_analysis, f"analysis_topicmodel_nmf_{num_topics}_doc2topic_assignments.txt"), "w", encoding='utf-8') as f:
        for idx, topic in enumerate(document_topics):
            # Split the file name at the underscore and take the first part as the ID
            #f.write(f"Document {file_names[idx]}: Assigned Topic {topic + 1}\n")
            file_id = file_names[idx].split('__')[0]
            f.write(f"{file_id}, {topic + 1}\n")


    return document_topics

# Run the NMF topic modeling
nmf_topic_modeling(corpus, num_topics=nmf_topics, max_iter=nmf_max_iter, terms_per_topic=nmf_terms_per_topic)
logger(f"NMF completed: {nmf_topics} topics")
if topic_modelling_multi1 > 0:
    nmf_topic_modeling(corpus, num_topics=nmf_topics*topic_modelling_multi1, max_iter=nmf_max_iter, terms_per_topic=nmf_terms_per_topic)
    logger(f"NMF completed: {nmf_topics*topic_modelling_multi1} topics")
if topic_modelling_multi2 > 0:
    nmf_topic_modeling(corpus, num_topics=nmf_topics*topic_modelling_multi2, max_iter=nmf_max_iter, terms_per_topic=nmf_terms_per_topic)
    logger(f"NMF completed: {nmf_topics*topic_modelling_multi2} topics")    
if topic_modelling_multi3 > 0:
    nmf_topic_modeling(corpus, num_topics=nmf_topics*topic_modelling_multi3, max_iter=nmf_max_iter, terms_per_topic=nmf_terms_per_topic)
    logger(f"NMF completed: {nmf_topics*topic_modelling_multi3} topics")      


###########################################################################
# 4. Topic Modelling with HDP
terms_per_topic_hdp = config.getint('topic_hdp', 'terms_per_topic_hdp', fallback=10)

def hdp_topic_modeling(corpus, terms_per_topic):
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(corpus)

    id2word = dict(enumerate(vectorizer.get_feature_names_out()))
    bow_corpus = matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)

    # Create HDP model
    hdp = HdpModel(corpus=bow_corpus, id2word=id2word)
    hdp_results = hdp.show_topics(num_words=terms_per_topic, formatted=False)
    num_topics_generated = len(hdp_results)  # Number of topics determined by HDP

    # Initialise topic document counts
    topic_doc_counts = {}
    document_topics = []

    # Assign dominant topic to each document
    for doc in bow_corpus:
        topic_probs = hdp[doc]  # Get topic distribution for the document

        if topic_probs:  # Check if topic_probs is not empty
            assigned_topic = max(topic_probs, key=lambda x: x[1])[0]  # Find the dominant topic
        else:
            assigned_topic = -1  # Or a default topic to indicate no topic was assigned

        document_topics.append(assigned_topic)

        # Update topic_doc_counts dynamically
        topic_key = f"Topic {assigned_topic + 1}" if assigned_topic != -1 else "No Topic"
        if topic_key in topic_doc_counts:
            topic_doc_counts[topic_key] += 1
        else:
            topic_doc_counts[topic_key] = 1

    # Save HDP topic modelling results
    with open(os.path.join(directory_analysis, "analysis_topicmodel_HDP.txt"), "w", encoding='utf-8') as f:
        for idx, topic in enumerate(hdp_results):
            topic_key = f"Topic {idx + 1}"
            doc_count = topic_doc_counts.get(topic_key, 0)
            f.write(f"{topic_key} ({doc_count} docs): ")
            
            # Extract word-probability pairs correctly based on `topic` structure
            topic_words = topic[1]  # `topic[1]` should be a list of (word, probability) pairs
            f.write(", ".join([f"{word} ({weight:.4f})" for word, weight in topic_words]) + "\n")

    # Save document-topic assignments
    with open(os.path.join(directory_analysis, "analysis_topicmodel_HDP_doc2topic_assignments.txt"), "w", encoding='utf-8') as f:
        for idx, topic in enumerate(document_topics):
            file_id = file_names[idx].split('__')[0]
            f.write(f"{file_id}, {topic + 1 if topic != -1 else 'No Topic'}\n")

    return num_topics_generated

# Run the HDP topic modeling
num_topics_generated = hdp_topic_modeling(corpus, 10)
logger(f"HDP completed: {num_topics_generated} topics")

logger("END")