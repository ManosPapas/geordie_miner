1. Initialization
- Configuration Loading: Loads settings and initializes directories using _functions.load_config().
- Logging: A logger function is defined to log messages with timestamps to both the console and a log file.
- NLTK Setup: Ensures required NLTK resources (e.g., stopwords, punkt) are downloaded and available.

2. PDF to Text Conversion
- Directory Management: Ensures the required directories (data, text, processed) exist, creating or clearing them as needed.
- File Processing:
- Text files are copied directly.
- PDF files are converted to text using PyPDF2.

3. Preprocessing
- Stopwords: Combines custom and NLTK stopwords, logs them, and removes them from the text.
- Substitutions: Applies text substitutions based on a configuration file.
- Low-Frequency Terms: Removes terms that appear less frequently than a specified threshold.
- Text Cleaning: Removes unwanted characters, links, and single-character terms.

4. Corpus Descriptive Statistics
- Calls _functions.descriptive_stats() to generate descriptive statistics for the processed text.

5. Term Analysis
- Lemmatization: Converts words to their base forms using WordNetLemmatizer.
- Frequency Analysis:
- Raw and lemmatized term frequencies are calculated.
- TF-IDF values are computed for terms.
- Output: Saves term analysis results to Excel files.

6. Word Cloud Generation
- Generates word clouds for raw and lemmatized terms using WordCloud.

7. Phrase Analysis
- N-Grams: Extracts and logs frequent bigrams and trigrams.
- Co-Occurrence Analysis: Analyzes term co-occurrence within a specified window size and exports results.
- Network Analysis: Builds a co-occurrence network and exports it in GEXF format for visualization.

8. Topic Modeling
- KMeans Clustering: Groups documents into clusters based on term frequency.
- LDA (Latent Dirichlet Allocation): Identifies topics in the corpus and assigns documents to topics.
- NMF (Non-Negative Matrix Factorization): Another method for topic modeling.
- HDP (Hierarchical Dirichlet Process): Dynamically determines the number of topics and assigns documents.

9. Hierarchical Clustering
- Performs agglomerative clustering using Jaccard distance and visualizes the results as a dendrogram.

10. Logging and Output
- Logs the progress and results of each step.
- Saves outputs (e.g., term frequencies, topic assignments, word clouds) to the analysis directory