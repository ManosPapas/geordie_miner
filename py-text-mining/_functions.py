import os
import shutil
import configparser
import sys
import re
from nltk.stem import WordNetLemmatizer
from datetime import datetime

# Declare global variables
language = None
directory_analysis = None
directory_data = None
directory_text = None
directory_processed = None

def load_config(debug=False):
    global language, directory_analysis, directory_data, directory_text, directory_processed

    # Use command-line argument if provided, otherwise default to 'config.txt'
    filename = sys.argv[1] if len(sys.argv) > 1 else 'config.txt'
    directory_data = sys.argv[2] if len(sys.argv) > 2 else 'data_pdf'
    
    # Get the full path of the config file
    full_path = os.path.abspath(filename)
    
    # Check if the configuration file exists
    if not os.path.exists(full_path):
        print(f"Error: Configuration file '{full_path}' not found. Please ensure it is in the correct directory.")
        sys.exit(1)  # Exit the script with a non-zero status to indicate an error

    # Read configuration settings from the file
    config = configparser.ConfigParser()

    # Attempt to read the configuration file
    try:
        config.read(full_path)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)  # Exit on any read errors

    # Check if any sections are found in the config
    if not config.sections():
        print("Error: No sections found in the configuration file.")
        sys.exit(1)  # Exit if no sections are found

    # language
    language = config.get('default', 'language', fallback='english')
    
    
    # Check if the configuration file exists
    if not os.path.exists(directory_data):
        print(f"Error: Data directory '{directory_data}' not found. ")
        sys.exit(1)  # Exit the script with a non-zero status to indicate an error
    
    # Setup directories for analysis 
    directory_analysis = "analyis_"+directory_data
    directory_text = os.path.join(directory_analysis, 'text')
    directory_processed = os.path.join(directory_analysis, 'text_processed')

    # Delete and recreate the directory
    def delete_and_recreate_directory(directory_path):
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)  # Recursively delete the directory and its contents
        os.makedirs(directory_path)
    
    delete_and_recreate_directory(directory_analysis)
    delete_and_recreate_directory(directory_text)
    delete_and_recreate_directory(directory_processed)
    
    
    # Print loaded values if debug is True
    if debug:
        print("Configuration settings loaded:")
        for section in config.sections():
            print(f"\n[{section}]")
            for key, value in config.items(section):
                print(f"{key} = {value}")

        # Print the setup directory paths
        print(f"\nSetup directories:")
        print(f"Language: {language}")
        print(f"Directory Export: {directory_analysis}")
        print(f"Directory Data: {directory_data}")
        print(f"Directory Text: {directory_text}")
        print(f"Directory Processed: {directory_processed}")        
        print(f"")

    # Save the current data and config variables to config.log
    save_config_log(config)

    return config

def save_config_log(config):
    """Save the current configuration variables to _config.log in the export directory."""
    log_file_path = os.path.join(directory_analysis, '_log_config.log')
    
    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format the date and time

    try:
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Log created on: {current_time}\n")  # Write date and time at the top
            log_file.write("\nConfiguration Settings:\n\n")
            log_file.write(f"Language: {language}\n")            
            log_file.write(f"Directory Export: {directory_analysis}\n")
            log_file.write(f"Directory Data: {directory_data}\n")
            log_file.write(f"Directory Text: {directory_text}\n")
            log_file.write(f"Directory Processed: {directory_processed}\n")
            log_file.write("\n\n")
            log_file.write("Configuration from file:\n\n")

            # Write all sections and their key-value pairs
            for section in config.sections():
                log_file.write(f"[{section}]\n")
                for key, value in config.items(section):
                    log_file.write(f"{key} = {value}\n")
                log_file.write("\n")

    except Exception as e:
        print(f"Error writing to log file: {e}")

# Function to preprocess the text
lemmatizer = WordNetLemmatizer()

def process_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
    tokens = text.split()  # Tokenise
    lemmatised_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatised_tokens

def descriptive_stats(dir):
    # Read files and create a new corpus
    corpus = []
    for filename in os.listdir(dir):
        if filename.endswith('.txt'):
            with open(os.path.join(dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                processed_text = process_text(text)
                corpus.append(processed_text)
    
    # Descriptive statistics about the corpus
    total_docs = len(corpus)
    total_words = sum(len(doc) for doc in corpus)
    all_terms = [term for doc in corpus for term in doc]
    unique_terms = set(all_terms)
    
    # Calculate terms per document, min/max words in a document
    terms_per_doc = [len(doc) for doc in corpus]
    min_terms = min(terms_per_doc) if terms_per_doc else 0
    max_terms = max(terms_per_doc) if terms_per_doc else 0
    
    # Save descriptive statistics
    with open(os.path.join(directory_analysis, "analysis_corpus.txt"), "a", encoding='utf-8') as f:
        f.write(f"\n")
        f.write(f"Directory: {dir}\n")
        f.write(f"Number of documents: {total_docs}\n")
        f.write(f"Total words in corpus: {total_words}\n")
        f.write(f"Total unique terms: {len(unique_terms)}\n")
        f.write(f"Number of terms per document: {round(sum(terms_per_doc) / total_docs) if total_docs else 0}\n")
        f.write(f"Minimum terms in a document: {min_terms}\n")
        f.write(f"Maximum terms in a document: {max_terms}\n")
