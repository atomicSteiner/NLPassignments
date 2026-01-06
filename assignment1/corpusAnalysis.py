import sys
import nltk
from nltk.corpus import gutenberg, stopwords
from nltk import FreqDist

def analyze_corpus(text_file):
    nltk.download('gutenberg')
    nltk.download('stopwords')

    # Get all tokens from the selected Gutenberg text
    tokens = gutenberg.words(text_file)
    num_tokens = len(tokens)

    # Get all unique word types
    types = set(tokens)
    num_types = len(types)

    # Remove stopwords from types
    types_without_stoppingwords = set([t for t in types if t.lower() not in stopwords.words('english')])
    num_types_without_stoppingwords = len(types_without_stoppingwords)

    # Frequency distribution of tokens
    fdist = FreqDist(tokens)
    most_common = fdist.most_common(10)

    # Find types longer than 13 characters
    long_types = [t for t in types if len(t) > 13]
    # Find types ending with 'ation'
    nouns_ending_in_ation = [t for t in types if t.lower().endswith('ation')]

    # Return all results in a dictionary
    return {
        'num_tokens': num_tokens,
        'num_types': num_types,
        'num_types_without_stoppingwords': num_types_without_stoppingwords,
        'most_common': most_common,
        'long_types': long_types,
        'nouns_ending_in_ation': nouns_ending_in_ation
    }
# Get the filename from command line, or use default
if(len(sys.argv) > 1):
    file_text = sys.argv[1]
else:
    file_text = 'austen-emma.txt'

# Analyze the corpus and print results
result = analyze_corpus(file_text)
print(f"Text: {file_text}")
print(f"Tokens: {result['num_tokens']}")
print(f"Types: {result['num_types']}")
print(f"Types excluding stopping words: {result['num_types_without_stoppingwords']}")
print(f"10 most common tokens: {result['most_common']}")
print(f"Long types: {result['long_types']}")
print(f"Nouns ending in 'ation': {result['nouns_ending_in_ation']}")