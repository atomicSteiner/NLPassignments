import nltk
from nltk import FreqDist
from nltk.corpus.reader import TaggedCorpusReader as tcr

# Get sentences, words, tagged sentences and tagged words from the corpus
reader = tcr(".", r".*\.sent", sep="/")
sents = reader.sents()
words = reader.words()
tagged_sents = reader.tagged_sents()
tagged_words = reader.tagged_words()

def print_basic_stats(sents, words, tagged_sents):
	# Print basic statistics about the corpus
	print(f"Number of sentences: {len(sents)}")
	print("Sentence no. 100:")
	# Print the 100th sentence (words only)
	sentence_100 = [word for word, tag in tagged_sents[99]]
	print(' '.join(sentence_100))
	print(f"Number of tokens: {len(words)}")
	# Print the number of unique word types
	types = set(words)
	print(f"Number of types: {len(types)}")

def print_freq_tokens(words):
	# Print the 10 most frequent tokens in the corpus
	fdist = FreqDist(words)
	print("The 10 most frequent tokens")
	for token, freq in fdist.most_common(10):
		print(f"{token} => {freq}")

def print_freq_tags(tagged_words):
	# Print the 20 most frequent part-of-speech tags
	fdist_tags = FreqDist([tag for word, tag in tagged_words])
	print("The 20 most frequent PoS tags")
	for tag, freq in fdist_tags.most_common(20):
		print(f"{tag} => {freq}")

def print_tags_after_af(tagged_words):
	# Find PoS tags that follow the tag 'AF' and print the 10 most frequent
	tags_after_af = [tagged_words[i+1][1] for i in range(len(tagged_words)-1) if tagged_words[i][1]=='AF']
	fdist_af = FreqDist(tags_after_af)
	print("The 10 most frequent PoS tags following the tag 'af'")
	for tag, freq in fdist_af.most_common(10):
		print(f"{tag} => {freq}")

print_basic_stats(sents, words, tagged_sents)
print_freq_tokens(words)
print_freq_tags(tagged_words)
print_tags_after_af(tagged_words)
