#penntreebank
import nltk
from nltk.corpus import treebank
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('treebank')

taggers = [nltk.AffixTagger, nltk.UnigramTagger, nltk.BigramTagger, nltk.TrigramTagger]
tagged_sentences = treebank.tagged_sents()
training_set = tagged_sentences[:3500]
test_set = tagged_sentences[3500:]

def main():
    #1.1 print out the total count of each set, and print the first sentence in the test set
    print("Training set size:", len(training_set))
    print("Test set size:", len(test_set))
    print("First sentence in test set:", test_set[0])

    #1.2 Construct four taggers trained on the training set: an instance of an AffixTagger, UnigramTagger, BigramTagger and a TrigramTagger (without any “backoff” model). Evaluate them on the test set, and print out
    #the evaluation results.
    print("Tagging accuracies:\n"
    "-------------------")
    for tagger_class in taggers:
        tagger = tagger_class(training_set)
        accuracy = tagger.accuracy(test_set)
        print(f"{tagger_class.__name__} accuracy: {accuracy*100:.2f} %")

    #1.3 Construct the latter three taggers again, but now with a backoff model, i.e. such that the trigram tagger uses a bigram tagger as backoff, which in turn uses a unigram tagger as backoff, which in turn uses the affixtagger as backoff. Print the evaluation results again.
    print("\nTagging accuracies with backoff:\n"
    "-------------------")
    tagger = None
    for tagger_class in taggers[0:]:
        tagger = tagger_class(training_set, backoff=tagger)
        accuracy = tagger.accuracy(test_set)
        print(f"{tagger_class.__name__} with backoff accuracy: {accuracy*100:.2f} %")

    #1.4 Explain why this is the case. In particular explain this for the case of the BigramTagger . Just add this as a comment in your .py file.
    """The individual taggers without backoff models have significantly lower accuracy because 
    they fail to assign any tag when they encounter unseen contexts, resulting in None tags.
    
    For BigramTagger specifically:
    - A BigramTagger looks at the previous word's tag and the current word to decide the POS tag
    - Without backoff: If the bigram (previous_tag, current_word) was never seen in training,
      the tagger returns None instead of making any prediction
    - This leads to many untagged words, dramatically reducing accuracy
    
    With backoff models:
    - BigramTagger first tries to use bigram context (previous_tag, current_word)
    - If that fails, it backs off to UnigramTagger (just looks at current_word)
    - If that fails, it backs off to AffixTagger (looks at word endings like -ing, -ed)
    - If that fails, it backs off to DefaultTagger (assigns most common tag)
    
    This cascading approach ensures every word gets tagged with the most specific 
    information available, rather than leaving words untagged. The backoff chain 
    gracefully degrades from specific context (bigrams) to general patterns (affixes)
    to reasonable defaults, maintaining high coverage while preserving accuracy 
    where specific information is available.
    """
    #1.5 Tag the test set with the default ("off-the-shelf") tagger in the NLTK, evaluate its accuracy and print out the result.
    sentences_to_tag = [[word for word, tag in sentence] for sentence in test_set]
    default_tagger_results = nltk.pos_tag_sents(sentences_to_tag)
    correct_tags = 0
    total_tags = 0
    for i, sentence in enumerate(default_tagger_results):
        for j, (word, predicted_tag) in enumerate(sentence):
            actual_tag = test_set[i][j][1]
            if predicted_tag == actual_tag:
                correct_tags += 1
            total_tags += 1
    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    print(f"Accuracy of the default tagger in NLTK: {accuracy*100:.2f} %")

main()
