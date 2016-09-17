import nltk
sentence = "At eight o'clock on Thursday morning on Thursday morning on Thursday morning."
tokens = nltk.word_tokenize(sentence)
bigram_tuples = list(nltk.bigrams(tokens))
trigram_tuples = list(nltk.trigrams(tokens))

count = {item : bigram_tuples.count(item) for item in set(bigram_tuples)}
ngrams = [item for item in set(bigram_tuples) if "on" in item]
default_tagger = nltk.DefaultTagger('NN')
tagged_sentence = default_tagger.tag(tokens)

# Show the description of the tag 'NN'
nltk.help.upenn_tagset('NN')
patterns = [(r'.*ing$', 'VBG'),(r'.*ed$', 'VBD'),(r'.*es$', 'VBZ'),(r'.*ed$', 'VB')]
regexp_tagger = nltk.RegexpTagger(patterns)
out = regexp_tagger.tag(tokens)
# print out
