import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    # collect all tokens
    uni_tup = list()
    bigram_tuples = list()
    trigram_tuples = list()
    total_lines = training_corpus.__len__()
    for line in training_corpus:
        t_dummy = list()
        t_dummy.append(START_SYMBOL)
        t_dummy.extend(line.split())
        t_dummy.append(STOP_SYMBOL)
        uni_tup.extend(line.split())
        uni_tup.append(STOP_SYMBOL)
        bigram_tuples.extend(list(nltk.bigrams(t_dummy)))
        trigram_tuples.extend(list(nltk.trigrams(t_dummy)))
    total_uni_size = uni_tup.__len__()
    # total_bi_size = bigram_tuples.__len__()
    # total_tri_size = trigram_tuples.__len__()

    # calculate probability
    # TODO: Optimize Count using Dictionary
    unigram_p = {item : 0.0 for item in set(uni_tup)}
    unigram_count = {item : 0.0 for item in set(uni_tup)}
    bigram_p = {item : 0.0 for item in set(bigram_tuples)}
    bigram_count = {item: 0.0 for item in set(bigram_tuples)}
    trigram_p = {item: 0.0 for item in set(trigram_tuples)}
    trigram_count = {item: 0.0 for item in set(trigram_tuples)}

    for item in uni_tup:
        unigram_count[item]+=1.0
    for item in bigram_tuples:
        bigram_count[item]+=1.0
    for item in trigram_tuples:
        trigram_count[item]+=1.0
    for item in unigram_p.keys():
        unigram_p[item] = math.log(unigram_count[item], 2.0)-math.log(total_uni_size, 2.0)
    for item in bigram_p.keys():
        if item[0] == START_SYMBOL:
            bigram_p[item] = math.log(bigram_count[item], 2.0) - math.log(total_lines, 2.0)
        else:
            bigram_p[item] = math.log(bigram_count[item], 2.0)-math.log(unigram_count[item[0]], 2.0)
    for item in trigram_p.keys():
        trigram_p[item] = math.log(trigram_count[item], 2.0) - math.log(bigram_count[(item[0], item[1])], 2.0)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    def ngram_tuple(sentence):
        log_prob = 0.0
        tokens = list()
        tokens.append(START_SYMBOL)
        tokens.extend(sentence.split())
        tokens.append(STOP_SYMBOL)
        if n==3:
            all_ngrams = list(nltk.trigrams(tokens))
        elif n==2:
            all_ngrams = list(nltk.bigrams(tokens))
        else:
            all_ngrams = sentence.split()
        for one in all_ngrams:
            if one in ngram_p:
                log_prob += ngram_p[one]
            else:
                log_prob = -1000
                break

        return log_prob

    scores = [ngram_tuple(sent) for sent in corpus]
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    def cal_inter_prob(tokens_of_3):
        log_value = 0.0
        for item in tokens_of_3:
            if item in trigrams:
                tri = math.pow(2.0,trigrams[item])
            else:
                tri = 0.0
            if (item[1],item[2]) in bigrams:
                bi = math.pow(2.0, bigrams[(item[1], item[2])])
            else:
                bi = 0.0
            if item in trigrams:
                uni = math.pow(2.0, unigrams[item[2]])
            else:
                uni = 0.0
            tot = uni+bi+tri
            if tot==0.0:
                log_value = -1000
                break
            else:
                log_value += math.log(1.0/3.0, 2.0)+math.log(uni+bi+tri, 2.0)
        return log_value
    scores = []
    for line in corpus:
        tokens = list(START_SYMBOL)
        tokens.extend(line.split())
        tokens.append(STOP_SYMBOL)
        # log2(1/3) + log2(puni+pbi+ptri)
        tri_tokens = list(nltk.trigrams(tokens))
        scores.append(cal_inter_prob(tri_tokens))

    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close()

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
