import requests
import unicodedata

from nltk import *
from nltk.corpus import *
from bs4 import BeautifulSoup


def parse_html(url):
    request = requests.get(url)
    html = request.content
    extractor = BeautifulSoup(html, features="html.parser")
    [x.extract() for x in extractor.find_all(['script', 'style', '[document]'])]

    text = extractor.get_text()
    text = re.sub('\[[^]]*\]', '', text)
    text = text.replace('\n', '')

    corpus = ''
    for t in text:
        corpus += t

    return corpus


def sentence_tokenize(text):
    sentences = sent_tokenize(text)

    return sentences


def words_tokenize(text):
    tokens = word_tokenize(text)

    return tokens


def normalisation(text):
    sentences = sentence_tokenize(text)
    corpus = ""
    for sentence in sentences:
        tokens = words_tokenize(sentence)

        for token in tokens:

            token = token.lower()
            token = unicodedata.normalize('NFKD', token).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            token = re.sub(r'[^\w\s]', '', token)
            corpus += token + ' '

    return corpus

def stemming_lemmatizating(text):
    sentences = sentence_tokenize(text)
    corpus = ""
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for sentence in sentences:
        tokens = words_tokenize(sentence)

        for token in tokens:

            token = wordnet_lemmatizer.lemmatize(token)
            token = stemmer.stem(token)
    # stop words removal from corpus
            if len(token) > 1 and token not in stopwords.words('english'):
                corpus += token + ' '

    return corpus


def computeTF(word_dictionary, corpus):
    tf_dict = {}
    corpus_count = len(corpus)
    for word, count in word_dictionary.items():
        tf_dict[word] = count / float(corpus_count)

    return tf_dict


def computeIDF(documents):
    import math
    N = len(documents)

    idf_dict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, values in document.items():
            if values > 0:
                idf_dict[word] += 1

    for word, values in idf_dict.items():
        idf_dict[word] = math.log(N/ float(values))

    return idf_dict

def computeTFIDF(corpus, idfs):
    tfidf = {}
    for word, values in corpus.items():
        tfidf[word] = values * idfs[word]

    return tfidf

def main():

    a = 'https://sites.google.com/view/siirh2020/'
    b = 'http://www.multimediaeval.org/mediaeval2019/memorability/'

    raw_textA = parse_html(a)
    raw_textB = parse_html(b)

###
### new file: plain_text contains the html retrieved from webpage links in raw form.
###
    try:
        with open('plain_text.txt', 'a') as f:

            f.write('Link 1: https://sites.google.com/view/siirh2020/ \n')
            f.write(raw_textA)
            f.write('\n')
            f.write('Link 2: http://www.multimediaeval.org/mediaeval2019/memorability/ \n')
            f.write(raw_textB)

        f.close()

    except OSError:
        print('could not write to file')

###
### new file: sentences.txt contains the raw text when split into sentences
###
    try:
        with open('sentences.txt', 'a') as f:

            f.write('Link 1: https://sites.google.com/view/siirh2020/ \n')
            sentences = sentence_tokenize(raw_textA)
            for sentence in sentences:
                f.write(sentence + '\n')

            f.write('\n')
            f.write('Link 2: http://www.multimediaeval.org/mediaeval2019/memorability/ \n')
            sentences = sentence_tokenize(raw_textB)
            for sentence in sentences:
                f.write(sentence + '\n')

        f.close()

    except OSError:
        print('could not write to file')

###
### new file: words.txt contains the raw text when word tokenised
###
    try:
        with open('words.txt', 'a') as f:

            f.write('Link 1: https://sites.google.com/view/siirh2020/ \n')
            sentences = sentence_tokenize(raw_textA)
            for sentence in sentences:
                tokens = words_tokenize(sentence)
                for token in tokens:
                    f.write(token + '\n')

            f.write('\n')
            f.write('Link 2: http://www.multimediaeval.org/mediaeval2019/memorability/ \n')
            sentences = sentence_tokenize(raw_textB)
            for sentence in sentences:
                tokens = words_tokenize(sentence)
                for token in tokens:
                    f.write(token + '\n')

        f.close()

    except OSError:
        print('could not write to file')

###
### new file: normalised_words.txt contains the raw text when sentence tokenised
###
    try:
        with open('normal_words.txt', 'a') as f:

            f.write('Link 1: https://sites.google.com/view/siirh2020/ \n')
            sentences = sentence_tokenize(normalisation(raw_textA))
            for sentence in sentences:
                ntokens = words_tokenize(sentence)
                for token in ntokens:
                    f.write(token + '\n')

            f.write('\n')
            f.write('Link 2: http://www.multimediaeval.org/mediaeval2019/memorability/ \n')
            sentences = sentence_tokenize(normalisation(raw_textB))
            for sentence in sentences:
                ntokens = words_tokenize(sentence)
                for token in ntokens:
                    f.write(token + '\n')

        f.close()

    except OSError:
        print('could not write to file')


### new file: processed_words.txt contains the raw text when stopwords are removed
### and words have been stemmed and lemmatized.
###
    try:
        with open('proccessed_words.txt', 'a') as f:

            f.write('Link 1: https://sites.google.com/view/siirh2020/ \n')
            sentences = sentence_tokenize(stemming_lemmatizating(normalisation(raw_textA)))
            for sentence in sentences:
                lemtokens = words_tokenize(sentence)
                for token in lemtokens:
                    f.write(token + '\n')

            f.write('\n')
            f.write('Link 2: http://www.multimediaeval.org/mediaeval2019/memorability/ \n')
            sentences = sentence_tokenize(stemming_lemmatizating(normalisation(raw_textB)))
            for sentence in sentences:
                lemtokens = words_tokenize(sentence)
                for token in lemtokens:
                    f.write(token + '\n')

        f.close()

    except OSError:
        print('could not write to file')

###
### computation of tf-idf to weight terms when indexing the corpus'
###

    corpusA = stemming_lemmatizating(normalisation(parse_html(a)))

    corpusB = stemming_lemmatizating(normalisation(parse_html(b)))

    unique_words = set(corpusA.split(' ')).union(set(corpusB.split(' ')))

    number_of_wordsA = dict.fromkeys(unique_words, 0)

    for word in corpusA.split(' '):
        number_of_wordsA[word] += 1
    number_of_wordsB = dict.fromkeys(unique_words, 0)
    for word in corpusB.split(' '):
        number_of_wordsB[word] += 1

    tfA = computeTF(number_of_wordsA, corpusA.split(' '))

    tfB = computeTF(number_of_wordsB, corpusB.split(' '))


    idfs = computeIDF([number_of_wordsA, number_of_wordsB])

    weightsA = computeTFIDF(tfA, idfs)
    sorted_weightsA = sorted(weightsA.items(), key=lambda x: x[1], reverse=True)
    weightsB = computeTFIDF(tfB, idfs)
    sorted_weightsB = sorted(weightsB.items(), key=lambda x: x[1], reverse=True)

### new file: results.txt contains the output of use of tf-idf
### to weight terms in both documents
###
    try:
        with open('results.txt', 'a') as f:
            f.write('Link 1: https://sites.google.com/view/siirh2020/ \n')
            f.write('WORD:          WEIGHTING: \n')
            count = 0
            for el in sorted_weightsA:
                f.write(el[0] + ' - ' + str(weightsA[el[0]]) + '\n')
                if count == 25:
                    break
                count += 1

            f.write('Link 2: http://www.multimediaeval.org/mediaeval2019/memorability/ \n')
            f.write('WORD:          WEIGHTING: \n')
            counter = 0
            for el in sorted_weightsB:
                f.write(el[0] + ' - ' + str(weightsB[el[0]]) + '\n')
                if counter == 25:
                    break
                counter += 1

    except OSError:
        print('could not create results file')


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
