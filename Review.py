import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import string
import numpy as np
from nltk.stem.porter import PorterStemmer


def load_doc(filename):
    "Read the text file and return it. Pass the file name as string argument"
    file = open(filename, 'r') # open the file as read only
    text = file.read() # read all text
    file.close() # close the file
    return text


filename = 'imdb_review/pos/cv000_29590.txt'
text = load_doc(filename)
text

doc = text[:]

tokens = doc.split()
print(tokens)

string.punctuation


table = str.maketrans('', '', string.punctuation)

[w.translate(table) for w in ['remove@','<punctuations>', 'from..', 'tokens!']]

tokens = [w.translate(table) for w in tokens]
print(tokens)

'123'.isalpha()


tokens = [word for word in tokens if word.isalpha()]
print(tokens)

stop_words = set(stopwords.words('english'))
print('Stop Words :%s \n' % stop_words)


tokens = [w for w in tokens if not w in stop_words]
print(tokens)

ps = PorterStemmer()
tokens = [ps.stem(word) for word in tokens]
print(tokens)


tokens = [word for word in tokens if len(word) > 1]
print(tokens)


def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

from collections import Counter
from os import listdir

print(listdir('imdb_review/pos'))

def process_doc(directory, vocab):
    for filename in listdir(directory):
        if filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)


vocab = Counter()
process_doc('imdb_review/pos', vocab)
process_doc('imdb_review/neg', vocab)
print(len(vocab))
print(vocab.most_common(50))

min_occurane = 2
tokens = [k for k,c in vocab.items() if c >= min_occurane]
print(len(tokens))
print(tokens)

def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


save_list(tokens, 'vocab1.txt')

def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)

def process_docs(directory, vocab, is_train=True):
    lines = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines

vocab_filename = 'vocab1.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

positive_lines = process_docs('imdb_review/pos', vocab)
negative_lines = process_docs('imdb_review/neg', vocab)
print(len(positive_lines), len(negative_lines))

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
docs_train = positive_lines + negative_lines
tokenizer.fit_on_texts(docs_train)

Xtrain = tokenizer.texts_to_matrix(docs_train, mode='binary')
print(Xtrain.shape)
print(Xtrain[:5])

positive_lines_test = process_docs('imdb_review/pos', vocab, False)
negative_lines_test = process_docs('imdb_review/neg', vocab, False)
docs_test = positive_lines_test + negative_lines_test 
Xtest = tokenizer.texts_to_matrix(docs_test, mode='binary')
print(Xtest.shape)

ytrain = np.array([1 for _ in range(900)] + [0 for _ in range(900)])
ytest = np.array([1 for _ in range(100)] + [0 for _ in range(100)])

print(len(ytrain))
print(len(ytest))

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(Xtrain, ytrain)

y_pred = naive_bayes_classifier.predict(Xtest)

cm = confusion_matrix(ytest, y_pred)
print(cm)

print('Test Accuracy: %.2f' % (np.sum(cm.diagonal())/np.sum(cm)))

cm_train = confusion_matrix(ytrain, naive_bayes_classifier.predict(Xtrain))
print('Train Accuracy: %.2f' % (np.sum(cm.diagonal())/np.sum(cm)))

def predict_sentiment(review, vocab, tokenizer, model):
    tokens = clean_doc(review)
    tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    print(line)
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    print(encoded)
    yhat = model.predict(encoded)
    return yhat
