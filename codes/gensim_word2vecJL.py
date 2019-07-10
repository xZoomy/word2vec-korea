import gensim
from gensim.models import word2vec
import logging
from keras.layers import Input, Embedding, merge
from keras.models import Model
from gensim.similarities import WmdSimilarity
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import pyparsing
import tensorflow as tf
import numpy as np
import re
import urllib.request
import os
import zipfile
import operator
from gensim import utils, matutils

MAX_WORDS_IN_BATCH = 10000




class Text8Corpus2(object):
    """Iterate over sentences from the "assembly.asm" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""
    def __init__(self, fname, max_sentence_length=MAX_WORDS_IN_BATCH):
        self.fname = fname
        self.max_sentence_length = max_sentence_length

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest = [], b''
        with utils.smart_open(self.fname) as fin:
            while True:

                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    words = utils.to_unicode(text).split()

                    sentence.extend(words)  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                last_token = text.rfind(b' ')  # last token may have been split in two... keep for next iteration


                words, rest = (utils.to_unicode(text[:last_token]).split(),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                # words, rest = (utils.to_unicode(text[:last_token]).split(),
                #               text[last_token:].strip()) if last_token >= 0 else ([], text)
                #print(words)
                sentence.extend(words)

                sentence = list(filter(None, sentence))  #delete empty string
                sentence = list(filter(operator.methodcaller('strip'), sentence)) #delete white spaces string
                print("Sentence : ")
                print(sentence[:5])
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]



vector_dim = 300
root_path = "./"
print(root_path)
def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

# convert the input data into a list of integer indexes aligning with the wv indexes
# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        #print(word)
        if word.decode() in wv:
            index_data.append(wv.vocab[word.decode()].index)
    return index_data
    
def gensim_demo():

    #filename = "main.c"
    filename="assembly.asm"
    #filename="textEssai.txt"
    #filename="RSTSP.py"
   # print("File name : " +filename)
    #sentences = word2vec.Text8Corpus((root_path + filename).strip('.zip'))
    sentences=Text8Corpus2(root_path+filename)
   # print(sentences[0])
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #Iter :  number of interation for training dataset
    #size 200 : 200 for asm file training dataset
    model = word2vec.Word2Vec(sentences, iter=1, min_count=1, size=200, workers=4)


    # get the most common libe
    print("Line commun")
    print(model.wv.index2word[1])

    print(model.wv.index2word[2])

    #print("vector")
    #print(model.wv[model.wv.index2word[1]])

    # get the least common words
    vocab_size = len(model.wv.vocab)
    print("Line pas commun")
    print(model.wv.index2word[vocab_size - 1])
    print(vocab_size)

    # some similarity fun
    print("Similarity")
    print(model.wv.index2word[3],model.wv.index2word[5])
    print(model.wv.similarity(model.wv.index2word[3], model.wv.index2word[5]))

    # what doesn't fit?
   # print("doesn't fit")

    #wv_from_text = KeyedVectors.load_word2vec_format(datapath(root_path+'RobotNettoyeur.cpp'), binary=False)  # C text format



    #str_data = read_data2(root_path + filename)
    #index_data = convert_data_to_index(str_data, model.wv)
    #print("Data and index")
    #print(str_data[:4], index_data[:4])

    # save and reload the model
    model.save(root_path + "asmModel")


def create_embedding_matrix(model):
    # convert the wv word vectors into a numpy matrix that is suitable for insertion
    # into our TensorFlow and Keras models
    embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
    for i in range(len(model.wv.vocab)):
        embedding_vector = model.wv[model.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def tf_model(embedding_matrix, wv):
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # embedding layer weights are frozen to avoid updating embeddings while training
    saved_embeddings = tf.constant(embedding_matrix)
    embedding = tf.Variable(initial_value=saved_embeddings, trainable=False)

    # create the cosine similarity operations
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # call our similarity operation
        sim = similarity.eval()
        # run through each valid example, finding closest words
        for i in range(valid_size):
            valid_word = wv.index2word[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = wv.index2word[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)


def keras_model(embedding_matrix, wv):
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # input words - in this case we do sample by sample evaluations of the similarity
    valid_word = Input((1,), dtype='int32')
    other_word = Input((1,), dtype='int32')
    # setup the embedding layer
    embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix])
    embedded_a = embeddings(valid_word)
    embedded_b = embeddings(other_word)
    similarity = merge([embedded_a, embedded_b], mode='cos', dot_axes=2)
    # create the Keras model
    k_model = Model(input=[valid_word, other_word], output=similarity)

    def get_sim(valid_word_idx, vocab_size):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = k_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

    # now run the model and get the closest words to the valid examples
    for i in range(valid_size):
        valid_word = wv.index2word[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        sim = get_sim(valid_examples[i], len(wv.vocab))
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = wv.index2word[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

if __name__ == "__main__":
    run_opt = 1
    if run_opt == 1:
        gensim_demo()
    elif run_opt == 2:
        model = gensim.models.Word2Vec.load(root_path + "mymodel")
        embedding_matrix = create_embedding_matrix(model)
        tf_model(embedding_matrix, model.wv)
    elif run_opt == 3:
        model = gensim.models.Word2Vec.load(root_path + "mymodel")
        embedding_matrix = create_embedding_matrix(model)
        keras_model(embedding_matrix, model.wv)
