import gensim
from gensim.models import word2vec
import logging

from keras.layers import Input, Embedding, merge
from keras.models import Model

import tensorflow as tf
import numpy as np

import urllib.request
import os
import zipfile

vector_dim = 300
root_path = "H:\Desktop\Cours\Stages\ING2\\ML\\tuto jeanluc\\" #input_assembly\\"

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
        if word.decode() in wv:
            index_data.append(wv.vocab[word.decode()].index)
    return index_data

def gensim_demo(nb_iters, name_model='mymodel', name_input='text8', nb_min=10):
    # url = 'http://mattmahoney.net/dc/'
    # filename = maybe_download('text8.zip', url, 31344016)
    filename = name_input
    print("Training on {} file ({} iterations)".format(filename, nb_iters))
    # if not os.path.exists((root_path + filename).strip('.zip')):
    #      zipfile.ZipFile(root_path+filename).extractall()
    sentences = word2vec.Text8Corpus((root_path + filename)) #.strip('.zip'))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, iter=nb_iters, min_count=nb_min, size=300, workers=4)
    # iter= nb epochs, min_count=nb mini pr etre ds le voc, size=size of word vector, workers=parallel
    # model = word2vec.Word2Vec(sentences, iter=10, min_count=10, size=300, workers=4) #default

    #print(model.wv['the'])
    vocab_size = len(model.wv.vocab)
    print('Most commond words are: {} {} {}'.format(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2]))
    print('Least commond words are: {} {} {}'.format(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3]))
    # print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'elephant'))
    # print(model.wv.doesnt_match("green blue red zebra".split()))
    # print('Index of "of" is: {}'.format(model.wv.vocab['of'].index))
    print(root_path+filename)
    # str_data = read_data(root_path + filename)
    # index_data = convert_data_to_index(str_data, model.wv)
    # print(str_data[:4], index_data[:4])
    print("Model saved as {}".format(name_model))
    model.save(root_path + name_model) # name ici


def gensim_load(name_model='mymodel'):
    model = word2vec.Word2Vec.load(root_path + name_model)
    vocab_size = len(model.wv.vocab)
    print("The vocabulary contains {} words.".format(vocab_size))
    while True:
        choice = int(input("\n 1 - Find the odd one out\n 2 - Similarity\n 3 - Get vector\n 0 - Back\n"))
        if choice == 1:
            words_input = input("Words ? (put space between them) : ")
            print(model.wv.doesnt_match(words_input.split()))
        elif choice == 2:
            words_input = input("2 words ? ").split()
            w1 = words_input[0]
            w2 = words_input[1]
            print("Link between {} and {} is".format(w1, w2), round(model.wv.similarity(w1, w2), 3))
        elif choice == 3:
            word_input = input("Word ? ")
            if word_input in model.wv.vocab:
                print(model.wv[word_input])
            else:
                print("This word is not in the vocabulary :(")
        elif choice == 0:
            break

if __name__ == "__main__":
    while True:
        print("\n---- MENU ----\n 1 - training model + examples\n 4 - loading model\n 0 - EXIT") #2 - tf model\n 3 - keras model(bugged)\n
        run_opt = int(input())
        if run_opt == 1:
            nb_iters = 1
            name_model = 'mymodel'
            name_input = 'target.c'
            nb_min = 1
            # nb_iters = int(input("Nb iters ? "))
            # name_model = input("Name of model (def='mymodel')")
            # name_input = input("Name of input file (def='text8') ? ")
            gensim_demo(nb_iters, name_model, name_input, nb_min)

        elif run_opt == 4:
            gensim_load()
        elif run_opt == 0:
            exit()
        else:
            print("Error! Please enter a valid option\n")
