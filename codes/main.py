import gensim
from gensim import utils
from gensim.models import word2vec
import logging
import operator
import re
import urllib.request
import os
import zipfile
from sys import exit  # Use this line for spyder (Ipython)
import numpy as np
from scipy import spatial

vector_dim = 300
# root_path = "./"
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
                regex = r'[\[\]+-:]'  # regex for adding space between caract

                r = re.compile(r'[;,\s]*\s*')  # regex for matching , ; and  space
                # split with regex

                words, rest = (r.split(re.sub(regex, ' \g<0> ', utils.to_unicode(text[:last_token]))),
                               text[last_token:].strip()) if last_token >= 0 else ([], text)
                # words, rest = (utils.to_unicode(text[:last_token]).split(),
                #               text[last_token:].strip()) if last_token >= 0 else ([], text)

                sentence.extend(words)

                sentence = list(filter(None, sentence))  # delete empty string
                sentence = list(filter(operator.methodcaller('strip'), sentence))  # delete white spaces string
                print("Sentence : {}".format(sentence[:5]))
                while len(sentence) >= self.max_sentence_length:
                    yield sentence[:self.max_sentence_length]
                    sentence = sentence[self.max_sentence_length:]


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


def confirm(msg):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Are you sure to {}? [Y/N] ".format(msg)).lower()
    return answer == "y"


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
    # sentences = word2vec.Text8Corpus(filename)
    sentences = Text8Corpus2(filename)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, iter=nb_iters, min_count=nb_min, size=300, workers=4)
    # iter= nb epochs, min_count=nb mini pr etre ds le voc, size=size of word vector, workers=parallel
    # model = word2vec.Word2Vec(sentences, iter=10, min_count=10, size=300, workers=4) #default

    # print(model.wv['the'])
    vocab_size = len(model.wv.vocab)
    print('Most commond words are: {} {} {}'.format(model.wv.index2word[0], model.wv.index2word[1],
                                                    model.wv.index2word[2]))
    print('Least commond words are: {} {} {}'.format(model.wv.index2word[vocab_size - 1],
                                                     model.wv.index2word[vocab_size - 2],
                                                     model.wv.index2word[vocab_size - 3]))
    # print(model.wv.similarity('woman', 'man'), model.wv.similarity('man', 'elephant'))
    # print(model.wv.doesnt_match("green blue red zebra".split()))
    # print('Index of "of" is: {}'.format(model.wv.vocab['of'].index))
    # print(root_path+filename)
    # str_data = read_data(root_path + filename)
    # index_data = convert_data_to_index(str_data, model.wv)
    # print(str_data[:4], index_data[:4])
    print("Similarity rbp rbx : ")
    print(model.wv.similarity('rbp', 'rbx'))
    print(model.wv.index2word[:30])
    print("Model saved as {}".format(name_model))
    model.save("../model/" + name_model)  # name ici


def gensim_load(name_model='mymodel'):
    print("Loading {} model.".format(name_model))
    model = word2vec.Word2Vec.load('../model/' + name_model)
    vocab_size = len(model.wv.vocab)
    print("The vocabulary contains {} words.".format(vocab_size))
    while True:
        choice = int(input("\n 1 - Find the odd one out\n 2 - Similarity\n 3 - Get vector\n 4 - Similarity line \n 0 - Back\n"))
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
        elif choice == 4:
            print("Similarity between sentences : ")
            wordvectors = model.wv  # KeyedVectors Instance gets stored
            i1="mov eax, [ebx]" #instruction 1
            i2="mov [esi+eax], cl" #instruction 2
            r = re.compile(r'[;,\s]*\s*')
            regex = r'[\[\]+-:]'

            s1 = list(filter(None, r.split(re.sub(regex, ' \g<0> ', i1))))
            m1 = np.mean(model[s1], axis=0)
            s2 = list(filter(None, r.split(re.sub(regex, ' \g<0> ', i2))))
            print(s1)
            print(s2)
            m2 = np.mean(model[s2], axis=0)
            result = 1 - spatial.distance.cosine(m1, m2)
            print("The similarity is4 : "+str(result))
        elif choice == 0:
            break


def training_model(nb_iters=10):
    filename = "../input/test.asm"
    A = []
    with open(filename, "r") as myfile:
        s = myfile.read()
    a = s.splitlines()
    r = re.compile(r'[;,\s]*\s*')
    regex = r'[\[\]+-:]'
    print("Training on {} file ({} iterations)".format(filename, nb_iters))
    for i in range(0, len(a)):
        A.append(list(filter(None, r.split(re.sub(regex, ' \g<0> ', a[i])))))
    model = word2vec.Word2Vec(iter=nb_iters, min_count=1, size=300, workers=4)
    model.build_vocab(A)
    model.train(A, total_examples=model.corpus_count, epochs=model.iter)
    model.save("../model/goodmodel")
    # sentence = list(filter(None, sentence))  #delete empty string
    # sentence = list(filter(operator.methodcaller('strip'), sentence)) #delete white spaces string




if __name__ == "__main__":
    while True:
        print(
            "\n---- MENU ----\n 1 - training mymodel + examples\n 2 - loading model\n 3 - training goodmodel\n 0 - EXIT")  # 2 - tf model\n 3 - keras model(bugged)\n
        run_opt = int(input())
        if run_opt == 1:
            if confirm("train the mymodel"):
                # nb_iters = 1
                name_model = 'mymodel'
                # name_input = '../input/assembly.asm'
                name_input = '../input/test.asm'
                nb_min = 1
                nb_iters = int(input("Nb iters ? "))
                # name_model = input("Name of model (def='mymodel')")
                # name_input = input("Name of input file (def='text8') ? ")
                gensim_demo(nb_iters, name_model, name_input, nb_min)
            else:
                print("Yeah i knew that !")
        elif run_opt == 2:
            # name_input = input("Nom du model ? ")
            name_input = "goodmodel"
            gensim_load(name_input)
        elif run_opt == 3:
            if confirm("train the goodmodel"):
                training_model()
            else:
                print("lol")
        elif run_opt == 0:
            exit()
        else:
            print("Error! Please enter a valid option\n")

    # similarity between lines
