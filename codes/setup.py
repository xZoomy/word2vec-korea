import os
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install('gensim')
os.mkdir('../input')
os.mkdir('../model')
os.mkdir('../databases')
