# Korea-git

## How to use

Run `setup.py`\
Put `kernel.sqlite` (1GB file) in `/databases` folder.\
Run `python createASM.py` in console in `/codes` folder.\
Run `python inputModifier.py`.\
Run `python gensim_word2vec.py` to play with the neural network.\

## Files

`setup.py` for install dependencies and creating right environment.\
`createASM.py` for creating `assembly.asm`\
`inputModifier.py` for creating `test.asm`, a modified version of `assembly.asm`\
`gensim_word2vec.py` for neural network

## Dependencies

• Python [3.6.8](https://www.python.org/downloads/release/python-368/)\
• [gensim](https://radimrehurek.com/gensim/)

## Documentation

SQLite Shell Command Doc [here](https://sqlite.org/cli.html).\
SQLite for Python Doc [here](https://docs.python.org/fr/3.6/library/sqlite3.html).\
Markdown Doc [here](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).\
Gensim Word2Vec Doc [here](https://radimrehurek.com/gensim/models/word2vec.html).
