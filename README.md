# Korea-git

## How to use

Create `/input`, `/model`.\
Put `kernel.sqlite` in `/databases` folder and `kernel.txt` in `/input`.\
Run `python createASM.py` in console in `/codes` folder.\
Run `python inputModifier.py`.\
Run `python gensim_word2vec.py`.\

## Files

`createASM.py` for creating `assembly.asm`
`inputModifier.py` for creating `test.asm`, a modified version of `assembly.asm`
`gensim_word2vec.py` for neural network

## Dependencies

[gensim](https://radimrehurek.com/gensim/) (`pip install --upgrade gensim`)

## Documentation

SQLite Shell Command Doc [here](https://sqlite.org/cli.html).\
SQLite for Python Doc [here](https://docs.python.org/fr/3.6/library/sqlite3.html).\
Markdown Doc [here](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).\
Gensim Word2Vec Doc [here](https://radimrehurek.com/gensim/models/word2vec.html).
