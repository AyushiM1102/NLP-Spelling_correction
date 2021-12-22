# nlp-ml

This is a spelling correction project that takes in text documents with identified mispelled words (with one letter in each corrupted as · [ASCII 183]) and predict the true words. The sample text can be found in text files ending with `.dot` and some of their uncorrupted version in the files with same prefix ending with `.txt`.

## Models

1. Character-based Naive Bayes (pattern matching, built from scratch)

2. Character-based Naive Bayes + N-gram Context => JamSpell (reference: https://github.com/bakwc/JamSpell#python)

3. Feed Forward Neural Network with Character Embeddings (built from scratch)

## Code

Code for all 3 models can be found in `ML 349_Project_NB+FFNN.ipynb`.

## Report

A comprehensive report in slides can be found in `Spelling correction- final presentation.pdf`. It includes the description of the task, datasets, successful methods metioned above and some failed attempts due to the limit of scope of this project.
