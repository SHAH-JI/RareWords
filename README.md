# RareWords

This repository includes our Final Year Project Code for our Bachelor's in Computer Science Degree from FAST-National University of Computer and Emerging Sciences, Islamabad, Pakistan

Translation is an open vocabulary problem which is treated as fixed vocabulary problem. All words of a language cannot be used for training the neural machine model (NMT) for translation. A fixed number of words are taken into the vocabulary which are the most frequently occurring words. The words not included in vocabulary are known as out-of-vocabulary words (rare words). Our work includes Roman-Urdu to Urdu transliteration which also handles rare word problem. We looked into different subword techniques to solve this problem. After thorough research, we decided to use the state of the art transformer model on tensor2tensor which is a new google library for neural machine learning.
The transformer model uses attention mechanism and looks at word/character level for transliteration. We used our own dataset that included around 6million sentences for Roman-Urdu and Urdu-script each. We made our dataset work on tensor2tensor transformer model by extending the problem class and overriding functions. The Roman Urdu sentences were given as input and their corresponding Urdu Sentences were given to the model as targets. We trained for 100k steps with a vocab size of 20k and tuned hyper parameters. The BLEU score we managed to achieve was 80.7 which exceeded our target. The loss function became quite stable after 70k steps. Further improvements can be done by some further tweaks. The outputs are shown in the report, any sentence in Roman Urdu can be given for input and its transliteration will be done by the model and results will be stored in a file. 


# License and Copyright
:copyright: 2019-2020, FAST-National University of Computer and Emerging Sciences, Pakistan.
