# A tweet potical inclination classifier

- preprocess_data.py contains scripts to label raw tweets using OpenAI API into one of this labels ('Democrat', 'Republican', 'Neutral') and clean input tweets before feeding them to the model
- train.py trains a neural network using LSTM(bidirectional), CNN in parallel. I have used the tensorflow vectorizer layer to tokenize tweets and a pretained embendding layer downloaded from Stanford University Glove project. ref: https://nlp.stanford.edu/projects/glove/
