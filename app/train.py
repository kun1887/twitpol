import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from keras.layers import Embedding
import keras
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure
import pickle
import lime
from lime.lime_text import LimeTextExplainer

breakpoint()


path = input("Insert clean dataframe path: ")
data = pd.read_csv(path)

data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# assign numerical value to the 3 classes

label = {"Neutral": "0", "Democrat": "1", "Republican": 2}
data["gpt_class"] = data["gpt_class"].map(label)

# create training, testing split

split = int(0.8 * len(data))
train = data.iloc[:split]
test = data.iloc[split:]

X_train = train["clean_tweet"]
X_test = test["clean_tweet"]
y_train = train["gpt_class"]
y_test = test["gpt_class"]

# categorize y
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# tokenize X
vectorizer = layers.TextVectorization(max_tokens=20_000, output_sequence_length=200)
vectorizer.adapt(X_train)
X_train_tok = vectorizer(X_train)
X_test_tok = vectorizer(X_test)

voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# Create the embendding layer with pretrained weights

path_to_glove_file = input("path to glove files: ")

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

num_tokens = len(voc) + 1
embedding_dim = 200
hits = 0
misses = 0

# Prepare embedding matrix

embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

embedding_layer = Embedding(num_tokens, embedding_dim, trainable=False, mask_zero=True)
embedding_layer.build((1,))
embedding_layer.set_weights([embedding_matrix])


# Model constructor
def classifier():

    int_sequences_input = keras.Input(shape=(None,), dtype="int32")
    # embendding layer with pretrained weights
    embedded_sequences = embedding_layer(int_sequences_input)

    x = layers.Bidirectional(layers.LSTM(100))(embedded_sequences)

    z = layers.Conv1D(32, 3)(embedded_sequences)
    z = layers.GlobalMaxPool1D()(z)

    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(0.2)(z)

    z = layers.Dense(16, activation="relu")(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Dense(8, activation="relu")(z)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    concated_outputs = layers.Concatenate()([x, z])

    preds = layers.Dense(3, activation="softmax")(concated_outputs)

    model = keras.Model(int_sequences_input, preds)

    # model compile

    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    return model


# Balancing classes
weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(np.array(y_train)), y=np.array(y_train)
)
class_weights = dict(enumerate(weights))
# Early stopping creterion
es = EarlyStopping(patience=5, restore_best_weights=True)

# train the model
model = classifier()
history = model.fit(
    X_train_tok,
    y_train_cat,
    validation_split=0.2,  # Use 20% of training data as validation for early stopping
    class_weight=class_weights,
    callbacks=[es],
    epochs=20,
)

# ------------------ End of the training script ---------------------------


# Some functions used to save trained model and tokenizer and evalute its performance


def classificatio_report(confidence_threshold=0.8):
    """
    Print the model classification report with adjustable confidence threshold
    """
    # Start timing
    start_time = time.time()
    # Get the predictions and the confidence scores
    predictions_with_confidence = model.predict(X_test_tok)
    confidence_scores = np.max(predictions_with_confidence, axis=-1)
    predictions = np.argmax(predictions_with_confidence, axis=-1)
    # Filter predictions based on the confidence threshold
    high_confidence_indices = confidence_scores >= confidence_threshold
    high_confidence_predictions = predictions[high_confidence_indices]
    high_confidence_actuals = y_test.iloc[high_confidence_indices]
    # Generate the classification report
    print(classification_report(high_confidence_actuals, high_confidence_predictions))
    print("Time taken to predict the model: " + str(time.time() - start_time))


def plot_cofusion_matrix():
    """
    Plot model confusion matrix
    """

    y_pred = np.argmax(model.predict(X_test_tok, verbose=True), axis=1)
    labels = list(["Neu", "Dem", "Rep"])
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    plt.show()


def save_tokenizer():
    """
    Save the tokenizer in a pickle file
    """
    pickle.dump(
        {"config": vectorizer.get_config(), "weights": vectorizer.get_weights()},
        open("tv_layer.pkl", "wb"),
    )
    print("Tokenizer saved in tv_layer.pkl ✅")


def save_model():
    model.save("model.h5")
    print("model saved as model.h5 ✅")


# Create a Keras Classifier Wrapper
class KerasClassifierWrapper:
    def __init__(self, model, tokenizer, maxlen):
        self.model = model
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def predict_proba(self, texts):
        sequences = self.tokenizer(texts)
        return self.model.predict(sequences)


# LIME model prediction explainer
def explain_pred_from_test(idx=None):
    wrapper = KerasClassifierWrapper(model, vectorizer, X_train_tok.shape[1])
    ls_X_test = list(X_test)  # Your test data should be prepared similarly
    class_names = {0: "neutral", 1: "democrat", 2: "republican"}

    explainer = LimeTextExplainer(class_names=class_names)

    if idx == None:
        idx = np.random.randint(len(test))

    explanation = explainer.explain_instance(
        ls_X_test[idx], wrapper.predict_proba, num_features=20, labels=(0, 1, 2)
    )
    print("Document id: %d" % idx)
    print("*" * 50)
    print("Text: ", ls_X_test[idx])
    print("*" * 50)
    # print('Probability democratic =', wrapper.predict_proba([ls_X_test[idx]]).round(3)[0,1])
    print("True class: %s" % class_names[test.iloc[idx]["target"]])
    print("*" * 50)
    # Show the explainability results with highlighted text
    print("0: 'neutral', 1: 'democrat', 2: 'republican'")
    explanation.show_in_notebook(text=True, labels=(0, 1, 2))


def explain_pred_prompt(text):

    wrapper = KerasClassifierWrapper(model, vectorizer, 200)
    class_names = {0: "neutral", 1: "democrat", 2: "republican"}

    explainer = LimeTextExplainer(class_names=class_names)

    explanation = explainer.explain_instance(
        text, wrapper.predict_proba, num_features=20, labels=(0, 1, 2)
    )
    print("*" * 50)
    print("Text: ", text)
    print("*" * 50)
    # Show the explainability results with highlighted text
    print("0: 'neutral', 1: 'democrat', 2: 'republican'")
    explanation.show_in_notebook(text=True, labels=(0, 1, 2))
