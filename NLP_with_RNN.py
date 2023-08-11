from keras.datasets import imdb
from keras_preprocessing import sequence
import tensorflow as tf
import numpy as np
import os
import keras


VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=VOCAB_SIZE)


train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

print(train_data[1])

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
    # This sigmoid function will put the output into a range between 0 and 1.
    # values less than .5 get classified as negative, greater than are positive
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    batch_size=BATCH_SIZE, epochs=3, validation_split=.2)

word_index = imdb.get_word_index()


def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]


text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key, value) in word_index.items()}


def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]


print(decode_integers(encoded))


def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])


positive_review = "That movie was awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive_review)

negative_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)

for i in range(10):
    predict(decode_integers(test_data[i]))
    print(test_labels[i])
