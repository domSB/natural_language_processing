#%%

"""
Data from Lazyprogrammer TF2.0 Course
"""
import tensorflow as tf
import pandas as pd
import altair as alt

from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/spam.csv', encoding='ISO-8859-1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.columns = ['label', 'sentence']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
y = df['label'].to_numpy()
x = df['sentence'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(x_train)
sequence_train = tokenizer.texts_to_sequences(x_train)
sequence_test = tokenizer.texts_to_sequences(x_test)

# replace with online padding
pad_seq_train = tf.keras.preprocessing.sequence.pad_sequences(sequence_train)
_, seq_len = pad_seq_train.shape
pad_seq_test = tf.keras.preprocessing.sequence.pad_sequences(sequence_test)

word2idx = tokenizer.word_index
vocab_size = len(word2idx)

recurrent_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(seq_len, )),
        tf.keras.layers.Embedding(vocab_size + 1, 32),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)

recurrent_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(f'Got {recurrent_model.count_params()} params')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
early_stop = tf.keras.callbacks.EarlyStopping(patience=2)

hist = recurrent_model.fit(
    pad_seq_train, y_train,
    validation_data=(pad_seq_test, y_test),
    epochs=20, batch_size=32,
    callbacks=[reduce_lr, early_stop])

results = pd.DataFrame(hist.history)
results.columns = ['Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy', 'Learning Rate']
with pd.HDFStore('../data/viz/results.h5') as store:
    store.put('experiment', results)
