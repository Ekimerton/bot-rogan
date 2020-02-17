import tensorflow as tf
import numpy as np
import os
import time

print("Running tensorflow:", tf.__version__) # Make sure the version is 2.0 or above otherwise I die!!

# Function for building the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

# Generate Text function given by TensorFlow Tutorials
def generate_text(model, start_string, length):
  num_generate = length
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []

  temperature = 1.0 # Lower temp is less suprising, higher is more

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))

### Dictionary for turning letters into numbers
vocab = [' ', '$', '%', '&', "'", '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '>', '@', '[', ']', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'á', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ñ', 'ò', 'ó', 'õ', 'ö', 'ø', 'ú', 'ü', 'ÿ']
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

### Building model from preset weights
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
BATCH_SIZE = 64

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE
)

model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights('big_train.h5')

model.build(tf.TensorShape([1, None]))

print(generate_text(model, start_string="the government ", length=200))
