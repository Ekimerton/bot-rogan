import tensorflow as tf
import numpy as np
import os
import time
import random
import tweepy

print("Running tensorflow:", tf.__version__) # Make sure the version is 2.0 or above otherwise I die!!

### Dictionary for turning letters into numbers
vocab = [' ', '$', '%', '&', "'", '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '>', '@', '[', ']', '^', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'á', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ñ', 'ò', 'ó', 'õ', 'ö', 'ø', 'ú', 'ü', 'ÿ']
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

### Building model from preset weights
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024
BATCH_SIZE = 64

consumer_token = os.environ['consumer_token']
consumer_token_secret = os.environ['consumer_token_secret']
access_token = os.environ['access_token']
access_token_secret = os.environ['access_token_secret']

auth = tweepy.OAuthHandler(consumer_token, consumer_token_secret)
auth.set_access_token(access_token, access_token_secret)
tp = tweepy.API(auth)

# Function for building the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Generate Text function given by TensorFlow Tutorials
# Lower temp is less suprising, higher is more
def generate_text(model, start_string, length, temperature=1.0):
    num_generate = length
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []


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

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def get_seed():
    seeds = ["martial", "chimps", "richard", "the government", "fight"]
    return random.choice(seeds)

#model = build_model(
 #   vocab_size = len(vocab),
  #  embedding_dim=embedding_dim,
   # rnn_units=rnn_units,
    #batch_size=BATCH_SIZE
#)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.compile(optimizer='adam', loss=loss)

model.load_weights('ai/jre.h5')

#model.build(tf.TensorShape([1, None]))
print("Starting tweet loop")
while True:
    seed = get_seed()
    tweet = generate_text(model, start_string=seed + " ", length=140,
                          temperature=0.7)
    tweet = "Seed phrase: " + seed + '\n' + tweet
    print(tweet)
    tp.update_status(tweet)
    time.sleep(60 * 60 * 12)
