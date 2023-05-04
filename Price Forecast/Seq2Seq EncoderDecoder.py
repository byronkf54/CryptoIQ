import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Flatten, Bidirectional, Reshape, Concatenate, \
    TimeDistributed, RepeatVector, dot, Activation, Lambda
from keras.optimizers import Adam
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from keras.utils.vis_utils import plot_model

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path

data = pd.read_csv('./btc-usd-max.csv')
# make folder for images
# Path("/app/prediction_plots/Seq2Seq-tuning-plots").mkdir(parents=True, exist_ok=True)
# print(data.iloc[2345])
# print(data.tail())

print(data.info())

print(data.head())

series = data['price']

seq = series.copy()

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(seq.values.reshape(-1, 1))

plt.figure(figsize=(15, 7))
plt.title("Scaled Bitcoin Price from 28/4/2013 to 3/2/2023", fontsize=18)
plt.xlabel("Index", fontsize=18)
plt.ylabel("Price (USD)", fontsize=18)
plt.plot(X)
plt.show()

x_train = X[:-540]
y_test = X[-540:]


def generate_train_sequences(x, input_seq_len):
    total_start_points = len(x) - input_seq_len - output_seq_len

    start_x_idx = np.random.permutation(total_start_points)

    # gather indexes of prices from start_x_idx to start_x_idx + input_seq_len
    input_batch_idxs = np.arange(input_seq_len) + start_x_idx[:, None]
    input_seq = x[input_batch_idxs]

    # gather indexes of prices from start_x_idx + input_seq_len to start_x_idx + input_seq_len + output_seq_len
    output_batch_idxs = np.arange(output_seq_len) + (start_x_idx + input_seq_len)[:, None]
    output_seq = x[output_batch_idxs]

    return input_seq, output_seq


def create_model(input_seq_len, output_seq_len, layers, bidirectional=False):
    # Define input sequence
    encoder_inputs = Input(shape=(input_seq_len, 1))

    # Define output sequence
    decoder_inputs = Input(shape=(output_seq_len, 1))

    if bidirectional:
        # Define encoder LSTM layer
        # Encoder LSTM 1
        encoder_lstm1 = Bidirectional(LSTM(input_seq_len, return_sequences=True,
                                           return_state=True, dropout=0.4,
                                           recurrent_dropout=0.4))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm1(encoder_inputs)

        for i in range(1, layers):
            # Encoder LSTM 2 etc.
            encoder_lstm1 = Bidirectional(LSTM(input_seq_len, return_sequences=True, return_state=True, dropout=0.4,
                                               recurrent_dropout=0.4))
            (encoder_outputs, forward_h, forward_c, backward_h, backward_c) = encoder_lstm1(encoder_outputs)

        # Concatenate the hidden states of the forward and backward LSTM layers
        encoder_h = Concatenate()([forward_h, backward_h])
        encoder_c = Concatenate()([forward_c, backward_c])

        # Add a dense layer to reduce the size of the concatenated encoder hidden and cell states
        encoder_h = Dense(input_seq_len, activation='relu')(encoder_h)
        encoder_c = Dense(input_seq_len, activation='relu')(encoder_c)

    else:
        # Define encoder LSTM layer
        encoder_lstm = LSTM(input_seq_len, return_sequences=True,
                            return_state=True, dropout=0.4,
                            recurrent_dropout=0.4)
        encoder_outputs, encoder_h, encoder_c = encoder_lstm(encoder_inputs)

        for i in range(1, layers):
            # Encoder LSTM 2 etc.
            encoder_lstm = LSTM(input_seq_len, return_sequences=True, return_state=True, dropout=0.4,
                                recurrent_dropout=0.4)
            encoder_outputs, encoder_h, encoder_c = encoder_lstm(encoder_outputs)

        # Add a dense layer to reduce the size of the encoder hidden and cell states
        encoder_h = Dense(input_seq_len, activation='relu')(encoder_h)
        encoder_c = Dense(input_seq_len, activation='relu')(encoder_c)

    # Define decoder LSTM layer
    decoder_lstm = LSTM(input_seq_len, return_sequences=True, return_state=True)

    # Flatten the encoder output sequence
    encoder_outputs_flattened = Flatten()(encoder_outputs)

    # Reshape the flattened encoder outputs to 2D
    encoder_outputs_reshaped = Reshape((-1, input_seq_len))(encoder_outputs_flattened)

    # Extract the last timestep of the encoder output sequence
    last_encoder_output = Lambda(lambda x: x[:, -1, :])(encoder_outputs_reshaped)

    # Repeat the last timestep
    encoder_outputs_repeated = RepeatVector(output_seq_len)(last_encoder_output)

    # Pass the output sequence through the decoder LSTM layer
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[encoder_h, encoder_c])

    # Define attention layer
    attention = dot([decoder_outputs, encoder_outputs_repeated], axes=[2, 2])

    # Compute the attention weights
    attention_weights = Activation('softmax')(attention)

    # Apply the attention weights to the encoder outputs
    context = dot([attention_weights, encoder_outputs_repeated], axes=[2, 1])

    # Concatenate the decoder outputs and context
    decoder_concat = Concatenate(axis=-1)([decoder_outputs, context])

    # Pass the concatenated outputs through the output layer
    outputs = TimeDistributed(Dense(1, activation='linear'))(decoder_concat)

    # Define the model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
    # print(model.summary())
    # plot_model(model, to_file="seq2seq-model.png", dpi=500)
    return model


def train_model(model, epochs, batch_size, input_seq_len, total_loss, total_val_loss, batches=1):
    for _ in range(batches):
        input_seq, output_seq = generate_train_sequences(x_train, input_seq_len)
        exit()
        encoder_input_data = input_seq
        decoder_target_data = output_seq
        decoder_input_data = np.zeros(decoder_target_data.shape)

        history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            shuffle=False)

        total_loss.append(history.history['loss'])
        total_val_loss.append(history.history['val_loss'])

    return total_loss, total_val_loss


def plot_loss(input_seq_len, train_loss, val_loss, bidi, layers, batch_size, nb_epochs):
    plt.figure(figsize=(12, 7))
    plt.plot(train_loss)
    plt.plot(val_loss)

    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error Loss')
    plt.title('Loss Over Time')
    plt.legend(['Train', 'Valid'])
    plt.savefig(
        f'/app/prediction_plots/Seq2Seq-tuning-plots/ETH-loss-{nb_epochs}epochs-{input_seq_len}seq_len-{batch_size}batch_size-bidirectional{bidi}-{layers}layers.png')


def run_model(params):
    print("PARAMS: ", params)
    batch_size, bidi, input_seq_len, layers, nb_epochs = int(params["batch_size"]), params["bidi"], int(
        params["input_seq_len"]), int(params["layers"]), int(params["nb_epochs"])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f'running with: {input_seq_len}=seq_len-{batch_size}=batch_size-bidirectional={bidi}-{layers}layers')
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # create model
    model = create_model(input_seq_len, output_seq_len, layers, bidirectional=bidi)

    total_loss = []
    total_val_loss = []

    model.compile(Adam(), loss='mean_squared_error')

    total_loss, total_val_loss = train_model(model, epochs=nb_epochs, batch_size=batch_size, total_loss=total_loss,
                                             total_val_loss=total_val_loss, input_seq_len=input_seq_len)

    total_loss = [j for i in total_loss for j in i]
    total_val_loss = [j for i in total_val_loss for j in i]

    plot_loss(input_seq_len, total_loss, total_val_loss, bidi, layers, batch_size, nb_epochs)

    input_seq_test = x_train[-input_seq_len:].reshape((1, input_seq_len, 1))
    output_seq_test = y_test[:output_seq_len]
    decoder_input_test = np.zeros((1, output_seq_len, 1))

    pred = model.predict([input_seq_test, decoder_input_test])

    pred_values = scaler.inverse_transform(pred.reshape(-1, 1))
    output_seq_test = scaler.inverse_transform(output_seq_test)

    plt.plot(pred_values, label="pred")
    plt.plot(output_seq_test, label="actual")
    plt.title("Prediction vs Actual")
    plt.ylabel("Price ($)", fontsize=12)
    plt.xlabel("future_days", fontsize=12)
    plt.legend()
    plt.savefig(
        f'/app/prediction_plots/Seq2Seq-tuning-plots/ETH-prediction-{nb_epochs}epochs-{input_seq_len}seq_len-{batch_size}batch_size-bidirectional{bidi}-{layers}layers.png')

    return {"loss": total_loss[-1], "status": STATUS_OK}


output_seq_len = 180

# changing input_seq_len, layer_dimensions, number of layers, batch_size, bidirectionality
space = {
    "nb_epochs": hp.quniform("nb_epochs", 1, 20, 1),
    "bidi": hp.choice("bidi", [True, False]),
    "layers": hp.quniform("layers", 1, 6, 1),
    "input_seq_len": hp.quniform("input_seq_len", 60, 540, 30),
    "batch_size": hp.quniform("batch_size", 8, 33, 2)
}

trials = Trials()

best = fmin(
    fn=run_model,
    space=space,
    algo=tpe.suggest,
    max_evals=1000,
    trials=trials
)
