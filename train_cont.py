import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, BatchNormalization, LSTM
from keras.models import Sequential
import sys
from rnn import convert_word2word

# Define Architecture
vocab_size = 9055
if str(sys.argv[1] == 'char'): vocab_size = 42
model = Sequential()
model.add(Dense(100, input_shape=(5, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
model.add(LSTM(100, use_bias=True, unit_forget_bias=True, return_sequences=True))
model.add(LSTM(100, use_bias=True, unit_forget_bias=True, return_sequences=True))
model.add(LSTM(100, use_bias=True, unit_forget_bias=True, return_sequences=True))
model.add(LSTM(100, use_bias=True, unit_forget_bias=True, return_sequences=True))
model.add(LSTM(100, use_bias=True, unit_forget_bias=True))
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(300, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(vocab_size, activation='softmax'))

# Load from checkpoint
model.load_weights(str(sys.argv[1]))
rmsp = optimizers.RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy',
                  optimizer=rmsp,
                  metrics=['accuracy'])
print(model.summary())
checkpoints = ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', save_weights_only=True, period=25)
batch_size=24000
epochs=600
val=0.33
X, y, _, _ = convert_word2word('lyrics.txt', 5)
model.fit(X, y, batch_size=batch_size, epochs=epochs, callbacks=[checkpoints], validation_split=val)
