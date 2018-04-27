import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers

from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt


# "Varje rad är en sekvens som är en vektor som innehåller 5 vektorer som
#  var och en innehåller ett element som utgör en tidspunkt i sekvensen, så att säga"
data1 = [ [ [(i+j)/200] for i in range(5) ] for j in range(100) ]
data2 = [ [ [(i+j)/200] for i in range(6) ] for j in range(100,200) ]
data = data1 + data2
target1 = [ (i+5)/200 for i in range(100) ]
target2 = [ (i+6)/200 for i in range(100,200) ]
target = target1 + target2

print(np.array(data).shape)
print(np.array(target).shape)  
print(np.array(data[0]))
print(np.array(data[100]))
print([target[100]])
Xt, Xv, yt, yv = train_test_split(data, target, test_size=0.2, random_state=4)


def fixXvandYv():
    a = np.zeros((len(Xv), 6, 1))
    t = np.zeros((len(Xv), 1))
    for i in range(len(Xv)):
        c = np.array(Xv[i])
        a[i][0:c.shape[0]] = c
        t[i] = np.array(yv[i])
    return a, t


def generator(batch_size, X, y):
    """ Trying batch-wise padding of sequences
    """
    while True:
        max_seq_length = 0
        sequences = []
        targets = np.zeros((batch_size, 1))
        for i in range(batch_size):
            #get features from data
            selection = random.choice(range(len(X)))
            sequence = np.array(X[selection])
            targets[i] = y[selection]


            # find longest sequence
            if len(sequence) > max_seq_length:
                max_seq_length = len(sequence)
            sequences.append(sequence)

        pad_seqs = np.zeros((batch_size, max_seq_length, 1))
        #print("pad_seqs.shape", pad_seqs.shape)

        # pad seqences        
        for i in range (batch_size):
            #print("batch shape", sequences[i].shape)
            pad_seqs[i][0:sequences[i].shape[0]] = sequences[i]
        #print(pad_seqs)
        yield pad_seqs, targets
        
                
model = Sequential()
model.add(LSTM((1), batch_input_shape=(None, None, 1), return_sequences=False ) )

opt = optimizers.Adam(lr=1e-4)
model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['accuracy'])
model.summary()

model.fit_generator(generator(40, Xt, yt), samples_per_epoch=50,
 epochs=400, validation_data=generator(40, Xv, yv), validation_steps=50)

predX, ytest = fixXvandYv()
print(predX.shape)
pred = model.predict(predX)

print("pred shape", pred.shape, "ytest shape", ytest.shape)

plt.scatter(range(40), pred, c='r')
plt.scatter(range(40), ytest, c='g')
plt.show()