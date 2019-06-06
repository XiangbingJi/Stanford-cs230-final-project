import matplotlib.pyplot as plt
import pickle

train_loss = []
val_loss = []
with open('history.json', 'rb') as f:
    while 1:
        try:
            his = pickle.load(f)
            train_loss.append(his['loss'])
            val_loss.append(his['val_loss'])
        except EOFError:
            break

plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
