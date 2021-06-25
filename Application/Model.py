from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization, TimeDistributed, ConvLSTM2D
from keras.layers.recurrent import LSTM

def myConv3d(inputShape, numClasses, filters, kernelSize, padding, activation, poolSize, dropout):
    model = Sequential()

    model.add(Conv3D(filters, kernel_size=kernelSize, padding=padding, activation=activation, input_shape=inputShape, data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=poolSize, padding=padding))

    model.add(Conv3D(filters*2, kernel_size=kernelSize, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=poolSize, padding=padding))
    
    model.add(Dropout(dropout))

    model.add(Conv3D(filters*4, kernel_size=kernelSize, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=poolSize, padding=padding))

    model.add(Conv3D(filters*2, kernel_size=kernelSize, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=poolSize, padding=padding))

    model.add(Dropout(dropout))

    model.add(Conv3D(filters, kernel_size=kernelSize, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=poolSize, padding=padding))

    model.add(Flatten())
    model.add(Dense(filters*4, activation='relu'))
    
    model.add(Dense(numClasses, activation='softmax'))    
    return model


def myConv2dLstm(inputShape, numClasses, filters, kernelSize, padding, lstmUnit, activation, poolSize, dropout):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters, kernel_size=kernelSize, padding=padding, activation=activation), input_shape=inputShape))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize, padding=padding)))

    model.add(TimeDistributed(Conv2D(filters*2, kernel_size=kernelSize, padding=padding, activation=activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize, padding=padding)))

    model.add(TimeDistributed(Conv2D(filters*4, kernel_size=kernelSize, padding=padding, activation=activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize, padding=padding)))

    model.add(TimeDistributed(Conv2D(filters*2, kernel_size=kernelSize, padding=padding, activation=activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize, padding=padding)))

    model.add(TimeDistributed(Conv2D(filters, kernel_size=kernelSize, padding=padding, activation=activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=poolSize, padding=padding)))
    
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(lstmUnit, return_sequences=False, dropout=dropout))
    model.add(BatchNormalization())
    model.add(Dense(filters*4, activation=activation))    
    model.add(Dense(numClasses, activation='softmax'))
    return model
