# Modifications are made on top of the original DeepCalib code.


from keras.layers import Layer, Input
import keras.backend as K


# Add attention layer to the deep learning network
## https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/
# first Attention model that came out - Bahdanau et al in 2014
class Attention(Layer):

    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(Attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def create_deep_learning_network(
    input_shape=(None, 50),
    lstm_layers_dimensions=(100, 50),
    number_of_outputs=2,
    DNN = 'LSTM',
    attention = False):
    """Creates and compiles a deep learning network.
    
    Inputs:    
    input_shape: Should be the same size of the input trajectory []
    lstm_layers_dimensions: number of neurons in each LSTM layer [tuple of positive integers]
        
    Output:
    network: deep learning network
    """    

    from keras import models, layers, optimizers, Model


    if attention == False:

        ### INITIALIZE DEEP LEARNING NETWORK
        network = models.Sequential()

        ### CONVOLUTIONAL BASIS
        for lstm_layer_number, lstm_layer_dimension in zip(range(len(lstm_layers_dimensions)), lstm_layers_dimensions):

            # add LSTM layer
            lstm_layer_name = 'lstm_' + str(lstm_layer_number + 1)


            ## GRU
            if DNN == 'GRU':
                if lstm_layer_number + 1 < len(lstm_layers_dimensions): # All layers but last
                    lstm_layer = layers.GRU(lstm_layer_dimension,
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name=lstm_layer_name)
                else: # Last layer
                    lstm_layer = layers.GRU(lstm_layer_dimension,
                                             return_sequences=False,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name=lstm_layer_name)

            
            ## LSTM
            if DNN == 'LSTM':
                if lstm_layer_number + 1 < len(lstm_layers_dimensions): # All layers but last
                    lstm_layer = layers.LSTM(lstm_layer_dimension,
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name=lstm_layer_name)
                else: # Last layer
                    lstm_layer = layers.LSTM(lstm_layer_dimension,
                                             return_sequences=False,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name=lstm_layer_name)


            network.add(lstm_layer)
        
        # OUTPUT LAYER
        output_layer = layers.Dense(number_of_outputs, name='output')
        network.add(output_layer)
    

    ## adding attention
    if attention == True:

        # set input shape
        x=Input(shape=input_shape)
        
        ## implement DNN model
        if DNN == 'SimpleRNN':
            lstm_layer_0 = layers.SimpleRNN(lstm_layers_dimensions[0],
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name='simpleRNN_0')

            lstm_layer_0 = lstm_layer_0(x)
            
            lstm_layer_1 = layers.SimpleRNN(lstm_layers_dimensions[1],
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name='simpleRNN_1')
            lstm_layer_1 = lstm_layer_1(lstm_layer_0)
        
        if DNN == 'LSTM':
            lstm_layer_0 = layers.LSTM(lstm_layers_dimensions[0],
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name='lstm_0')

            lstm_layer_0 = lstm_layer_0(x)
            
            lstm_layer_1 = layers.LSTM(lstm_layers_dimensions[1],
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name='lstm_1')
            lstm_layer_1 = lstm_layer_1(lstm_layer_0)

        if DNN == 'GRU':
            lstm_layer_0 = layers.GRU(lstm_layers_dimensions[0],
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name='gru_0')

            lstm_layer_0 = lstm_layer_0(x)
            
            lstm_layer_1 = layers.GRU(lstm_layers_dimensions[1],
                                             return_sequences=True,
                                             dropout=0,
                                             recurrent_dropout=0,
                                             input_shape=input_shape,
                                             name='gru_1')
            lstm_layer_1 = lstm_layer_1(lstm_layer_0)


        # add attention layer
        attention_layer = Attention()(lstm_layer_1)
        
        # output layer
        outputs= layers.Dense(number_of_outputs, name='output')(attention_layer)
        
        # custome model
        network= Model(x,outputs)


    ## compile
    network.compile(optimizer=optimizers.Adam(lr=1e-3), loss='mse', metrics=['mse', 'mae'])
    
    return network