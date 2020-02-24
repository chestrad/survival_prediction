#A custom loss function is used which represents the negative log likelihood of the survival model.
#The following codes were developed by Michael F. Gensheimer and Balasubramanian Narasimhan. 
#Gensheimer MF, Narasimhan B. 2019. A scalable discrete-time survival model for neural networks. PeerJ 7:e6257 https://doi.org/10.7717/peerj.6257
#Copyright belongs to the original authors.
https://github.com/MGensheimer/nnet-survival

from __future__ import print_function
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer

def surv_likelihood(n_intervals):
  """Create custom Keras loss function for neural network survival model. 
  Arguments
      n_intervals: the number of survival time intervals
  Returns
      Custom loss function that can be used with Keras
  """
  def loss(y_true, y_pred):
    """
    Required to have only 2 arguments by Keras.
    Arguments
        y_true: Tensor.
          First half of the values is 1 if individual survived that interval, 0 if not.
          Second half of the values is for individuals who failed, and is 1 for time interval during which failure occured, 0 for other intervals.
          See make_surv_array function.
        y_pred: Tensor, predicted survival probability (1-hazard probability) for each time interval.
    Returns
        Vector of losses for this minibatch.
    """
    cens_uncens = 1. + y_true[:,0:n_intervals] * (y_pred-1.) #component for all individuals
    uncens = 1. - y_true[:,n_intervals:2*n_intervals] * y_pred #component for only uncensored individuals
    return K.sum(-K.log(K.clip(K.concatenate((cens_uncens,uncens)),K.epsilon(),None)),axis=-1) #return -log likelihood
  return loss 
