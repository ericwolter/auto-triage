from keras.models import Model
from keras.layers import Input, Lambda, Dense
from keras.applications.vgg16 import VGG16

def remove_last_layer(model):
  model.layers.pop()
  model.outputs = [model.layers[-1].output]
  model.layers[-1].outbound_nodes = []

def create_model():
  input_a = Input(shape = (224, 224, 3))
  input_b = Input(shape = (224, 224, 3))

  feature_extractor = VGG16()
  remove_last_layer(feature_extractor)

  feature_a = feature_extractor(input_a)
  feature_b = feature_extractor(input_b)
  delta = Lambda(lambda x: x[0] - x[1], output_shape = lambda s: s[0])([feature_a, feature_b])

  hidden1 = Dense(128, activation = "tanh")(delta)
  hidden2 = Dense(128, activation = "tanh")(hidden1)
  output = Dense(2, activation = "softmax")(hidden2)

  return Model([input_a, input_b], output)