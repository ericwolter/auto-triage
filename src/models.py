from keras.layers import Input, Lambda, Dense, Dropout
from keras.regularizers import l2
from keras.models import Model

def add_regularizer(model, kernel_regularizer = l2(), bias_regularizer = l2()):
  for layer in model.layers:
    if hasattr(layer, "kernel_regularizer"):
      layer.kernel_regularizer = kernel_regularizer
    if hasattr(layer, "bias_regularizer"):
      layer.bias_regularizer = bias_regularizer

def create_model(FLAGS):
  input_a = Input(shape = (224, 224, 3))
  input_b = Input(shape = (224, 224, 3))

  if FLAGS.model == "vgg16":
    from keras.applications.vgg16 import VGG16
    feature_extractor = VGG16()
  elif FLAGS.model == "vgg19":
    from keras.applications.vgg19 import VGG19
    feature_extractor = VGG19()
  elif FLAGS.model == "resnet50":
    from keras.applications.resnet50 import ResNet50
    feature_extractor = ResNet50()
  else:
    raise NotImplementedError

  feature_extractor.layers.pop()
  add_regularizer(feature_extractor)

  feature_a = feature_extractor(input_a)
  feature_b = feature_extractor(input_b)
  delta = Lambda(lambda x: x[0] - x[1], output_shape = lambda s: s[0])([feature_a, feature_b])

  hidden1 = Dropout(0.5)(Dense(128, activation = "tanh")(delta))
  hidden2 = Dense(128, activation = "tanh")(hidden1)
  output = Dense(2, activation = "softmax")(hidden2)

  model = Model([input_a, input_b], output)
  add_regularizer(model)
  return model