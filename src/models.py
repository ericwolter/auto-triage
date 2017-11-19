import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python.keras.layers import Input, Flatten, Lambda, Concatenate, Dense, Dropout
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.models import Model

def remove_last_layer(model):
  model.layers.pop()
  if not model.layers:
    model.outputs = []
    model.inbound_nodes = []
    model.outbound_nodes = []
  else:
    model.outputs = [model.layers[-1].output]
    #model.layers[-1].outbound_nodes = []

def add_regularizer(model, kernel_regularizer = l2(), bias_regularizer = l2()):
  for layer in model.layers:
    if hasattr(layer, "kernel_regularizer"):
      layer.kernel_regularizer = kernel_regularizer
    if hasattr(layer, "bias_regularizer"):
      layer.bias_regularizer = bias_regularizer

def feature_extractor(FLAGS, suffix = ""):
  weights = FLAGS.weights if FLAGS.weights != "random" else None

  if FLAGS.model == "vgg16":
    from tensorflow.python.keras.applications.vgg16 import VGG16
    feature_extractor = VGG16(weights = weights)
    remove_last_layer(feature_extractor)
  elif FLAGS.model == "vgg19":
    from tensorflow.python.keras.applications.vgg19 import VGG19
    feature_extractor = VGG19(weights = weights)
    remove_last_layer(feature_extractor)
  elif FLAGS.model == "resnet50":
    from tensorflow.python.keras.applications.resnet50 import ResNet50
    feature_extractor = ResNet50(weights = weights)
    remove_last_layer(feature_extractor)
  elif FLAGS.model == "mobilenet":
    from tensorflow.python.keras.applications.mobilenet import MobileNet
    feature_extractor = MobileNet(weights = weights, include_top = False, input_shape=(224, 224, 3), pooling="avg")
  else:
    raise NotImplementedError

  #feature_extractor.name = FLAGS.model + suffix

  #if FLAGS.regularizer == "l2":
    #add_regularizer(feature_extractor)
  #elif FLAGS.regularizer != "none":
    #raise NotImplementedError

  #feature_extractor.trainable = False

  return feature_extractor

def create_model(FLAGS):
  input_a = Input(shape = (224, 224, 3))
  input_b = Input(shape = (224, 224, 3))

  if FLAGS.siamese == "share":
    extractor = feature_extractor(FLAGS)
    feature_a = extractor(input_a)
    feature_b = extractor(input_b)
  elif FLAGS.siamese == "separate":
    extractor_a = feature_extractor(FLAGS, "a")
    extractor_b = feature_extractor(FLAGS, "b")
    feature_a = extractor_a(input_a)
    feature_b = extractor_b(input_b)
  else:
    raise NotImplementedError

  if FLAGS.module == "subtract":
    feature = Lambda(lambda x: x[0] - x[1])([feature_a, feature_b])
  elif FLAGS.module == "bilinear":
    raise NotImplementedError
  elif FLAGS.module == "neural":
    feature = Concatenate()([feature_a, feature_b])
    feature = Dense(128, activation = FLAGS.activation)(feature)
  else:
    raise NotImplementedError

  hidden1 = Dropout(0.5)(Dense(128, activation = FLAGS.activation)(feature))
  hidden2 = Dense(128, activation = FLAGS.activation)(hidden1)
  output = Dense(2, activation = "softmax")(hidden2)

  model = Model([input_a, input_b], output)

  if FLAGS.regularizer == "l2":
    add_regularizer(model)
  elif FLAGS.regularizer != "none":
    raise NotImplementedError
  return model
