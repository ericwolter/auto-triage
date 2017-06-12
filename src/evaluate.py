import os, json, argparse

from models import create_model
from data import load_data

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", default = "0")
  parser.add_argument("--exp", default = "default")

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  arguments = json.load(open("../exp/" + FLAGS.exp + "/arguments.json", "r"))
  for key, value in FLAGS.__dict__.iteritems():
    arguments[key] = value
  for key, value in arguments.iteritems():
    setattr(FLAGS, key, value)

  if not hasattr(FLAGS, "optimizer"):
    setattr(FLAGS, "optimizer", "sgdm")
  if not hasattr(FLAGS, "model"):
    setattr(FLAGS, "model", "vgg16")
  if not hasattr(FLAGS, "siamese"):
    setattr(FLAGS, "siamese", "share")
  if not hasattr(FLAGS, "weights"):
    setattr(FLAGS, "weights", "imagenet")
  if not hasattr(FLAGS, "module"):
    setattr(FLAGS, "module", "subtract")
  if not hasattr(FLAGS, "activation"):
    setattr(FLAGS, "activation", "tanh")
  if not hasattr(FLAGS, "regularizer"):
    setattr(FLAGS, "regularizer", "l2")

  X, Y = load_data(FLAGS)
  model = create_model(FLAGS)
  model.load_weights("../exp/" + FLAGS.exp + "/weights.hdf5")

  if FLAGS.optimizer == "sgd":
    from keras.optimizers import SGD
    optimizer = SGD(lr = 0.001)
  elif FLAGS.optimizer == "sgdm":
    from keras.optimizers import SGD
    optimizer = SGD(lr = 0.001, momentum = 0.9)
  elif FLAGS.optimizer == "adam":
    from keras.optimizers import Adam
    optimizer = Adam(lr = 0.01)
  else:
    raise NotImplementedError

  model.compile(optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

  results = {}
  for set in ["train", "valid"]:
    loss, accuracy = model.evaluate(X[set], Y[set], batch_size = FLAGS.batch, verbose = 1)
    results[set] = accuracy
    print set, "accuracy:", accuracy
  json.dump(results, open("../exp/" + FLAGS.exp + "/results.json", "w"))