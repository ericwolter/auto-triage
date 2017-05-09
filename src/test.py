import os, argparse

from models import create_model
from data import load_data

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", default = "0")
  parser.add_argument("--exp", default = "default")
  parser.add_argument("--model", default = "vgg16")
  parser.add_argument("--optimizer", default = "sgd")
  parser.add_argument("--batch", default = 4, type = int)

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  X, Y = load_data(FLAGS)
  model = create_model(FLAGS)
  model.summary()

  model.load_weights("../exp/" + FLAGS.exp + "/weights.hdf5")

  if FLAGS.optimizer == "sgd":
    from keras.optimizers import SGD
    optimizer = SGD(lr = 0.001, momentum = 0.9)
  elif FLAGS.optimizer == "adam":
    from keras.optimizers import Adam
    optimizer = Adam()
  else:
    raise NotImplementedError

  model.compile(optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

  for set in ["train", "valid"]:
    loss, accuracy = model.evaluate(X[set], Y[set], batch_size = FLAGS.batch, verbose = 1)
    print set, "accuracy:", accuracy