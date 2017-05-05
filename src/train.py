import os, argparse

from models import create_model
from data import load_data

from keras.optimizers import SGD

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", dest = "gpu", default = "0")
  parser.add_argument("--model", dest = "model", default = "vgg16")
  parser.add_argument("--epochs", dest = "epochs", default = 16, type = int)
  parser.add_argument("--batch", dest = "batch", default = 4, type = int)

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  X, Y = load_data()
  model = create_model(FLAGS)
  print model.summary()

  model.compile(optimizer = SGD(lr = 0.001, momentum = 0.9), loss = "binary_crossentropy", metrics = ["accuracy"])
  model.fit(X["train"], Y["train"], validation_data = [X["valid"], Y["valid"]], nb_epoch = FLAGS.epochs, batch_size = FLAGS.batch, verbose = 1)