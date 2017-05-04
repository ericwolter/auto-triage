import os, argparse

from models import create_model
from data import load_data

from keras.optimizers import Adam

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", dest = "gpu", default = "0")

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  X, Y = load_data()
  model = create_model()
  print model.summary()

  model.compile(optimizer = Adam(), loss = "binary_crossentropy", metrics = ["accuracy"])
  model.fit(X["train"], Y["train"], validation_data = [X["valid"], Y["valid"]], batch_size = 4, nb_epoch = 128, verbose = 1)