import os, argparse

from models import create_model
from data import load_data

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", default = "0")
  parser.add_argument("--exp", default = "default")
  parser.add_argument("--model", default = "vgg16")
  parser.add_argument("--optimizer", default = "sgd")
  parser.add_argument("--epochs", default = 16, type = int)
  parser.add_argument("--batch", default = 4, type = int)

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  X, Y = load_data(FLAGS)
  model = create_model(FLAGS)
  print model.summary()

  if FLAGS.optimizer == "sgd":
    from keras.optimizers import SGD
    optimizer = SGD(lr = 0.001, momentum = 0.9)
  elif FLAGS.optimizer == "adam":
    from keras.optimizers import Adam
    optimizer = Adam()
  else:
    raise NotImplementedError

  os.system("rm -rf ../exp/" + FLAGS.exp + "/")
  os.system("mkdir -p ../exp/" + FLAGS.exp + "/")

  callbacks = [
    ModelCheckpoint("../exp/" + FLAGS.exp + "/weights-{val_acc:.2f}.hdf5", monitor = "val_acc", verbose = 1, save_best_only = True, save_weights_only = True),
    TensorBoard(log_dir = "../exp/" + FLAGS.exp + "/logs/"),
    ReduceLROnPlateau(monitor = "val_acc", factor = 0.5, patience = 2, verbose = 1)
  ]

  model.compile(optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
  model.fit(X["train"], Y["train"], validation_data = [X["valid"], Y["valid"]], epochs = FLAGS.epochs, batch_size = FLAGS.batch, callbacks = callbacks, verbose = 1)