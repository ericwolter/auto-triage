import os, json, argparse
import random
import numpy as np
import shutil
import PIL

from models import create_model
from data import load_data

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import Sequence


IMAGE_FULL_SIZE = 246
IMAGE_TARGET_SIZE = 224

def load_image(path, flip=False, random_size=IMAGE_TARGET_SIZE, random_offset_h=0, random_offset_v=0):
    img = image.load_img(path)
    img = img.resize((IMAGE_FULL_SIZE, IMAGE_FULL_SIZE), resample=PIL.Image.LANCZOS)

    window = (random_offset_h, random_offset_v,
              random_size + random_offset_h, random_size + random_offset_v)
    img = img.crop(window)

    if random_size != IMAGE_TARGET_SIZE:
        img = img.resize((IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE), resample=PIL.Image.LANCZOS)

    if flip:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    x = image.img_to_array(img)
    x = preprocess_input(x)

    return x

class TriageSequence(Sequence):
    def __init__(self, x0_set, x1_set, y_set, batch_size, augmented=True):
        self.X0,self.X1,self.y = x0_set,x1_set,y_set
        self.batch_size = batch_size
        self.augmented = augmented

    def __len__(self):
        return len(self.y) // self.batch_size

    def __getitem__(self,idx):
        batch_x0 = self.X0[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x1 = self.X1[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

        flip = False
        random_size = IMAGE_FULL_SIZE
        random_offset_h = 0
        random_offset_v = 0

        if self.augmented:
            if np.random.random() < 0.5:
                flip = True

            random_size = random.randint(IMAGE_TARGET_SIZE, IMAGE_FULL_SIZE)
            random_offset_h = random.randint(0, IMAGE_FULL_SIZE - random_size)
            random_offset_v = random.randint(0, IMAGE_FULL_SIZE - random_size)

        batch_x0 = np.array([load_image(file_name, flip=flip, random_size=random_size, random_offset_h=random_offset_h, random_offset_v=random_offset_v) for file_name in batch_x0])
        batch_x1 = np.array([load_image(file_name, flip=flip, random_size=random_size, random_offset_h=random_offset_h, random_offset_v=random_offset_v) for file_name in batch_x1])
        batch_y = np.array(batch_y)

        return ({"input_1": batch_x0,"input_2": batch_x1}, batch_y)

    def on_epoch_end(self):
        pass


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp", default = "default")
  parser.add_argument("--gpu", default = "0")
  parser.add_argument("--epochs", default = 128, type = int)
  parser.add_argument("--batch", default = 16, type = int)
  parser.add_argument("--optimizer", default = "nadam")
  parser.add_argument("--model", default = "mobilenet")
  parser.add_argument("--siamese", default = "share")
  parser.add_argument("--weights", default = "imagenet")
  parser.add_argument("--module", default = "neural")
  parser.add_argument("--activation", default = "relu")
  parser.add_argument("--regularizer", default = "l2")

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  shutil.rmtree("../exp/" + FLAGS.exp + "/")
  os.makedirs("../exp/" + FLAGS.exp + "/")

  print(FLAGS.__dict__)

  json.dump(FLAGS.__dict__, open("../exp/" + FLAGS.exp + "/arguments.json", "w"))

  print("Loading data...")
  X, Y = load_data(FLAGS)
  #training_sequence = list(zip(X["train"][0], X["train"][1], Y["train"]))
  #validation_sequence = list(zip(X["valid"][0], X["valid"][1], Y["valid"]))
  training_sequence = TriageSequence(X["train"][0], X["train"][1], Y["train"], FLAGS.batch, augmented=True)
  validation_sequence = TriageSequence(X["valid"][0], X["valid"][1], Y["valid"], FLAGS.batch, augmented=False)

  print("Creating model...")
  model = create_model(FLAGS)
  model.summary()

  if FLAGS.optimizer == "sgd":
    from tensorflow.python.keras.optimizers import SGD
    optimizer = SGD(lr = 0.001)
  elif FLAGS.optimizer == "sgdm":
    from tensorflow.python.keras.optimizers import SGD
    optimizer = SGD(lr = 0.001, momentum = 0.9)
  elif FLAGS.optimizer == "adam":
    from tensorflow.python.keras.optimizers import Adam
    optimizer = Adam(lr = 0.01)
  elif FLAGS.optimizer == "nadam":
    from tensorflow.python.keras.optimizers import Nadam
    optimizer = Nadam()
  else:
    raise NotImplementedError

  callbacks = [
    ModelCheckpoint("../exp/" + FLAGS.exp + "/weights.hdf5", monitor = "val_acc", save_best_only = True, save_weights_only = True),
    TensorBoard(log_dir = "../exp/" + FLAGS.exp + "/logs/")
    #ReduceLROnPlateau()
    #ReduceLROnPlateau(monitor = "val_acc", factor = 0.5, patience = 2, verbose = 1)
  ]

  print("Compiling model...")
  model.compile(optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

  print("Training...")
  # 12075, 483, 2585
  training_iterations = len(training_sequence)
  validation_iterations = len(validation_sequence)
  print("Training:", training_iterations * FLAGS.batch)
  print("Validation:", validation_iterations * FLAGS.batch)

  model.fit_generator(training_sequence, training_iterations,
    validation_data = validation_sequence, validation_steps = validation_iterations,
    epochs = FLAGS.epochs, callbacks = callbacks, use_multiprocessing=True, workers=4, shuffle=True)

  #model.fit(X["train"], Y["train"], validation_data = [X["valid"], Y["valid"]], epochs = FLAGS.epochs, batch_size = FLAGS.batch, callbacks = callbacks, verbose = 1)
