import os, cv2, json, argparse
import numpy as np

from models import create_model

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--gpu", default = "0")
  parser.add_argument("--exp", default = "default")
  parser.add_argument("images", nargs = "+")

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

  for image in FLAGS.images:
    if not os.path.exists(image):
      raise Exception("<" + image + "> not found")

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

  for a in xrange(len(FLAGS.images)):
    for b in xrange(a + 1, len(FLAGS.images)):
      X = []
      for i, name in enumerate([FLAGS.images[a], FLAGS.images[b]]):
        image = cv2.imread(name)
        width, height = image.shape[:2]
        if width < height:
          width = int(224. * width / height)
          height = 224
        else:
          height = int(224. * height / width)
          width = 224
        image = cv2.resize(image, (height, width)).astype(np.float32)
        image = np.pad(image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode = "constant", constant_values = 0)
        image = np.expand_dims(image, axis = 0)
        X.append(image / 255.)
      score = model.predict(X, batch_size = 1)[0] * 100
      if score[0] > score[1]:
        print "<" + FLAGS.images[a] + ">", "is better than", "<" + FLAGS.images[b] + ">", "with {:.1f}% confidence".format(score[0])
      else:
        print "<" + FLAGS.images[b] + ">", "is better than", "<" + FLAGS.images[a] + ">", "with {:.1f}% confidence".format(score[1])