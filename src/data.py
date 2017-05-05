import cv2, tqdm
import numpy as np

def load_data():
  X, Y = {}, {}
  for set in ["train", "valid"]:
    X[set], Y[set] = [[], []], []
    with open("../data/" + set + ".txt") as file:
      for line in tqdm.tqdm(file.readlines()):
        elements = line.strip().split()
        prefix, a, b = map(int, elements[:3])
        score = float(elements[3])
        if score >= 0.3 and score <= 0.7:
          continue
        for i, x in enumerate([a, b]):
          image = cv2.imread("../data/images/{:06d}-{:02d}.JPG".format(prefix, x))
          width, height = image.shape[:2]
          if width < height:
            width = int(224. * width / height)
            height = 224
          else:
            height = int(224. * height / width)
            width = 224
          image = cv2.resize(image, (height, width)).astype(np.float32)
          image = np.pad(image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode = "constant", constant_values = 0)
          X[set][i].append(image)
        if score < 0.5:
          Y[set].append([1, 0])
        else:
          Y[set].append([0, 1])
    X[set][0], X[set][1], Y[set] = map(np.array, [X[set][0], X[set][1], Y[set]])
  return X, Y