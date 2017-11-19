import os, cv2, tqdm
import numpy as np

def load_data(FLAGS):
  X, Y = {}, {}
  for set in ["train", "valid", "test"]:
    if os.path.exists("../data/" + set + ".npz"):
      files = np.load(open("../data/" + set + ".npz", "r"))
      X[set], Y[set] = [files["X0"], files["X1"]], files["Y"]
    else:
      X[set], Y[set] = [[], []], []
      with open("../data/" + set + ".txt", "r") as file:
        for line in tqdm.tqdm(file.readlines(), desc = set):
          elements = line.strip().split()
          prefix, a, b = map(int, elements[:3])
          for i, x in enumerate([a, b]):
            path = "../data/images/{:06d}-{:02d}.JPG".format(prefix, x)
            # image = cv2.imread(path)
            # width, height = image.shape[:2]
            # if width < height:
            #   width = int(224. * width / height)
            #   height = 224
            # else:
            #   height = int(224. * height / width)
            #   width = 224
            # image = cv2.resize(image, (height, width)).astype(np.float32)
            # image = np.pad(image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode = "constant", constant_values = 0)
            # X[set][i].append(image / 255.)
            X[set][i].append(path)
          if set in ["train", "valid"]:
            score = float(elements[3])
            if score < 0.5:
              Y[set].append([1, 0])
            else:
              Y[set].append([0, 1])
            if set == "valid" and score >= 0.3 and score <= 0.7:
              X[set][0].pop()
              X[set][1].pop()
              Y[set].pop()
      X[set][0], X[set][1], Y[set] = map(np.array, [X[set][0], X[set][1], Y[set]])
      #np.savez(open("../data/" + set + ".npz", "w"), X0 = X[set][0], X1 = X[set][1], Y = Y[set])
  return X, Y
