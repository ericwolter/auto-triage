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
          path1 = "../data/images/{:06d}-{:02d}.JPG".format(prefix, a)
          path2 = "../data/images/{:06d}-{:02d}.JPG".format(prefix, b)

          # forward comparison
          X[set][0].append(path1)
          X[set][1].append(path2)

          if set in ["train", "valid"]:
              score = float(elements[3])
              if score < 0.5:
                Y[set].append([1, 0])
              else:
                Y[set].append([0, 1])

              if set == "train":
                  #backward comparison
                  X[set][0].append(path2)
                  X[set][1].append(path1)
                  if score < 0.5:
                    Y[set].append([0, 1])
                  else:
                    Y[set].append([1, 0])

      X[set][0], X[set][1], Y[set] = map(np.array, [X[set][0], X[set][1], Y[set]])
  return X, Y
