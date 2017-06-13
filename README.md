# `auto-triage`

## Introduction
This project is a reproduction of the `SIGGRAPH 2016` paper `Automatic Triage for a Photo Series` written in `Python` with the help of `Keras` and `TensorFlow`.

## Structure
```
data/             # benchmark dataset (Princeton Adobe photo triage dataset)
| demos/          # three demo scenarios (jpeg files)
| download.sh     # data downloading and preparation
src/              # source code
| data.py         # data loading and preprocessing
| models.py       # models with different settings
| train.py        # script for training
| evaluate.py     # script for evaluation
| predict.py      # script for prediction
```

## Usages

### Requirements
* `Python 2.7`
* `OpenCV 2`
* `Keras 2.0+`
* `TensorFlow 1.0+`

### Training
```bash
python train.py
```

### Evaluation

```
python evaluate.py
```

### Prediction

```
python predict.py <image-list>
```

## License
This project is released under the [open-source MIT license](https://github.com/zhijian-liu/auto-triage/blob/master/LICENSE).
