# `auto-triage`

## Introduction
This project is a reproduction of the `SIGGRAPH 2016` paper `Automatic Triage for a Photo Series` written in `Python` with the help of `Keras` and `TensorFlow`.

## Structure
```
data/             # 
| demos/          #
| download.sh     #
src/              # 
| data.hpp        # 
| models.hpp      # 
| train.py        # 
| test.py         #
| predict.py      #
```

## Usages

### Requirements
* `Python`
* `Keras`
* `OpenCV`
* `numpy`
* `TensorFlow`

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
python predict.py
```

## License
This project is released under the [open-source MIT license](https://github.com/zhijian-liu/auto-triage/blob/master/LICENSE).
