# Data Preparation
Here, we will proceed with an VAE model training tutorial using the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset by default.
Please refer to the following instructions to utilize custom datasets.


### 1. MNIST
If you want to train on the MNIST dataset, simply set the `MNIST_train` value in the `config/config.yaml` file to `True` as follows.
```yaml
MNIST_train: True       
MNIST:
    path: data/
    MNIST_valset_proportion: 0.2 
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train your custom dataset, set the `MNIST_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
```yaml
MNIST_train: False       
MNIST:
    path: data/
    MNIST_valset_proportion: 0.2 
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```

