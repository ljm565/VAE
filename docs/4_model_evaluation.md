# Trained Model Evaluation
Here, we provide guides for evaluating the trained VAE model.


### 1. Evaluation
#### 1.1 Arguments
There are several arguments for running `src/run/validation.py`:
* [-r, --resume_model_dir]: Directory to the model to be evaluated. Provide the path up to `{$project}/{$name}`, and it will automatically select the model from `{$project}/{$name}/weights/` to evaluate.
* [-l, --load_model_type]: Choose one of [`loss`, `last`].
    * `loss` (default): Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [-d, --dataset_type]: (default: `test`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/validation.py` file is used to evaluate the model with the following command:
```bash
python3 src/run/validation.py --resume_model_dir {$project}/{$name}
```
