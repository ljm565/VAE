# Generated Data Visualization
Here, we provide guides for visualizing generated data from the trained VAE model.

### 1. Visualization
#### 1.1 Arguments
There are several arguments for running `src/run/latent_visualization.py`:
* [-r, --resume_model_dir]: Directory to the model to be evaluated. Provide the path up to `{$project}/{$name}`, and it will automatically select the model from `{$project}/{$name}/weights/` to generate data.
* [-l, --load_model_type]: Choose one of [`loss`, `last`].
    * `loss` (default): Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [-d, --dataset_type]: (default: `test`) Choose one of [`train`, `validation`, `test`].


#### 1.2 Command
`src/run/latent_visualization.py` file is used to visualize the generated image from the trained model.
```bash
python3 src/run/latent_visualization.py --resume_model_dir {$project}/{$name}
```
