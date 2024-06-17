# Generated Data Visualization
여기서는 VAE 모델이 학습한 latent space와 VAE 디코더가 생성한 이미지를 가시화하는 가이드를 제공합니다.

### 1. Visualization
#### 1.1 Arguments
`src/run/tnse_test.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-r`, `--resume_model_dir`]: 데이터를 생성할 모델의 경로. `${project}/${name}`까지의 경로만 입력하면, 자동으로 `${project}/${name}/weights/`의 모델을 선택하여 모델을 로드합니다.
* [`-l`, `--load_model_type`]: [`loss`, `last`] 중 하나를 선택.
    * `loss`(default): Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-d`, `--dataset_type`]: (default: `test`) [`train`, `validation`, `test`] 중 하나를 선택.


#### 1.2 Command
`src/run/tsne_test.py` 파일로 다음과 같은 명령어를 통해 학습된 모델이 생성한 이미지를 가시화합니다.
```bash
python3 src/run/tsne_test.py --resume_model_dir ${project}/${name}
```
