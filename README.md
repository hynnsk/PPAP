# Progressive Proxy Anchor Propagation for Unsupervised Semantic Segmentation (ECCV 2024)
Hyun Seok Seong</sup>, WonJun Moon</sup>, SuBeen Lee</sup>, Jae-Pil Heo</sup>

Sungkyunkwan University

[[Arxiv](https://arxiv.org/abs/2407.12463)] | [[Paper]()]

![main_figure](https://github.com/user-attachments/assets/9721f19d-326d-4033-a1f1-08510276f251)

## Abstract
> The labor-intensive labeling for semantic segmentation has spurred the emergence of Unsupervised Semantic Segmentation. Recent studies utilize patch-wise contrastive learning based on features from image-level self-supervised pretrained models. However, relying solely on similarity-based supervision from image-level pretrained models often leads to unreliable guidance due to insufficient patch-level semantic representations. To address this, we propose a Progressive Proxy Anchor Propagation (PPAP) strategy. This method gradually identifies more trustworthy positives for each anchor by relocating its proxy to regions densely populated with semantically similar samples. Specifically, we initially establish a tight boundary to gather a few reliable positive samples around each anchor. Then, considering the distribution of positive samples, we relocate the proxy anchor towards areas with a higher concentration of positives and adjust the positiveness boundary based on the propagation degree of the proxy anchor. Moreover, to account for ambiguous regions where positive and negative samples may coexist near the positiveness boundary, we introduce an instance-wise ambiguous zone. Samples within these zones are excluded from the negative set, further enhancing the reliability of the negative set. Our state-of-the-art performances on various datasets validate the effectiveness of the proposed method for Unsupervised Semantic Segmentation.
----------


## Requirements
Install following packages.
```
- python=3.6.9
- pytorch=1.10.2
- torchvision=0.9.1
- torchmetrics=0.8.2
- pytorch-lightning
- matplotlib
- tqdm
- scipy
- hydra-core
- seaborn
- pydensecrf
```

## Prepare datasets
Change the `pytorch_data_dir` variable in `dataset_download.py` according to your data directory where datasets are stored and run:
```
python ./dataset/dataset_download.py
```
Then, extract the zip files.

## Training & Evaluation
You should modify the data path in "<path_to_PPAP>/json/server/cocostuff.json" according to your dataset path.

```data_path
"dataset": {
        "data_type": "cocostuff27",
        "data_path": "<YOUR_COCOSTUFF_PATH>",
```

To train the model, run the code as below:
```train
python run.py --opt ./json/server/cocostuff.json --debug
```
If you wish to see the training progress through wandb, configure the wandb settings in the JSON file and remove --debug.

To evaluate, you should modify the checkpoint path in "<path_to_PPAP>/json/server/cocostuff_eval.json" according to the saved checkpoint path:
```ckpt_path
"output_dir": "./output/",
"checkpoint": "ppap_saved",
```

Then run the evaluation code as below:
```
python eval.py --opt ./json/server/***_eval.json --debug
```

Note that all of our experiments are tested on single A6000 GPU.

### Checkpoints
Dataset | Backbone | Model file
 -- | -- | --
COCO-stuff | ViT-S/8 | [checkpoint](https://drive.google.com/file/d/162h3jZfribkfHcGQ_6THp8mLYbnST_P0/view?usp=sharing)
COCO-stuff | ViT-S/16 | [checkpoint](https://drive.google.com/file/d/1U6fsOnZJev5gj6LTjtb8x1iPjzjy1nOX/view?usp=sharing)
Cityscapes | ViT-B/8 | [checkpoint](https://drive.google.com/file/d/1kATTeEH3LllmGz3PPyn34lyBxavSUGLe/view?usp=sharing)
Potsdam-3 | ViT-B/8 | [checkpoint](https://drive.google.com/file/d/1CDngRnD8bVqzuILTmbink61hB4IUVzxE/view?usp=sharing)


## Licence
This repository is built based on [STEGO](https://github.com/mhamilton723/STEGO) and [HP](https://github.com/hynnsk/HP).
Our codes are released under [MIT](https://opensource.org/licenses/MIT) license.

## Citation
If you find this project useful, please consider the following citation:
```
@article{seong2024progressive,
  title={Progressive Proxy Anchor Propagation for Unsupervised Semantic Segmentation},
  author={Seong, Hyun Seok and Moon, WonJun and Lee, SuBeen and Heo, Jae-Pil},
  journal={arXiv preprint arXiv:2407.12463},
  year={2024}
}
```

