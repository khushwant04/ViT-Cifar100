# ViT on CIFAR-100

A from-scratch Vision Transformer in PyTorch, trained on CIFAR-100. The code is split into separate modules for the model, dataset, training loop and helpers.

## What's inside

- `VisionTransformer` (`src/models/models.py`): Conv2d patch embedding, a learnable `[CLS]` token and positional embedding, a stack of encoder layers with hand-written multi-head self-attention, and a linear classification head on the `[CLS]` output.
- Encoder layers are post-norm (`LayerNorm(x + sublayer(x))`) with a ReLU MLP. The ViT paper uses pre-norm and GELU.
- `Train` (`src/train.py`): train/test loop with an explicit L2 penalty (`0.5 * lambda * sum(w^2)`) and TensorBoard logging to `runs/`.
- CIFAR-100 loaders (`src/datasets/cifar100.py`) that download the dataset to `./data`.

## Default configuration (`main.py`)

| Setting | Value |
| --- | --- |
| Patch size / embed dim | 16 / 768 |
| Heads / layers / MLP dim | 12 / 12 / 3072 |
| Dropout | 0.1 |
| Optimizer | AdamW, lr 1e-3 |
| L2 lambda | 1e-3 |
| Batch size / epochs | 128 / 100 |

`image_size` is set to 224, but the CIFAR-100 transform has no resize, so the model actually sees 32x32 images. With 16x16 patches that is only 4 patches plus the `[CLS]` token. The forward pass slices the positional embedding to fit, so it runs without errors. To train at 224x224, add `transforms.Resize(224)` in `src/datasets/cifar100.py`. Another option is to use a smaller patch size, e.g. 4.

## Project structure

```
main.py                     Entry point: build model, train on CIFAR-100
src/
  models/models.py          PatchEmbedding, MultiHeadAttention, encoder, VisionTransformer
  datasets/cifar100.py      CIFAR-100 train/test DataLoaders
  datasets/imagenet1k.py    Placeholder (empty)
  train.py                  Train class (train/test steps, TensorBoard)
  utils/helper.py           accuracy_fn, print_train_time
  plots.ipynb               Loads CIFAR-100 and shows a sample image
```

## Getting started

```bash
git clone https://github.com/khushwant04/ViT-Cifar100.git
cd ViT-Cifar100
pip install -r requirements.txt
python main.py
tensorboard --logdir runs
```

Run `main.py` from the repository root so that the `src` imports resolve. The script uses CUDA when available. CIFAR-100 (about 170 MB) downloads automatically on the first run. No trained weights or results are included.

## References

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009 (CIFAR-100)
