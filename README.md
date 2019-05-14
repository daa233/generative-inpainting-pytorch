# generative-inpainting-pytorch
A PyTorch reimplementation for paper Generative Image Inpainting with Contextual Attention (https://arxiv.org/abs/1801.07892)


Converted TF model: [[Link](https://drive.google.com/file/d/1vz2Qp12_iwOiuvLWspLHrC1UIuhSLojx/view?usp=sharing)]


## Inpainting Example:

```bash
python test_tf_model.py \
	--image examples/imagenet/imagenet_patches_ILSVRC2012_val_00008210_input.png \
	--mask examples/center_mask_256.png \
	--output examples/output.png \
	--model-path torch_model.p
```
