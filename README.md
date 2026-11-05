# FedSRBF: Federated Scribble-Guided Medical Image Segmentation with Region-Boundary Diffusion Fusion
This is the test code for our ISBI 2027 submitted paper "FedSRBF: Federated Scribble-Guided Medical Image Segmentation with Region-Boundary Diffusion Fusion".

> Abstract: Medical images collected across institutions often differ in appearance and target characteristics, making it difficult for federated models to learn consistent segmentation features. We propose a sparse-guided dual-branch federated segmentation framework that jointly learns region and boundary representations. In our model, scribbles condition a shared encoder, whose features feed two parallel decoders for region segmentation and boundary learning. The boundary decoder branch incorporates a feature-space diffusion to recover intermediate features through a single denoising evaluation. A learned spatial gate selectively uses these features to correct the region prediction. The network is trained jointly using dense mask supervision, while Model Alignment and
Reference-Magnitude Consistency (MARC) aggregation weights client models according to parameter similarity and update-magnitude agreement. Experiments on polyp and breast tumor segmentation show
higher Dice scores and lower HD95 than the compared federated methods.
> 
## Dependencies
- Python 3.10
- PyTorch 2.13
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
- `conda create -n FedSRBF python=3.10`
- `conda activate FedSRBF`
- `pip install -r requirements.txt`

## Testing
- Download the datasets folder from [GoogleDrive](https://drive.google.com/file/d/1JtwsvKA5viT3WwE15wlsvUph9uMwYDXQ/view?usp=sharing) and place them in the FedSRBF project directory.

- The folder structure within `FedSRBF/datasets` should be organized as follows:
```
datasets/
├── B1/
│   ├── test/
│   │   ├── image/
│   │   └── mask/
├── B2/
├── B3/
├── B4/

```

- Test the segmentation performance of Breast tumor dataset

```
$ python test.py  --model model.pth  --data-root /path/to/the/folder/datasets
```


