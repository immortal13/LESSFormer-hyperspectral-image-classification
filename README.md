# LESSFormer-hyperspectral-image-classification
Demo code of "LESSFormer: Local-Enhanced Spectral-Spatial Transformer for Hyperspectral Image Classification"

## Step 1: prepare dataset
**Xiong’an** (1580 × 3750 pixels): http://www.hrs-cas.com/a/share/shujuchanpin/2019/0501/1049.html

**Clipped Xiong’an** (800 × 1000 pixels, Xiong’an[300:1100, 1500:2500, :]): [百度网盘，提取码 1313](https://pan.baidu.com/s/1jD33VRXhClcdxnGt_yGBaQ?pwd=1313) or [Google Drive](https://drive.google.com/drive/folders/1G3x6f7UkczMdPdjbiNIbL2A1d-Hw1e6u?usp=drive_link)

I have organized the processing codes "HSI_dataset_processing.py" so that you can obtain your own experiment areas. 🫡🫡

**Pavia University**: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

**Houston University (withouot cloud)**: https://github.com/danfenghong/IEEE_TGRS_SpectralFormer

## Step 2: compiling cuda files
```
cd lib
. install.sh ## please wait for about 5 minutes
```
you can also refer to [ESCNet](https://github.com/Bobholamovic/ESCNet) for the compiling process.

## Step3: train and test
```
cd ..
CUDA_VISIBLE_DEVICES='7' python main.py
```

## Step3: record classification result


## Citation
If you find this work interesting in your research, please kindly cite:
```
@article{zou2022lessformer,
  title={LESSFormer: Local-enhanced spectral-spatial transformer for hyperspectral image classification},
  author={Zou, Jiaqi and He, Wei and Zhang, Hongyan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--16},
  year={2022},
  publisher={IEEE}
}
```
Thank you very much! (*^▽^*)

This code is constructed based on [vit-pytorch](https://github.com/lucidrains/vit-pytorch), [ESCNet](https://github.com/Bobholamovic/ESCNet), and [CEGCN](https://github.com/qichaoliu/CNN_Enhanced_GCN), thanks~💕.

If you have any questions, please feel free to contact me (Jiaqi Zou, immortal@whu.edu.cn).
