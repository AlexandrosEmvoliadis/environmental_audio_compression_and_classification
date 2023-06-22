# Environmental Audio_Compression & Classification
Repository that contains scripts to reproduce results presented on "A Robust Deep Learning based System for Environmental Audio Compression and Classification"
Both Auto-Encoder models(5-Folds) are contained in the folder **autoencoders**
# Auto-Encoder

1) Retrieve data structure from the following link:
https://drive.google.com/drive/folders/15yfoJC5PvlbIR0AZv3BGIXcbGB9ib5Pa?usp=sharing
2) Save the folder into the root directory (i.e. EnvCompClass)
3) Run autoencoders.py to train the Auto-Encoder architectures, set use_SE to True in order to include Squeeze-and-Excitation Network. This script will store each architecture in
**autoencoders** folder
4) autoencoders.py will return PSNR, SSIM & PESQ metrics for test sets (5-fold cross validation)


   
