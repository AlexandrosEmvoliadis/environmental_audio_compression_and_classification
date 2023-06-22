# Environmental Audio_Compression & Classification
Repository that contains scripts to reproduce results & more, presented on "A Robust Deep Learning based System for Environmental Audio Compression and Classification"
Both Auto-Encoder models(5-Folds) are contained in the folder **pretrained_autoencoders**

# Auto-Encoder

1) Retrieve data structure from the following link:
https://drive.google.com/drive/folders/15yfoJC5PvlbIR0AZv3BGIXcbGB9ib5Pa?usp=sharing
2) Save the folder into the root directory (i.e. EnvCompClass)
3) Run autoencoders.py to train the Auto-Encoder architectures, set use_SE to True in order to include Squeeze-and-Excitation Network. This script will store each architecture in
**autoencoders** folder
4) autoencoders.py will return PSNR, SSIM & PESQ metrics for test sets (5-fold cross validation)

# Classifier

There are 2 tested Classification Tasks
1) ESC-50 total Classification, involving:
   - ACDNet
   - SE-ACDNet
   - CNN-1D
2) Binary Classification (PR & NPR), involving :
   - ACDNet
      - On Original & Reconstructed Audio
   - SE-ACDNet
      - On Original & Reconstructed Audio
   - CNN-1D
      - On reconstructed Audio
      - On compressed representation
   Experiments on Reconstructed & Compressed representation, define the Overall System (Compression & Classification) tests
  
Run:

1) Visit https://github.com/mohaimenz/acdnet, follow instructions for data preparation.
2) Set opt.binary = True and opt.nClasses = 2 to conduct binary classification experiments
3) 

