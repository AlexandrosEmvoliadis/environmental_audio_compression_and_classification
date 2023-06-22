# Environmental Audio_Compression & Classification
Repository that contains scripts to reproduce results & more, presented on "A Robust Deep Learning based System for Environmental Audio Compression and Classification, Audio Engineering Society, 154 (May,2023)"
Both Auto-Encoder models (5-Folds) are contained in the folder **pretrained_autoencoders**

# Auto-Encoder

1) Retrieve data structure from the following link:
https://drive.google.com/drive/folders/15yfoJC5PvlbIR0AZv3BGIXcbGB9ib5Pa?usp=sharing
2) Save the folder into the root directory (EnvCompClass)
3) Run autoencoders.py to train the Auto-Encoder architectures, set use_SE to True in order to include Squeeze-and-Excitation Network. This script will store each architecture in **autoencoders** folder. Script will also return validation metrics (PSNR, SSIM & PESQ) per folder and total (Mean & Std Deviation)

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
  
# Run

1) Visit https://github.com/mohaimenz/acdnet, follow instructions for data preparation.
2) In opts.py, set opt.binary = True, opt.nClasses = 2 to conduct binary classification experiments. In val_generator.py, set opt.batchsize = 1600 or lower
3) For Original audio as input:
   - In ./torch/trainer.py, exclude enc model (comment) and make sure to remove (comment out) enc model from training & validation processes
   - Using ACDNet and SE-ACDNet:
     - Reshape the input to match(Batch_size,1,1,22050)
4) For Reconstructed audio as input:
   - Include enc model. Specify model's path in opts.py (there's an already example)
   - Using CAE as Auto-Encoder:
      - in ./torch/resources/models.py go to get_ae and remove the lines referring to squeeze-and-excitation networks
5) For Compressed audio as input:
   - in ./torch/resources/models.py specify as output of the auto-encoder the bottleneck output
Don't forget to include the autoencoder in the trainer.py __validate function and, if using (SE)ACDNet, reshape to match models' input
6) Run ./torch/tester.py following the screen-instructions to test models' performance
      
