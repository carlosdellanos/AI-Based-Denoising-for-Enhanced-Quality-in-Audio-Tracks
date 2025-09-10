This repository contains the complete codebase for denoising audio tracks using a U-Net implementation.
It includes all the necessary steps to process raw audio, prepare datasets, train the neural network, and reconstruct denoised tracks.

FEATURES:
- Dataset preparation: split, rename, and normalize audio files.

- Noise modeling:
    Additive white noise
    Quantization noise
    Dithered quantization noise
    Recorded noise

- U-Net model for spectrogram-based denoising.

- Training & evaluation pipeline with metrics:
    Loss functions (MSE, SmoothL1, PSNR)
    PSNR computation
    R² score evaluation

 - Audio reconstruction from spectrograms using Griffin-Lim.

 - Visualization tools: spectrogram comparison plots (clean, noisy, denoised).

WORKFLOW FOR TRAINING THE MODEL
1. Prepare dataset
    Place raw audio files in data/raw_data
    Run the dataset split/rename script
    Compute statistics

2. (Optional) Calculate mean and standard deviation for normalization
    In case this step is conducted, values should be changed in the variables for stats in dataset.py, reconstruct_audio and the rest of scripts
   
4. Train and test the model
    Train and test the U-Net model with your chosen noise transform with train_and_test.py
    (Best model weights are saved in best_model.pth)
    Noisy and denoised spectrograms are saved
    Loss functions: MSE, SmoothL1, PSNR-based are implemented
    Train loss, validation loss and validation R2score plots are shown when training ends
    Metrics are shown after each epoch and after testing

6. Reconstruct audio
   Use reconstruct_audio.py to reconstruct all tracks, or reconstruct_one_audio to reconstruct a specific one
   (Spectrograms are converted back to .wav with Griffin-Lim)
   Optionally join audio segments into full-length tracks using construct_track.py

7. Visualize
   Compare clean, noisy, and denoised spectrograms in .svg format using compute_one_spec and plot_spectrograms.py


