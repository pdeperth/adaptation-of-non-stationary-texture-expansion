#!/usr/bin/env bash
python train_input_GAN.py --dataroot ./datasets/half/434_taue_cinq_appr --name 434_taue_input_GAN_14x14 --use_style --padding_type replicate --model input_GAN --which_model_netG resnet_2x_6blocks --which_model_netD n_layers --n_layers_D 4 --which_direction AtoB --lambda_A 100 --dataset_mode input_GAN --norm batch --resize_or_crop no --pool_size 0 --niter_decay 50000 --niter 200000 --save_epoch_freq 2000 --gpu_ids 0 --no_rotation --N_CRITIC 1 --display_freq 2000