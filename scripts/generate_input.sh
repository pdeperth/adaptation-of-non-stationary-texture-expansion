#!/usr/bin/env bash
python generate_input.py --dataroot ./datasets/half/435_cnrm_full_600 --name 435_input_GAN_14x14 --which_epoch 100000 --model test --which_model_netG resnet_2x_6blocks --which_direction AtoB --dataset_mode single --norm batch --resize_or_crop none --gpu_ids 0 --how_many 150
