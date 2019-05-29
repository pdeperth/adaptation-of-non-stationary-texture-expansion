#!/usr/bin/env bash
python test_recurrent_several.py --dataroot ./datasets/half/435_cnrm_full_600 --name 435_cnrm_full_600_half_style_14x14 --results_dir ./results/from_generated/ --which_epoch 350000 --model test --which_model_netG resnet_2x_6blocks --which_direction AtoB --dataset_mode single --norm batch --resize_or_crop none --gpu_ids 0,1 --how_many=100
