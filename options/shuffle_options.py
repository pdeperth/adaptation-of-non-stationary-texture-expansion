import argparse
import os
from util import util
import torch

class ShuffleOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--results_dir', type=str, default='./datasets/half/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='tiles', help='train, val, test, etc')
        self.parser.add_argument('--how_many', type=int, default=5, help='how many shuffled images to make from each original image')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--no_rotation', action='store_true', help='if specified, do not rotate the images for data augmentation')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | half_crop | ... etc]')
        self.parser.add_argument('--model', type=str, default='test', help='chooses which model to use. cycle_gan, pix2pix, test, etc')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.isTrain = False

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()):
        #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('-------------- End ----------------\n')
        return self.opt
