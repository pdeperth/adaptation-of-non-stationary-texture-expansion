# make shuffled tiles.
from options.shuffle_options import ShuffleOptions
from data.data_loader import CreateDataLoader
from util import random_tile

import time
import os
# from models.models import create_model
# from util.visualizer import Visualizer
# from util import html
# import random


opt = ShuffleOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
res_dir= opt.dataroot + '/tiles'
input_dir = opt.dataroot + '/test/'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
# model = create_model(opt)
# visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
# for i, data in enumerate(dataset):
it = 0
for filename in os.listdir(input_dir):
    random_tile.tile_function(res_dir, input_dir + filename , img_number=it, tile_num=4, tile_times=20)
    it += 1

#     model.set_input(data)
#     # print("haha")
#     # model.test()
#     # visuals = model.recurrent_test()
#     # visuals = model.recurrent_test_l2_searching()
#     # visuals = model.stress_test_up()
#     # visuals = model.stress_test_up_center(step=1, crop_size=192)
#     # visuals = model.random_crop()
#     visuals = model.random_crop_256x256()
#     #visuals = model.get_current_visuals()
#     img_path = model.get_image_paths()
#     print('process image... %s' % img_path)
#     visualizer.save_images(webpage, visuals, img_path)
#
# webpage.save()
