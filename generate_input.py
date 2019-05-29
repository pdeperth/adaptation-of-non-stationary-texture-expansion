import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
from util import util
import torchvision

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = False # flip
opt.no_rotation = False
# opt.dataroot = opt.dataroot + '/tiles'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
k, data = list(enumerate(dataset))[0]
ex = data['A']
I, J, w, h = ex.size()
for i in range(opt.how_many):
    img = util.get_white_noise_image(h, w)  # l'inversion h, w est juste.
    img = img.convert('RGB')
    # print(img.size)
    conv = torchvision.transforms.ToTensor()
    img = conv(img)
    # I, J = data['A'].size()[0:2]
    for j in range(I):
        # for j in range(J):
        data['A'][j,:,:,:] = img
    # data = data.convert('RGB')
    model.set_input(data)
    # print("haha")
    # model.test()
    # visuals = model.recurrent_test()
    # visuals = model.recurrent_test_l2_searching()
    # visuals = model.stress_test_up()
    # visuals = model.stress_test_up_center(step=1, crop_size=192)
    # visuals = model.random_crop()
    visuals = model.random_crop_fineSize(opt.fineSize)
    #visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    # print(i)
    visualizer.save_images(webpage, visuals, img_path, i)

webpage.save()
