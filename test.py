"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    sketch_model = create_model(opt)
    model.setup(opt, model_id = 1)    # regular setup: load and print networks; create schedulers
    sketch_model.setup(opt, model_id = 2)
    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir_real = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    web_dir_sketch = os.path.join(opt.results_dir, opt.name + '_sketch', '{}_{}'.format(opt.phase, opt.epoch))

    if opt.load_iter > 0:
        web_dir_real = '{:s}_iter{:d}'.format(web_dir_real, opt.load_iter)
        web_dir_sketch = '{:s}_iter{:d}'.format(web_dir_sketch, opt.load_iter)

    print('Creating web directories:')
    print(f'Real images: {web_dir_real}')
    print(f'Sketch images: {web_dir_sketch}')
    webpage_real = html.HTML(web_dir_real, 'Experiment M1 = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    webpage_sketch = html.HTML(web_dir_sketch, 'Experiment M2 = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
        sketch_model.eval2()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        # Model 1
        model.set_input(data)  # unpack data from data loader
        model.set_input_t1(data)
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths

        img_gen = model.get_current_visuals()  # Get generated images
        img_gen2 = model.get_current_visuals2()
        fake_B = img_gen["fake_B"].detach()
        real_A = img_gen["real_A"].detach()
        # real_B = img_gen["real_B"].detach()
        real2_A = img_gen2["real2_A"].detach()


        # Model 2
        sketch_model.real_A = model.real_A
        sketch_model.fake_B = model.fake_B
        # sketch_model.real_B = model.real_B
        ## With the generated image
        sketch_input = {
            "A_t": real_A,      
            "B_t": fake_B, # Use synthetic image as input for sketch model
            "A_paths": data["A_paths"],
            "B_paths": fake_B,
        }
        '''
        ## With the original image
        sketch_input = {
            "A": real_A,      
            "B": real_B, # Use original image as input for sketch model
            "A_paths": data["A"],
            "B_paths": data["B"],
        }
        '''
        sketch_input_t1 = {
                "A_t1": real2_A,      
                "B_t1": fake_B, # Use synthetic image as input for sketch model
                "A_paths": data["A_paths"],
                "B_paths": fake_B,
            }
        sketch_model.set_input_2(sketch_input)
        sketch_model.set_input_2_t1(sketch_input_t1)
        sketch_model.test2()

        img_path_sketch = img_path
        visuals_sketch = sketch_model.get_current_visuals2()

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage_real, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        save_images(webpage_sketch, visuals_sketch, img_path_sketch, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

    webpage_real.save()  # save the HTML
    webpage_sketch.save()
