import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import numpy as np
from util.util import tensor2im



class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt, model_id = None):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            if model_id == 1:
                self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers[0:2]]
                print('Schedulers 1: %f' % len(self.schedulers))
            else:
                self.schedulers2 = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers[2:4]]
                print('Schedulers 2: %f' % len(self.schedulers2))

        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix, model_id)
        self.print_networks(opt.verbose)

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def test2(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward(model_id = 2)
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def update_learning_rate_2(self):
        """Update learning rates for all the networks; called at the end of every epoch"""        
        old_lr = self.optimizers[2].param_groups[0]['lr']
        for scheduler in self.schedulers2:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[2].param_groups[0]['lr']
        print('Model 2 learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals0(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    def get_current_visuals1(self):
        """Return all images to visualize: including separated fake_B frames."""
        visual_ret = OrderedDict()

        # Separar fake_B en dos imágenes si no se ha hecho aún
        if hasattr(self, 'fake_B'):
            fake_B_t, fake_B_t1 = torch.chunk(self.fake_B, 2, dim=1)
            visual_ret['fake_B_t'] = fake_B_t
            visual_ret['fake_B_t1'] = fake_B_t1

        # Añadir todas las imágenes reales según visual_names
        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)

        return visual_ret
    
    def get_current_visuals_t1(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret_t1 = OrderedDict()

        # Durante entrenamiento, solo lo básico
        if self.isTrain:
            visual_ret_t1['real2_B'] = self.real2_B
            visual_ret_t1['real2_A'] = self.real2_A
            return visual_ret_t1
        

    def get_current_visuals2(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret2 = OrderedDict()

        # Durante entrenamiento, solo lo básico
        if self.isTrain:
            # visual_ret2['real2_B'] = self.real2_B
            visual_ret2['fake_B'] = self.fake_B
            visual_ret2['fake_A'] = self.fake_A
            visual_ret2['real_A'] = self.real_A
            visual_ret2['real2_A'] = self.real2_A
            return visual_ret2

        # Durante test: usar visualización extendida
        from util.util import tensor2im
        import numpy as np

        def ensure_rgb(img):
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 2:
                third = np.zeros_like(img[:, :, 0:1])
                img = np.concatenate([img, third], axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            return img
        '''
        fake_B_t, fake_B_t1 = torch.chunk(self.fake_B, 2, dim=1)

        visual_ret2['fake_B_t'] = fake_B_t
        visual_ret2['fake_B_t1'] = fake_B_t1

        img_t = ensure_rgb(tensor2im(fake_B_t))
        img_t1 = ensure_rgb(tensor2im(fake_B_t1))

        sequence = np.concatenate([img_t, img_t1], axis=1)
        diff = np.abs(img_t1.astype(np.int16) - img_t.astype(np.int16)).astype(np.uint8)

        visual_ret2['fake_B_sequence'] = sequence
        visual_ret2['fake_B_diff'] = diff
        '''
        # resto de imágenes reales
        for name in self.visual_names2:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret2[name] = getattr(self, name)

        return visual_ret2
    
    def get_current_visuals2_2(self):
        """Return visual results, including a sequence image and frame difference."""
        def ensure_rgb(img):
            """Asegura que la imagen tiene exactamente 3 canales RGB."""
            if img.ndim == 2:  # (H, W) → (H, W, 3)
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 1:  # (H, W, 1) → (H, W, 3)
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 2:  # (H, W, 2) → (H, W, 3) con canal vacío
                third_channel = np.zeros_like(img[:, :, 0:1])
                img = np.concatenate([img, third_channel], axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]  # recortar si hay más de 3 canales
            return img

        visual_ret = OrderedDict()

        if hasattr(self, 'fake_B'):
            '''
            fake_B_t, fake_B_t1 = torch.chunk(self.fake_B, 2, dim=1)

            visual_ret['fake_B_t'] = fake_B_t
            visual_ret['fake_B_t1'] = fake_B_t1

            img_t = ensure_rgb(tensor2im(fake_B_t))
            img_t1 = ensure_rgb(tensor2im(fake_B_t1))

            sequence = np.concatenate([img_t, img_t1], axis=1)
            diff = np.abs(img_t1.astype(np.int16) - img_t.astype(np.int16)).astype(np.uint8)

            visual_ret['fake_B_sequence'] = sequence
            visual_ret['fake_B_diff'] = diff
            '''

        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)

        return visual_ret
    

    def get_current_visuals(self):
        """Return visualization images, simplified during training."""
        visual_ret = OrderedDict()

        # Durante entrenamiento, solo lo básico
        if self.isTrain:
            visual_ret['real_A'] = self.real_A
            visual_ret['real2_A'] = self.real2_A
            visual_ret['fake_B'] = self.fake_B
            visual_ret['real_B'] = self.real_B
            visual_ret['real2_B'] = self.real2_B
            return visual_ret

        # Durante test: usar visualización extendida
        from util.util import tensor2im
        import numpy as np

        def ensure_rgb(img):
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 2:
                third = np.zeros_like(img[:, :, 0:1])
                img = np.concatenate([img, third], axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            return img
        '''
        fake_B_t, fake_B_t1 = torch.chunk(self.fake_B, 2, dim=1)

        visual_ret['fake_B_t'] = fake_B_t
        visual_ret['fake_B_t1'] = fake_B_t1

        img_t = ensure_rgb(tensor2im(fake_B_t))
        img_t1 = ensure_rgb(tensor2im(fake_B_t1))

        sequence = np.concatenate([img_t, img_t1], axis=1)
        diff = np.abs(img_t1.astype(np.int16) - img_t.astype(np.int16)).astype(np.uint8)

        visual_ret['fake_B_sequence'] = sequence
        visual_ret['fake_B_diff'] = diff
        
        # resto de imágenes reales
        for name in self.visual_names:
            if isinstance(name, str) and hasattr(self, name):
                visual_ret[name] = getattr(self, name)
        '''
        ## For recurrent generation
        visual_ret['real_A'] = self.real_A
        visual_ret['fake_B'] = self.fake_B    

        ## For static generation
        '''
        visual_ret['real_A'] = self.real_A
        visual_ret['real2_A'] = self.real2_A
        visual_ret['fake_B'] = self.fake_B
        visual_ret['real_B'] = self.real_B
        visual_ret['real2_B'] = self.real2_B 
        '''
        return visual_ret


    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def get_current_losses2(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names2:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch, model_id = None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str) and model_id == 1:
                save_filename = '%s_M%s_net_%s.pth' % (epoch, model_id, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def save_networks2(self, epoch, model_id = None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names2:
            if isinstance(name, str)and model_id == 2:
                save_filename = '%s_M%s_net_%s.pth' % (epoch, model_id, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks0(self, epoch, model_id = 1):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        if model_id == 1:
            model_list = self.model_names
        elif model_id == 2:
            model_list = self.model_names2
        else:
            raise ValueError(f"Unsupported model_id={model_id}. Expected 1 or 2.")

        for name in model_list:
            load_filename = f'{epoch}_M{model_id}_net_{name}.pth'
            load_path = os.path.join(self.save_dir, load_filename)

            net = getattr(self, 'net' + name)
            print(f'Loading network {name} from {load_path}')
            
            try:
                state_dict = torch.load(load_path, map_location=self.device)
                net.load_state_dict(state_dict)
            except Exception as e:
                print(f'Error loading {name} from {load_filename}: {e}')

    def load_networks(self, epoch, model_id=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        all_names = ['G','G2']
        the_name = all_names[model_id-1]
        # print("hola %s" %the_name)
        for name in self.model_names:
            # print("adios %s" % name)
            if not model_id==None:
                # print("Se identifica modelo")
                if not name == the_name:                
                    # print("este no es")
                    continue
            # print("patata %s" % name)
            if isinstance(name, str):
                # load_filename = '%s_net_%s.pth' % (epoch, name)

                
                if name=='G':
                    load_filename = '%s_M1_net_%s.pth' % (epoch, name)
                else:   
                    load_filename = '%s_M2_net_%s.pth' % (epoch, name)
                
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
