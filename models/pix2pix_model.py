import torch
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt, model_id = 1):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'D_t1_real', 'D_t1_fake']
        self.loss_names2 = ['G2_GAN', 'G2_L1', 'D2_real', 'D2_fake', 'D2_t1_real', 'D2_t1_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real2_A', 'fake_B', 'real_B', 'real2_B']
        self.visual_names2 = ['real2_B', 'fake_B', 'fake_A', 'real_A', 'real2_A'] # Substitute between real or fake B (just put the selected)
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'D_t1']
            self.model_names2 = ['G2', 'D2', 'D2_t1']
        else:  # during test time, only load G
            self.model_names = ['G', 'G2']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_t1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD2_t1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN_lsgan = networks.GANLoss(gan_mode = "lsgan").to(self.device)
            self.criterionGAN_vanilla = networks.GANLoss(gan_mode = "vanilla").to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_t1 = torch.optim.Adam(self.netD_t1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2_t1 = torch.optim.Adam(self.netD_t1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D2)
            self.optimizers.append(self.optimizer_D_t1)
            self.optimizers.append(self.optimizer_D2_t1)

        self.model_id = model_id
        self.model_id = getattr(opt, 'model_id', 1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A_t' if AtoB else 'B_t'].to(self.device)
        self.real_B = input['B_t' if AtoB else 'A_t'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_2(self, input2):
        AtoB = self.opt.direction == 'AtoB'
        self.fake_B = input2['B_t'].to(self.device) # Realistic
        # self.real_B = input2['B_t'].to(self.device) # Original realistic image
        self.real_A = input2['A_t'].to(self.device) # Sketch
        self.image_paths = input2['A_paths' if AtoB else 'B_paths']

    def set_input_t1(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real2_A = input['A_t1' if AtoB else 'B_t1'].to(self.device)
        self.real2_B = input['B_t1' if AtoB else 'A_t1'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input_2_t1(self, input2_t1):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.fake_B = self.fake_B.to(self.device)
        self.real2_A = input2_t1['A_t1'].to(self.device)
        # self.image_paths = input2_t1['A_paths' if AtoB else 'B_paths']

    def forward(self, model_id = 1):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.model_id = model_id
        if self.model_id == 1:
            self.fake_B = self.netG(self.real_A)  # Generate fake_B from real_A (Main image generation)
        else:
            self.fake_B = self.fake_B.detach()
            self.fake_A = self.netG2(self.fake_B.clone())  # Generate fake_A from fake_B (Sketch generation)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN_vanilla(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN_vanilla(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_graph=True)

    def backward_D_t1(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB2 = torch.cat((self.real2_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake2 = self.netD_t1(fake_AB2.detach())
        self.loss_D_t1_fake = self.criterionGAN_vanilla(pred_fake2, False)
        # Real
        real_AB2 = torch.cat((self.real2_A, self.real2_B), 1)
        pred_real2 = self.netD_t1(real_AB2)
        self.loss_D_t1_real = self.criterionGAN_vanilla(pred_real2, True)
        # combine loss and calculate gradients
        self.loss_D_t1 = (self.loss_D_t1_fake + self.loss_D_t1_real) * 0.5
        self.loss_D_t1.backward()

    def backward_D2_t1(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_BA2 = torch.cat((self.fake_B, self.fake_A), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake2_t1 = self.netD2_t1(fake_BA2.detach())
        self.loss_D2_t1_fake = self.criterionGAN_lsgan(pred_fake2_t1, False)
        # Real
        real_BA2 = torch.cat((self.fake_B, self.real2_A), 1)
        pred_real2_t1 = self.netD2_t1(real_BA2)
        self.loss_D2_t1_real = self.criterionGAN_lsgan(pred_real2_t1, True)
        # combine loss and calculate gradients
        self.loss_D2_t1 = (self.loss_D2_t1_fake + self.loss_D2_t1_real) * 0.5
        self.loss_D2_t1.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN_vanilla(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def backward_D2(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_A
        ## With the generated image
        fake_BA = torch.cat((self.fake_B, self.fake_A), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        ## With the original image
        # fake_BA = torch.cat((self.real_B, self.fake_A), 1)

        pred_fake = self.netD2(fake_BA.detach())
        self.loss_D2_fake = self.criterionGAN_lsgan(pred_fake, False)
        # Real
        ## With the generated image
        real_BA = torch.cat((self.fake_B, self.real_A), 1)
        ## With the original image
        # real_BA = torch.cat((self.real_B, self.real_A), 1)

        pred_real = self.netD2(real_BA)
        self.loss_D2_real = self.criterionGAN_lsgan(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        # print(self.loss_D2)
        self.c = False
        self.loss_D2.backward()

    def backward_G2(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        ## With the generated image
        fake_BA = torch.cat((self.fake_B, self.fake_A), 1)
        ## With the original image
        # fake_BA = torch.cat((self.real_B, self.fake_A), 1)
        pred_fake2 = self.netD2(fake_BA)
        self.loss_G2_GAN = self.criterionGAN_vanilla(pred_fake2, True)
        # Second, G(A) = B
        self.loss_G2_L1 = self.criterionL1(self.fake_A, self.real_A) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G2 = self.loss_G2_GAN + self.loss_G2_L1
        # print(self.loss_G2)
        self.d = False
        self.loss_G2.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights

        # update D_t1
        self.set_requires_grad(self.netD_t1, True)  # enable backprop for D
        self.optimizer_D_t1.zero_grad()     # set D's gradients to zero
        self.backward_D_t1()                # calculate gradients for D
        self.optimizer_D_t1.step()          # update D's weights

    def optimize_parameters_2(self):
        ## MODEL 2
        # print("Model 2")
        self.model_id = 2
        self.forward(model_id=2)                    # uses fake_B or real_B (as selected) as input
        # update D2
        self.set_requires_grad(self.netD2, True)  # enable backprop for D
        self.optimizer_D2.zero_grad()     # set D's gradients to zero
        self.backward_D2()                # calculate gradients for D
        self.optimizer_D2.step()          # update D's weights
        # update G2
        self.set_requires_grad(self.netD2, False)  # D requires no gradients when optimizing G
        self.optimizer_G2.zero_grad()        # set G's gradients to zero
        self.backward_G2()                   # calculate graidents for G
        self.optimizer_G2.step()             # update G's weights

        # update D2_t1
        self.set_requires_grad(self.netD2_t1, True)  # enable backprop for D
        self.optimizer_D2_t1.zero_grad()     # set D's gradients to zero
        self.backward_D2_t1()                # calculate gradients for D
        self.optimizer_D2_t1.step()          # update D's weights