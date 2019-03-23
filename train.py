from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import lambdaLR
from utils import Logger
from utils import weights_init_normal
from dataset import ImageDataset

import torchvision.transforms as transforms
from torch.utils.data import Dataloader
from torch.autograd import Variable
from PIL import Image
import torch
import itertools


epoch_number = 200
batchsize=1
lr=0.0002
decay=100
size=256
input_nc=3
output_nc=3


netG_A2B = Generator(input_nc,output_nc).cuda()
netG_B2A = Generator(input_nc,output_nc).cuda()

netD_A = Discriminator(input_nc).cuda()
netD_B = Discriminator(output_nc).cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netD_A.apply(weights_init_normal)

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1loss()
criterion_identity = torch.nn.L1loss()

optimizer_G = torch.optim.Adam


