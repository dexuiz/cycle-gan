import torch.n as nn
import torch.nn.functional as F

class Residual(nn.Module):
	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()

		conv_block = [  nn.ReflectionPad2d(1),
						nn.Conv2d(in_features, in_features, 3)
						nn.InstanceNorm2d(in_features),
						nn.Relu(inplace=True),
						nn.ReflectionPad2d(1),
						nn.Conv2d(in_features,in_features, 3)
						nn.InstanceNorm2d(in_features) ]

		self.conv_block=nn.Sequential(*conv_block)


	def forward(self,x):
		return x+self.conv_block(x)


class Generator(nn.Module):
	def __init__(self, input_nc, output_nc, n_residual=9):
		super(Generator, self).__init__()

		#encoding
		model = [ nn.ReflectionPad2d(3),
				  nn.Conv2d(input_nc,64,7),
				  nn.InstanceNorm2d(64),
				  nn.Relu(inplace=True),

				  nn.Conv2d(64,128,7),
				  nn.InstanceNorm2d(128),
				  nn.Relu(inplace=True),

				  nn.Conv2d(128,256,7),
				  nn.InstanceNorm2d(256),
				  nn.Relu(inplace=True)]

		#residual

		for _ in range(n_residual):
			model+= [Residual(256)]

		#decoding

		model+= [  nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
				   nn.InstanceNorm2d(128),
				   nn.Relu(inplace=True),

				   nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
				   nn.InstanceNorm2d(64),
				   nn.Relu(inplace=True)]

		#output 

		model+= [  nn.ReflectionPad2d(3),
				   nn.Conv2d(64, output_nc, 7),
				   nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, x);
		return self.model(x)



class Discriminator(nn.Module);
	def __init__(self, input_nc);
		super(Discriminator, self).__init__()

		model=[ nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
				nn.LeakyRelU(0.2, inplace=True),

				nn.Conv2d(64, 128, 4, stride=2, padding=1),
				nn.InstanceNorm2d(128),
				nn.LeakyRelU(0.2, inplace=True),

				nn.Conv2d(128, 256, stride=2, padding=1 ),
				nn.InstanceNorm2d(256),
				nn.LeakyRelU(0.2, inplace=True),

				nn.Conv2d(256, 512, stride=2, padding=1),
				nn.InstanceNorm2d(512),
				nn.LeakyRelU(0.2, inplace=True) ]

		model+= [nn.Conv2d(512,1,4, padding=1)]

		self.model = nn.Sequential(*model)


	def forward(self, x):
		x = self.model(x)
		return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

