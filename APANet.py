from spatial_pyramid_pooling import spp
import torch
import torch.nn as nn

class SingleAttention(nn.Module):
	"""docstring for SingleAttention"""
	def __init__(self, v0 = None):
		super(SingleAttention, self).__init__()
		self.v0 = v0

	def forward(self, F_r):
		F_nr = torch.clone(F_r)
		for i in range(F_r.shape[2]):
			F_nr[:,:,i] = F_nr[:,:,i].unsqueeze(-1).norm(dim = 2)

		a = nn.Conv2d(in_channels=F_r.shape[1], out_channels=1, kernel_size=(1,1),bias = False)
		if self.v0 != None:
			a.weight = nn.Parameter(self.v0)
		else:
			a.weight = nn.Parameter(torch.rand((1,F_r.shape[1],1,1)))
		#print(a(F_nr).shape, F_r.shape, sep = ' ')
		F_1 = a(F_nr)*F_r

		
		return F_1		
	

class APANet(nn.Module):
	"""docstring for APANet"""
	def __init__(self, scale_vector, cascade = False):
		super(APANet, self).__init__()
		self.cascade =cascade
		self.scale_vector = scale_vector

	def forward(self, x):
		x = spp(x, self.scale_vector, x.shape[2], x.shape[3])
		
		single_block = SingleAttention()
		F_1 = single_block(x)
		sum_pool = nn.AvgPool2d((F_1.shape[2],1), divisor_override=1)
		F_1 = sum_pool(F_1)
		#If want to add cascade attention block
		if self.cascade:
			#After sum pool do linear transformation in a fc
			fc = nn.Linear(in_features=F_1.shape[1], out_features=F_1.shape[1])
			v_1 = fc(F_1.view(F_1.shape[0],-1)).view(F_1.shape[0], F_1.shape[1], 1, 1)
			#Use v_1 as prior knowledge into single attention block
			single_block = SingleAttention(v0 = v_1)
			F_1 = single_block(x)
			return sum_pool(F_1)
		else:
			return F_1


