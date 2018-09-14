import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import requests
from PIL import Image
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from pdb import set_trace as st
from sys import argv
import argparse
import time
from math import cos, sin, pi, sqrt
import sys
import time

USE_CUDA = torch.cuda.is_available()

class InverseBatch(torch.autograd.Function):

    def forward(self, input):
        batch_size, h, w = input.size()
        assert(h == w)
        H = torch.Tensor(batch_size, h, h).type_as(input)
        for i in range(0, batch_size):
            H[i, :, :] = input[i, :, :].inverse()
        # self.save_for_backward(H)
        self.H = H
        return H

    def backward(self, grad_output):
        # print(grad_output.is_contiguous())
        # H, = self.saved_tensors
        H = self.H

        [batch_size, h, w] = H.size()
        assert(h == w)
        Hl = H.transpose(1,2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
        # print(Hl.view(batch_size, h, h, h, 1))
        Hr = H.repeat(1, h, 1).view(batch_size*h*h, 1, h)
        # print(Hr.view(batch_size, h, h, 1, h))

        r = Hl.bmm(Hr).view(batch_size, h, h, h, h) * \
            grad_output.contiguous().view(batch_size, 1, 1, h, h).expand(batch_size, h, h, h, h)
        # print(r.size())
        return -r.sum(-1).sum(-1)
        # print(r)

def InverseBatchFun(input):
    batch_size, h, w = input.size()
    assert(h == w)
    H = torch.Tensor(batch_size, h, h).type_as(input)
    for i in range(0, batch_size):
        H[i, :, :] = input[i, :, :].inverse()

    return H

class GradientBatch(nn.Module):

	def __init__(self):
		super(GradientBatch, self).__init__()
		wx = torch.FloatTensor([-.5, 0, .5]).view(1, 1, 1, 3)
		wy = torch.FloatTensor([[-.5], [0], [.5]]).view(1, 1, 3, 1)
		self.register_buffer('wx', wx)
		self.register_buffer('wy', wy)
		self.padx_func = torch.nn.ReplicationPad2d((1,1,0,0))
		self.pady_func = torch.nn.ReplicationPad2d((0,0,1,1))

	def forward(self, img):
		batch_size, k, h, w = img.size()
		img_ = img.view(batch_size * k, h, w)
		img_ = img_.unsqueeze(1)

		img_padx = self.padx_func(img_)
		img_dx = torch.nn.functional.conv2d(input=img_padx,
											weight=Variable(self.wx),
											stride=1,
											padding=0).squeeze(1)

		img_pady = self.pady_func(img_)
		img_dy = torch.nn.functional.conv2d(input=img_pady,
											weight=Variable(self.wy),
											stride=1,
											padding=0).squeeze(1)

		img_dx = img_dx.view(batch_size, k, h, w)
		img_dy = img_dy.view(batch_size, k, h, w)

		if not isinstance(img, torch.autograd.variable.Variable):
			img_dx = img_dx.data
			img_dy = img_dy.data

		return img_dx, img_dy

def normalize_img_batch(img):
	# per-channel zero-mean and unit-variance of image batch

	# img [in, Tensor N x C x H x W] : batch of images to normalize
	N, C, H, W = img.size()

	# compute per channel mean for batch, subtract from image
	img_vec = img.view(N, C, H * W, 1)
	mean = img_vec.mean(dim=2, keepdim=True)
	img_ = img - mean

	# compute per channel std dev for batch, divide img
	std_dev = img_vec.std(dim=2, keepdim=True)
	img_ = img_ / std_dev

	return img_

def warp_hmg(img, p):
	# perform warping of img batch using homography transform with batch of parameters p
	# img [in, Tensor N x C x H x W] : batch of images to warp
	# p [in, Tensor N x 8 x 1] : batch of warp parameters
	# img_warp [out, Tensor N x C x H x W] : batch of warped images
	# mask [out, Tensor N x H x W] : batch of binary masks indicating valid pixels areas

	batch_size, k, h, w = img.size()

	if isinstance(img, torch.autograd.variable.Variable):
		if USE_CUDA:
			x = Variable(torch.arange(w).cuda())
			y = Variable(torch.arange(h).cuda())
		else:
			x = Variable(torch.arange(w))
			y = Variable(torch.arange(h))
	else:
		x = torch.arange(w)
		y = torch.arange(h)

	X, Y = meshgrid(x, y)

	H = param_to_H(p)

	if isinstance(img, torch.autograd.variable.Variable):
		if USE_CUDA:
		# create xy matrix, 2 x N
			xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), Variable(torch.ones(1, X.numel()).cuda())), 0)
		else:
			xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), Variable(torch.ones(1, X.numel()))), 0)
	else:
		xy = torch.cat((X.view(1, X.numel()), Y.view(1, Y.numel()), torch.ones(1, X.numel())), 0)

	xy = xy.repeat(batch_size, 1, 1)

	xy_warp = H.bmm(xy)

	# extract warped X and Y, normalizing the homog coordinates
	X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
	Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]

	X_warp = X_warp.view(batch_size,h,w) + (w-1)/2
	Y_warp = Y_warp.view(batch_size,h,w) + (h-1)/2

	img_warp, mask = grid_bilinear_sampling(img, X_warp, Y_warp)

	return img_warp, mask

def grid_bilinear_sampling(A, x, y):
	batch_size, k, h, w = A.size()
	x_norm = x/((w-1)/2) - 1
	y_norm = y/((h-1)/2) - 1
	grid = torch.cat((x_norm.view(batch_size, h, w, 1), y_norm.view(batch_size, h, w, 1)), 3)
	Q = grid_sample(A, grid, mode='bilinear')

	if isinstance(A, torch.autograd.variable.Variable):
		if USE_CUDA:
			in_view_mask = Variable(((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data).cuda())
		else:
			in_view_mask = Variable(((x_norm.data > -1+2/w) & (x_norm.data < 1-2/w) & (y_norm.data > -1+2/h) & (y_norm.data < 1-2/h)).type_as(A.data))
	else:
		in_view_mask = ((x_norm > -1+2/w) & (x_norm < 1-2/w) & (y_norm > -1+2/h) & (y_norm < 1-2/h)).type_as(A)
		Q = Q.data

	return Q.view(batch_size, k, h, w), in_view_mask

def param_to_H(p):
	# batch parameters to batch homography
	batch_size, _, _ = p.size()

	if isinstance(p, torch.autograd.variable.Variable):
		if USE_CUDA:
			z = Variable(torch.zeros(batch_size, 1, 1).cuda())
		else:
			z = Variable(torch.zeros(batch_size, 1, 1))
	else:
		z = torch.zeros(batch_size, 1, 1)

	p_ = torch.cat((p, z), 1)

	if isinstance(p, torch.autograd.variable.Variable):
		if USE_CUDA:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1).cuda())
		else:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1))
	else:
		I = torch.eye(3,3).repeat(batch_size, 1, 1)

	H = p_.view(batch_size, 3, 3) + I

	return H

def H_to_param(H):
	# batch homography to batch parameters
	batch_size, _, _ = H.size()

	if isinstance(H, torch.autograd.variable.Variable):
		if USE_CUDA:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1).cuda())
		else:
			I = Variable(torch.eye(3,3).repeat(batch_size, 1, 1))
	else:
		I = torch.eye(3,3).repeat(batch_size, 1, 1)


	p = H - I

	p = p.view(batch_size, 9, 1)
	p = p[:, 0:8, :]

	return p

def meshgrid(x, y):
	imW = x.size(0)
	imH = y.size(0)

	x = x - x.max()/2
	y = y - y.max()/2

	X = x.unsqueeze(0).repeat(imH, 1)
	Y = y.unsqueeze(1).repeat(1, imW)
	return X, Y

class vgg16Conv(nn.Module):
	def __init__(self, model_path):
		super(vgg16Conv, self).__init__()

		print('Loading pretrained network...',end='')
		vgg16 = torch.load(model_path)
		print('done')

		self.features = nn.Sequential(
			*(list(vgg16.features.children())[0:15]),
		)

		# freeze conv1, conv2
		for p in self.parameters():
			if p.size()[0] < 256:
				p.requires_grad=False

		'''
	    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): ReLU(inplace)
	    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (3): ReLU(inplace)
	    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (6): ReLU(inplace)
	    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (8): ReLU(inplace)
	    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (11): ReLU(inplace)
	    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (13): ReLU(inplace)
	    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (15): ReLU(inplace)
	    (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (18): ReLU(inplace)
	    (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (20): ReLU(inplace)
	    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (22): ReLU(inplace)
	    (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (25): ReLU(inplace)
	    (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (27): ReLU(inplace)
	    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (29): ReLU(inplace)
	    (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    '''

	def forward(self, x):
		# print('CNN stage...',end='')
		x = self.features(x)
		# print('done')
		return x

class noPoolNet(nn.Module):
	def __init__(self, model_path):
		super(noPoolNet, self).__init__()

		print('Loading pretrained network...',end='')

		vgg16 = torch.load(model_path)

		print('done')

		vgg_features = list(vgg16.features.children())
		vgg_features[2].stride = (2,2)
		vgg_features[7].stride = (2,2)

		self.custom = nn.Sequential(
			*(vgg_features[0:4] +
				vgg_features[5:9] +
				vgg_features[10:15]),
		)

		layer = 0

		for p in self.parameters():
			if layer < 8:
				p.requires_grad = False
			
			layer += 1

	def forward(self, x):
		x = self.custom(x)
		return x

class vgg16fineTuneAll(nn.Module):
	def __init__(self, model_path):
		super(vgg16fineTuneAll, self).__init__()

		print('Loading pretrained network...',end='')
		vgg16 = torch.load(model_path)
		print('done')

		self.features = nn.Sequential(
			*(list(vgg16.features.children())[0:15]),
		)

		'''
	    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (1): ReLU(inplace)
	    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (3): ReLU(inplace)
	    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (6): ReLU(inplace)
	    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (8): ReLU(inplace)
	    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (11): ReLU(inplace)
	    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (13): ReLU(inplace)
	    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (15): ReLU(inplace)
	    (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (18): ReLU(inplace)
	    (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (20): ReLU(inplace)
	    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (22): ReLU(inplace)
	    (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (25): ReLU(inplace)
	    (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (27): ReLU(inplace)
	    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
	    (29): ReLU(inplace)
	    (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
	    '''

	def forward(self, x):
		x = self.features(x)
		return x

class custom_net(nn.Module):
	def __init__(self, model_path):
		super(custom_net, self).__init__()

		print('Loading pretrained network...',end='')
		self.custom = torch.load(model_path, map_location=lambda storage, loc: storage)
		print('done')

	def forward(self, x):
		x = self.custom(x)
		return x

class custConv(nn.Module):
	def __init__(self, model_path):
		super(custom_net, self).__init__()

		print('Loading pretrained network...',end='')
		self.custom = torch.load(model_path)
		print('done')

	def forward(self, x):
		x = self.custom(x)
		return x

class DeepLK(nn.Module):
	def __init__(self, conv_net):
		super(DeepLK, self).__init__()
		self.img_gradient_func = GradientBatch()
		self.conv_func = conv_net
		self.inv_func = InverseBatch()

	def forward(self, img, temp, init_param=None, tol=1e-3, max_itr=500, conv_flag=0, ret_itr=False):

		if conv_flag:
			start = time.time()
			Ft = self.conv_func(temp)
			stop = time.time()
			Fi = self.conv_func(img)

			# print('Feature size: '+str(Ft.size()))

		else:
			Fi = img
			Ft = temp

		batch_size, k, h, w = Ft.size()

		Ftgrad_x, Ftgrad_y = self.img_gradient_func(Ft)

		dIdp = self.compute_dIdp(Ftgrad_x, Ftgrad_y)
		dIdp_t = dIdp.transpose(1, 2)

		invH = self.inv_func(dIdp_t.bmm(dIdp))

		invH_dIdp = invH.bmm(dIdp_t)

		if USE_CUDA:
			if init_param is None:
				p = Variable(torch.zeros(batch_size, 8, 1).cuda())
			else:
				p = init_param

			dp = Variable(torch.ones(batch_size, 8, 1).cuda()) # ones so that the norm of each dp is larger than tol for first iteration
		else:
			if init_param is None:
				p = Variable(torch.zeros(batch_size, 8, 1))
			else:
				p = init_param

			dp = Variable(torch.ones(batch_size, 8, 1))

		itr = 1

		r_sq_dist_old = 0

		while (float(dp.norm(p=2,dim=1,keepdim=True).max()) > tol or itr == 1) and (itr <= max_itr):
			Fi_warp, mask = warp_hmg(Fi, p)

			mask.unsqueeze_(1)

			mask = mask.repeat(1, k, 1, 1)

			Ft_mask = Ft.mul(mask)

			r = Fi_warp - Ft_mask

			r = r.view(batch_size, k * h * w, 1)

			dp_new = invH_dIdp.bmm(r)
			dp_new[:,6:8,0] = 0

			if USE_CUDA:
				dp = (dp.norm(p=2,dim=1,keepdim=True) > tol).type(torch.FloatTensor).cuda() * dp_new
			else:
				dp = (dp.norm(p=2,dim=1,keepdim=True) > tol).type(torch.FloatTensor) * dp_new

			p = p - dp

			itr = itr + 1

		print('finished at iteration ', itr)

		if (ret_itr):
			return p, param_to_H(p), itr
		else:
			return p, param_to_H(p)

	def compute_dIdp(self, Ftgrad_x, Ftgrad_y):

		batch_size, k, h, w = Ftgrad_x.size()

		x = torch.arange(w)
		y = torch.arange(h)

		X, Y = meshgrid(x, y)

		X = X.view(X.numel(), 1)
		Y = Y.view(Y.numel(), 1)

		X = X.repeat(batch_size, k, 1)
		Y = Y.repeat(batch_size, k, 1)

		if USE_CUDA:
			X = Variable(X.cuda())
			Y = Variable(Y.cuda())
		else:
			X = Variable(X)
			Y = Variable(Y)

		Ftgrad_x = Ftgrad_x.view(batch_size, k * h * w, 1)
		Ftgrad_y = Ftgrad_y.view(batch_size, k * h * w, 1)

		dIdp = torch.cat((
			X.mul(Ftgrad_x), 
			Y.mul(Ftgrad_x),
			Ftgrad_x,
			X.mul(Ftgrad_y),
			Y.mul(Ftgrad_y),
			Ftgrad_y,
			-X.mul(X).mul(Ftgrad_x) - X.mul(Y).mul(Ftgrad_y),
			-X.mul(Y).mul(Ftgrad_x) - Y.mul(Y).mul(Ftgrad_y)),2)

		# dIdp size = batch_size x k*h*w x 8
		return dIdp

def main():
	sz = 200
	xy = [0, 0]
	sm_factor = 8

	sz_sm = int(sz/sm_factor)

	# conv_flag = int(argv[3])

	preprocess = transforms.Compose([
		transforms.ToTensor(),
	])

	img1 = Image.open(argv[1]).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
	img1_coarse = Variable(preprocess(img1.resize((sz_sm, sz_sm))))
	img1 = Variable(preprocess(img1))

	img2 = Image.open(argv[2]).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
	img2_coarse = Variable(preprocess(img2.resize((sz_sm, sz_sm))))
	img2 = Variable(preprocess(img2)) #*Variable(0.2*torch.rand(3,200,200)-1)

	transforms.ToPILImage()(img1.data).show()
	# transforms.ToPILImage()(img2.data).show()

	scale = 1.6
	angle = 15
	projective_x = 0
	projective_y = 0
	translation_x = 0
	translation_y = 0

	rad_ang = angle / 180 * pi

	p = Variable(torch.Tensor([scale + cos(rad_ang) - 2,
							   -sin(rad_ang),
							   translation_x,
							   sin(rad_ang),
							   scale + cos(rad_ang) - 2,
							   translation_y,
							   projective_x, 
							   projective_y]))
	p = p.view(8,1)
	pt = p.repeat(5,1,1)

	# p = Variable(torch.Tensor([0.4, 0, 0, 0, 0, 0, 0, 0]))
	# p = p.view(8,1)
	# pt = torch.cat((p.repeat(10,1,1), pt), 0)

	# print(p)

	dlk = DeepLK()

	img1 = img1.repeat(5,1,1,1)
	img2 = img2.repeat(5,1,1,1)
	img1_coarse = img1_coarse.repeat(5,1,1,1)
	img2_coarse = img2_coarse.repeat(5,1,1,1)

	wimg2, _ = warp_hmg(img2, H_to_param(dlk.inv_func(param_to_H(pt))))

	wimg2_coarse, _ = warp_hmg(img2_coarse, H_to_param(dlk.inv_func(param_to_H(pt))))

	transforms.ToPILImage()(wimg2[0,:,:,:].data).show()

	img1_n = normalize_img_batch(img1)
	wimg2_n = normalize_img_batch(wimg2)

	img1_coarse_n = normalize_img_batch(img1_coarse)
	wimg2_coarse_n = normalize_img_batch(wimg2_coarse)

	start = time.time()
	print('start conv...')
	p_lk_conv, H_conv = dlk(wimg2_n, img1_n, tol=1e-4, max_itr=200, conv_flag=1)
	print('conv time: ', time.time() - start)

	start = time.time()
	print('start raw...')
	p_lk, H = dlk(wimg2_coarse_n, img1_coarse_n, tol=1e-4, max_itr=200, conv_flag=0)
	print('raw time: ', time.time() - start)

	print((p_lk_conv[0,:,:]-pt[0,:,:]).norm())
	print((p_lk[0,:,:]-pt[0,:,:]).norm())
	print(H_conv)
	print(H)

	warped_back_conv, _ = warp_hmg(wimg2, p_lk_conv)
	warped_back_lk, _ = warp_hmg(wimg2, p_lk) 

	transforms.ToPILImage()(warped_back_conv[0,:,:,:].data).show()
	transforms.ToPILImage()(warped_back_lk[0,:,:,:].data).show()

	conv_loss = train.corner_loss(p_lk_conv, pt)
	lk_loss = train.corner_loss(p_lk, pt)

	pdb.set_trace()

if __name__ == "__main__":
	main()













