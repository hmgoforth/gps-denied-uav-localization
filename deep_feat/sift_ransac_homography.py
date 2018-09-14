import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace
import torch

from matplotlib import pyplot as plt

from torch.autograd import Variable

# sift = cv2.xfeatures2d.SIFT_create()

# kp1, des1 = sift.detectAndCompute(gray,None)
# kp2, des2 = sift.detectAndCompute(dst_gray,None)

# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)

# # store all the good matches as per Lowe's ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

# src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
# dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
# M_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

# # gray_kp = cv2.drawKeypoints(gray,kp1,None)
# # gray_dst_kp = cv2.drawKeypoints(dst_gray,kp2,None)

# # plt.subplot(121),plt.imshow(gray_kp),plt.title('Input')
# # plt.subplot(122),plt.imshow(gray_dst_kp),plt.title('Output')
# # plt.show()

# # set_trace()

# # kp1_loc = np.float32([kp1[i].pt for i in range(len(kp1))])
# # kp2_loc = np.float32([kp2[i].pt for i in range(len(kp2))])

# # M_found, mask = cv2.findHomography(kp1_loc, kp2_loc, cv2.RANSAC, 5.0)

# print(H_gt)
# print(M_found)

# # cv2.imshow('image',img)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()

def get_param(img_batch, template_batch, image_sz):
	template = template_batch.data.squeeze(0).cpu().numpy()
	img = img_batch.data.squeeze(0).cpu().numpy()

	if template.shape[0] == 3:
		template = np.swapaxes(template, 0, 2)
		template = np.swapaxes(template, 0, 1)
		img = np.swapaxes(img, 0, 2)
		img = np.swapaxes(img, 0, 1)

		template = (template * 255).astype('uint8')
		img = (img * 255).astype('uint8')

	# set_trace()

	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()
	# sift = cv2.SIFT()

	kp1, des1 = sift.detectAndCompute(template_gray,None)
	kp2, des2 = sift.detectAndCompute(img_gray,None)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	img3 = cv2.drawMatches(template_gray,kp1,img_gray,kp2,matches[:20],None,flags=2)
	plt.imshow(img3),plt.show()

	set_trace()


	# template_gray_with_kp = cv2.drawKeypoints(template_gray,kp1,None)
	# img_gray_with_kp = cv2.drawKeypoints(img_gray,kp2,None)
	# cv2.imshow('template',template_gray_with_kp)
	# cv2.imshow('image',img_gray_with_kp)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	if (len(kp1) >= 2) and (len(kp2) >= 2):

		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1,des2,k=2)

		# store all the good matches as per Lowe's ratio test
		good = []
		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		        good.append(m)

		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		
		src_pts = src_pts - image_sz/2
		dst_pts = dst_pts - image_sz/2

		if (src_pts.size == 0) or (dst_pts.size == 0):
			H_found = np.eye(3)
		else:
			H_found, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)

		if H_found is None:
			H_found = np.eye(3)

	else:
		H_found = np.eye(3)

	H = torch.from_numpy(H_found).float()
	I = torch.eye(3,3)

	p = H - I

	p = p.view(1, 9, 1)
	p = p[:, 0:8, :]

	if torch.cuda.is_available():
		return Variable(p.cuda())
	else:
		return Variable(p)



if __name__ == "__main__":
	img = cv2.imread('../duck.jpg')
	img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	rows,cols,ch = img_color.shape

	pts1 = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
	pts2 = np.float32([[0,0],[0,rows],[cols+200,rows],[cols+200,0]])

	H_gt = cv2.getPerspectiveTransform(pts1,pts2)

	print(H_gt)

	dst_img = cv2.warpPerspective(img_color,H_gt,(cols,rows))

	template_batch = Variable(torch.from_numpy(img_color).unsqueeze(0))
	img_batch = Variable(torch.from_numpy(dst_img).unsqueeze(0))

	print(get_param(img_batch, template_batch))











