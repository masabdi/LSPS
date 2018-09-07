"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import sys
if sys.version_info[0] >= 3:
  import _pickle as cPickle
else:
  import cPickle
import gzip
import cv2
import os, gc
import numpy as np
import torch.utils.data as data
import torch
import urllib

from data.importers import NYUImporter, ICVLImporter, MSRA15Importer
from data.dataset import NYUDataset, ICVLDataset, MSRA15Dataset
from data.basetypes import DepthFrame
from data.transformations import transformPoints2D
from utils.handdetector import HandDetector


def normalize(img, com, cube):
  img[img == 0] = com[2] + (cube[2] / 2.)
  img -= com[2]
  img /= (cube[2] / 2.)
  return img


def augmentCrop(img, gt3Dcrop, com, cube, M, aug_modes, hd, normZeroOne=False, sigma_com=None,
                sigma_sc=None, rot_range=None, rng=None):

    """
    Commonly used function to augment hand poses
    :param img: image
    :param gt3Dcrop: 3D annotations
    :param com: center of mass in image coordinates (x,y,z)
    :param cube: cube
    :param aug_modes: augmentation modes
    :param hd: hand detector
    :param normZeroOne: normalization
    :param sigma_com: sigma of com noise
    :param sigma_sc: sigma of scale noise
    :param rot_range: rotation range in degrees
    :return: image, 3D annotations, com, cube
    """
    #print(img.shape)
    assert len(img.shape) == 2
    assert isinstance(aug_modes, list)

    if sigma_com is None:
        sigma_com = 10.

    if sigma_sc is None:
        sigma_sc = 0.05

    if rot_range is None:
        rot_range = 180.

    if normZeroOne is True:
        img = img * cube[2] + (com[2] - (cube[2] / 2.))
    else:
        img = img * (cube[2] / 2.) + com[2]
    premax = img.max()

    mode = rng.randint(0, len(aug_modes))
    off = rng.randn(3) * sigma_com  # +-px/mm
    rot = rng.uniform(-rot_range, rot_range)
    sc = abs(1. + rng.randn() * sigma_sc)
    if aug_modes[mode] == 'com':
	#print('aug com', off)
        rot = 0.
        sc = 1.
        imgD, new_joints3D, com, M = hd.moveCoM(img.astype('float32'), cube, \
				com, off, gt3Dcrop, M, pad_value=0)
        curLabel = new_joints3D / (cube[2] / 2.)
    elif aug_modes[mode] == 'rot':
	#print('aug rot', rot)
        off = np.zeros((3,))
        sc = 1.
        imgD, new_joints3D, rot = hd.rotateHand(img.astype('float32'), cube, \
				com, rot, gt3Dcrop, pad_value=0)
        curLabel = new_joints3D / (cube[2] / 2.)
    elif aug_modes[mode] == 'sc':
        off = np.zeros((3,))
        rot = 0.
        imgD, new_joints3D, cube, M = hd.scaleHand(img.astype('float32'), cube, \
				com, sc, gt3Dcrop, M, pad_value=0)
        curLabel = new_joints3D / (cube[2] / 2.)
    elif aug_modes[mode] == 'none':
        off = np.zeros((3,))
        sc = 1.
        rot = 0.
        imgD = img
        curLabel = gt3Dcrop / (cube[2] / 2.)
    else:
        raise NotImplementedError()

    if normZeroOne is True:
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= (com[2] - (cube[2] / 2.))
        imgD /= cube[2]
    else:
        imgD[imgD == premax] = com[2] + (cube[2] / 2.)
        imgD[imgD == 0] = com[2] + (cube[2] / 2.)
        imgD[imgD >= com[2] + (cube[2] / 2.)] = com[2] + (cube[2] / 2.)
        imgD[imgD <= com[2] - (cube[2] / 2.)] = com[2] - (cube[2] / 2.)
        imgD -= com[2]
        imgD /= (cube[2] / 2.)


    return imgD, None, curLabel, np.asarray(cube), com, np.array(M, dtype='float32'), rot

#####################################################################################
##################################### ICVL ##########################################
#####################################################################################

class dataset_hand_ICVL(data.Dataset):
  def __init__(self, specs):
    seed = specs['seed']
    root = specs['root']
    subset = specs['subset']
    docom = specs['docom'] 

    self.rng = np.random.RandomState(seed)
    self.sampled_poses = None
    self.pose_only = False
    self.nmax = np.inf
    self.augment = specs['augment']
    self.num_sample_poses = specs['sample_poses']
    self.joint_subset = specs['joint_subset']
    self.aug_modes = [ 'none', 'com', 'rot' ]  
    print("create data")

    self.di = ICVLImporter(root, cacheDir='../../cache/')

    self.Seq = self.di.loadSequence(subset, ['0'], rng=self.rng, shuffle=True, docom=docom)

    #print(self.Seq.data[0].gt3Dcrop)
    #self.di.showAnnotatedDepth(self.Seq.data[0])

    # create training data
    cube = np.asarray(self.Seq.config['cube'], 'float32')
    com = np.asarray(self.Seq.data[0].com, 'float32')
    img = np.asarray(self.Seq.data[0].dpt.copy(), 'float32')
    img = normalize(img, com, cube)

    self.hd = HandDetector(img, abs(self.di.fx), abs(self.di.fy), importer=self.di)
    self.num = len(self.Seq.data)
    print(' data loaded with %d samples' % self.num)

  def sample_poses(self):
    train_cube = np.asarray([self.Seq.config['cube']]*self.num, dtype='float32')
    train_com = np.asarray([d.com for d in self.Seq.data], dtype='float32')
    train_gt3D = np.asarray([d.gt3Dcrop for d in self.Seq.data], dtype='float32')

    self.sampled_poses = self.hd.sampleRandomPoses(self.di, self.rng, train_gt3D, train_com,\
	train_cube, self.num_sample_poses, self.nmax, \
	self.aug_modes).reshape((-1, train_gt3D.shape[1]*3))
    self.num = self.sampled_poses.shape[0]
    self.nmax = self.sampled_poses.shape[0]
    print('%d sample poses created!' % self.num)

  def __getitem__(self, i):

    if self.pose_only and self.sampled_poses is not None:
	return self.sampled_poses[i]

    cube = np.asarray(self.Seq.config['cube'], 'float32')
    com = np.asarray(self.Seq.data[i].com, 'float32')
    M = np.asarray(self.Seq.data[i].T, dtype='float32')
    gt3D = np.asarray(self.Seq.data[i].gt3Dcrop, dtype='float32') 
    img = np.asarray(self.Seq.data[i].dpt.copy(), 'float32')
    img = normalize(img, com, cube)




    if not self.augment:
      if self.pose_only:
	return gt3D.flatten() / (cube[2] / 2.)
      #print(img.shape, gt3D.flatten().shape, com.shape, M.shape, cube.shape)
      return np.expand_dims(img, axis=0), gt3D.flatten() / (cube[2] / 2.), com, M, cube

    img, _, gt3D, cube, com2D, M, _ = augmentCrop(img, gt3D, \
	self.di.joint3DToImg(com), cube, M, self.aug_modes, self.hd, rng=self.rng)


    if self.pose_only:
	return gt3D.flatten()

    #print(imgD.shape, gt3Dn.flatten().shape, com.shape, M.shape, cube.shape)
    return np.expand_dims(img, axis=0), gt3D.flatten() , self.di.jointImgTo3D(com2D), M, cube

  def set_nmax(self, frac):
    self.nmax = int(self.num*frac)
    print('self.nmax %d' % self.nmax)

  def __len__(self):
    return np.minimum(self.num, self.nmax)


class dataset_hand_ICVL_test(dataset_hand_ICVL):
  def __init__(self, specs):
    seed = specs['seed']
    root = specs['root']
    subset = specs['subset']
    docom = specs['docom'] 
    print("create data")

    self.rng = np.random.RandomState(seed)
    self.di = ICVLImporter(root, refineNet=None, cacheDir='../../cache/')

    self.Seq1 = self.di.loadSequence(subset, docom=docom)
    self.Seq2 = self.di.loadSequence(subset.replace('1', '2'), docom=docom)

    self.num = len(self.Seq1.data) + len(self.Seq2.data)
    print(' data loaded with %d samples' % self.num)

    self.len_seq1 = len(self.Seq1.data)

  def __getitem__(self, i):

    if i < self.len_seq1:
      cube = np.asarray(self.Seq1.config['cube'], 'float32')
      com = np.asarray(self.Seq1.data[i].com, 'float32')
      M = np.asarray(self.Seq1.data[i].T, dtype='float32')
      gt3D = np.asarray(self.Seq1.data[i].gt3Dcrop, dtype='float32') 
      img = np.asarray(self.Seq1.data[i].dpt.copy(), 'float32')
    else:
      cube = np.asarray(self.Seq2.config['cube'], 'float32')
      com = np.asarray(self.Seq2.data[i-self.len_seq1].com, 'float32')
      M = np.asarray(self.Seq2.data[i-self.len_seq1].T, dtype='float32')
      gt3D = np.asarray(self.Seq2.data[i-self.len_seq1].gt3Dcrop, dtype='float32') 
      img = np.asarray(self.Seq2.data[i-self.len_seq1].dpt.copy(), 'float32')

    img = normalize(img, com, cube)

    return np.expand_dims(img, axis=0), gt3D.flatten() / (cube[2] / 2.), com, M, cube

  def __len__(self):
    return self.num


#####################################################################################
##################################### NYU ##########################################
#####################################################################################

class dataset_hand_NYU(data.Dataset):
  def __init__(self, specs):

    seed = specs['seed']
    root = specs['root']
    subset = specs['subset']
    docom = specs['docom'] 

    self.rng = np.random.RandomState(seed)
    self.sampled_poses = None
    self.pose_only = False
    self.nmax = np.inf
    self.augment = specs['augment']
    self.num_sample_poses = specs['sample_poses']
    self.joint_subset = specs['joint_subset']
    self.aug_modes = [ 'none', 'com', 'rot' ]  
    print("create data")

    self.flip_y = False
    com_idx = 32
    cube_size = 300

    if 'MSRA' in self.joint_subset:
      self.joint_subset = np.asarray([29, 23, 22, 20, 18, 17, 16, 14, 12, 11, 10,\
					8, 6, 5, 4, 2, 0, 28, 27, 25, 24], dtype='int32')
      com_idx = 17
    elif 'ICVL' in self.joint_subset:
      self.joint_subset = np.asarray([34, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,\
					8, 6, 4, 2, 0], dtype='int32')
      self.flip_y = True
      com_idx = 34
      cube_size = 350
    else:
      self.joint_subset = np.arange(36)

    self.di = NYUImporter(root, refineNet=None, allJoints=True, com_idx=com_idx, cacheDir='../../cache/')

    if 'synth' in subset:
      self.di.default_cubes[subset] = (cube_size, cube_size, cube_size)
      print(self.di.default_cubes[subset])

    self.Seq = self.di.loadSequence(subset, rng=self.rng, shuffle=True, docom=docom)

    #print(self.Seq.data[0].gt3Dcrop)
    #self.di.showAnnotatedDepth(self.Seq.data[0])

    #print('joint_subset', self.joint_subset)

    # create training data
    cube = np.asarray(self.Seq.config['cube'], 'float32')
    com = np.asarray(self.Seq.data[0].com, 'float32')
    img = np.asarray(self.Seq.data[0].dpt.copy(), 'float32')
    img = normalize(img, com, cube)

    self.hd = HandDetector(img, abs(self.di.fx), abs(self.di.fy), importer=self.di)
    self.num = len(self.Seq.data)
    print(' data loaded with %d samples' % self.num)

  def sample_poses(self):
    train_cube = np.asarray([self.Seq.config['cube']]*self.num, dtype='float32')
    train_com = np.asarray([d.com for d in self.Seq.data], dtype='float32')
    train_gt3D = np.asarray([d.gt3Dcrop for d in self.Seq.data], dtype='float32')

    self.sampled_poses = self.hd.sampleRandomPoses(self.di, self.rng, train_gt3D, train_com,\
	train_cube, self.num_sample_poses, self.nmax, \
	self.aug_modes)#.reshape((-1, train_gt3D.shape[1]*3))
    self.num = self.sampled_poses.shape[0]
    self.nmax = self.sampled_poses.shape[0]
    print('%d sample poses created!' % self.num)


  def __getitem__(self, i):

    if self.pose_only and self.sampled_poses is not None:
	pos = self.sampled_poses[i][self.joint_subset]
    	if self.flip_y:
	  pos[:,1] *= -1
	return pos.flatten()

    cube = np.asarray(self.Seq.config['cube'], 'float32')
    com = np.asarray(self.Seq.data[i].com, 'float32')
    M = np.asarray(self.Seq.data[i].T, dtype='float32')
    gt3D = np.asarray(self.Seq.data[i].gt3Dcrop, dtype='float32') 
    img = np.asarray(self.Seq.data[i].dpt.copy(), 'float32')
    img = normalize(img, com, cube)

    if not self.augment:
      if self.joint_subset is not None:
	gt3D = gt3D[self.joint_subset]
      if self.flip_y:
	gt3D[:,1] *= -1

      if self.pose_only:
	return gt3D.flatten() / (cube[2] / 2.)

      #print(img.shape, gt3D.flatten().shape, com.shape, M.shape, cube.shape)
      return np.expand_dims(img, axis=0), gt3D.flatten() / (cube[2] / 2.), com, M, cube, cube

    img, _, gt3D, cube, com2D, M, _ = augmentCrop(img, gt3D, \
	self.di.joint3DToImg(com), cube, M, self.aug_modes, self.hd, rng=self.rng)

    if self.joint_subset is not None:
	gt3D = gt3D[self.joint_subset]
    if self.flip_y:
	gt3D[:,1] *= -1

    if self.pose_only:
	return gt3D.flatten()

    #print(imgD.shape, gt3Dn.flatten().shape, com.shape, M.shape, cube.shape)
    return np.expand_dims(img, axis=0), gt3D.flatten() , self.di.jointImgTo3D(com2D), M, cube, cube

  def set_nmax(self, frac):
    self.nmax = int(self.num*frac)
    print('self.nmax %d' % self.nmax)

  def __len__(self):
    return np.minimum(self.num, self.nmax)



class dataset_hand_NYU_test(dataset_hand_NYU):
  def __init__(self, specs):
    seed = specs['seed']
    root = specs['root']
    subset = specs['subset']
    docom = specs['docom'] 
    print("create data")

    self.rng = np.random.RandomState(seed)
    self.di = NYUImporter(root, refineNet=None, allJoints=True, cacheDir='../../cache/')

    self.Seq = self.di.loadSequence(subset, shuffle=False, rng=self.rng, docom=docom)
    self.num = len(self.Seq.data)
    print(' data loaded with %d samples' % self.num)

  def __getitem__(self, i):


    cube = np.asarray(self.Seq.config['cube'], 'float32')
    com = np.asarray(self.Seq.data[i].com, 'float32')
    M = np.asarray(self.Seq.data[i].T, dtype='float32')
    gt3D = np.asarray(self.Seq.data[i].gt3Dcrop, dtype='float32') 
    img = np.asarray(self.Seq.data[i].dpt.copy(), 'float32')
    img = normalize(img, com, cube)
    if False:
	dpt = self.di.loadDepthMap(self.Seq.data[i].fileName)
	#print(dpt.shape)


    return np.expand_dims(img, axis=0), gt3D.flatten() / (cube[2] / 2.), com, M, cube, cube#, self.Seq.data[i].fileName

    return np.expand_dims(img, axis=0), gt3D.flatten(), com, M, cube


  def __len__(self):
    return self.num



