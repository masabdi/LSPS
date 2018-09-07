#!/usr/bin/env python

"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from common import prepare_snapshot_and_image_folder, write_html, write_loss
from common import get_data_loader, get_dataset
from utils import *
from trainers import *
from data import *
import sys
import torchvision
from itertools import izip
import tensorboardX
from tensorboardX import summary
from optparse import OptionParser
import time


from utils.util import Camera
import utils.util

from utils.handpose_evaluation import NYUHandposeEvaluation,ICVLHandposeEvaluation
from data.importers import NYUImporter, ICVLImporter


parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--frac', type=float, help="fraction of real labels to use", default=1.)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--log', type=str, help="log path")

MAX_EPOCHS = 100000


def visPair(di, depth, pose=None, trans=None, com=None, cube=None, ratio=None):
  img = depth[0].copy()
  img = img.reshape((128,128,1))
  img = (img+1)*127.5
  img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
  if pose is None:
      return img

  pose = pose.reshape((-1,3))
  gtorig = di.joints3DToImg((pose) * (cube[0]/2.) + com) 
  gtcrop = transformPoints2D(gtorig, trans)

  skel2 = []
  for idx, pt2 in enumerate(gtcrop):
      skel2.append((pt2[0],pt2[1]))
      cv2.circle(img, (pt2[0],pt2[1]), 2, utils.util.figColor[colorPlatte[idx]] , -1)
  for b in bones:
      pt1 = skel2[b[0]]
      pt2 = skel2[b[1]]
      color = b[2]
      cv2.line(img,pt1,pt2,color,1)
  return img


def main(argv):
  (opts, args) = parser.parse_args(argv)

  global colorPlatte, bones, Evaluation

  if 'nyu' in opts.config:
    colorPlatte = utils.util.nyuColorIdx
    bones = utils.util.nyuBones
    Evaluation = NYUHandposeEvaluation
  elif 'icvl' in opts.config:
    colorPlatte = utils.util.icvlColorIdx
    bones = utils.util.icvlBones
    Evaluation = ICVLHandposeEvaluation

  # Load experiment setting
  assert isinstance(opts, object)
  config = NetConfig(opts.config)

  batch_size = config.hyperparameters['batch_size_pose'] 
  max_iterations = 200000 #config.hyperparameters['max_iterations']
  frac = opts.frac

  dataset_a = get_dataset(config.datasets['train_a'])
  dataset_b = get_dataset(config.datasets['train_b'])
  dataset_test = get_dataset(config.datasets['test_b'])

  train_loader_a = get_data_loader(dataset_a, batch_size, shuffle=True)
  train_loader_b = get_data_loader(dataset_b, batch_size, shuffle=True)
  test_loader_real = get_data_loader(dataset_test, 1, shuffle=True)

  cmd = "trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer']
  local_dict = locals()
  exec(cmd,globals(),local_dict)
  trainer = local_dict['trainer']

  iterations = 0
  trainer.cuda(opts.gpu)

  dataset_a.pose_only = True
  dataset_b.pose_only = True

  if frac > 0. and frac < 1.:
    dataset_b.set_nmax(frac)

  di_a = dataset_a.di
  di_b = dataset_b.di

  dataset_a.sample_poses()
  dataset_b.sample_poses()

  ###################################################################
  # Setup logger and repare image outputs
  train_writer = tensorboardX.FileWriter("%s/%s" % (opts.log,os.path.splitext(os.path.basename(opts.config))[0]))
  image_directory, snapshot_directory = prepare_snapshot_and_image_folder(config.snapshot_prefix, iterations, config.image_save_iterations)


  print('using %.2f percent of the labeled real data' % frac)
  start_time = time.time()
  for ep in range(0, MAX_EPOCHS):
    for it, ((labels_a), (labels_b)) in enumerate(izip(train_loader_a,train_loader_b)):
      if labels_a.size(0) != batch_size or labels_b.size(0) != batch_size:
        continue
      labels_a = Variable(labels_a.cuda(opts.gpu))
      labels_b = Variable(labels_b.cuda(opts.gpu))
      labels = labels_a

      if frac > 0.:
        labels = torch.cat((labels_a,labels_b), 0)

      if (iterations+1) % 1000 == 0:
	trainer.vae_sch.step()

      recon_pose = trainer.vae_update(labels, config.hyperparameters)

      # Dump training stats in log file
      if (iterations+1) % config.display == 0:
	elapsed_time = time.time() -  start_time
        write_loss(iterations, max_iterations, trainer, train_writer, elapsed_time)
	start_time = time.time()

      if (iterations+1) % (10*config.image_save_iterations) == 0:
	if True:
          score, maxerr = 0, 0
          num_samples = 0
	  maxJntError = []
	  img2sav = None
    	  gt3D = []
    	  joints = []
          for tit, (test_images_b, test_labels_b, com_b, trans_b, cube_b, _) in enumerate(test_loader_real):
            test_images_b = Variable(test_images_b.cuda(opts.gpu))
            test_labels_b = Variable(test_labels_b.cuda(opts.gpu))

            pred_pose = trainer.vae.decode(trainer.vae.encode(test_labels_b)[1])

            gt3D.append(test_labels_b.data.cpu().numpy().reshape((-1, 3))*(cube_b.numpy()[0]/2.) +\
									com_b.numpy())

            joints.append(pred_pose.data.cpu().numpy().reshape((-1, 3))*(cube_b.numpy()[0]/2.) +\
									com_b.numpy())


	    if True and tit < 8:
              real_img = visPair(di_b, test_images_b.data.cpu().numpy(), test_labels_b.data.cpu().numpy(), \
				trans_b.numpy(), com_b.numpy(), cube_b.numpy(), 50.0)
              est_img = visPair(di_b, test_images_b.data.cpu().numpy(), pred_pose.data.cpu().numpy(), \
				trans_b.numpy(), com_b.numpy(), cube_b.numpy(), 50.0)

	      if img2sav is None:
	        img2sav = np.vstack((real_img,est_img))
	      else:
	        img2sav = np.hstack((img2sav,np.vstack((real_img,est_img))))


            num_samples += test_images_b.size(0)

	  cv2.imwrite(image_directory + '/_test.jpg', img2sav.astype('uint8'))
	  #maxerr = Evaluation.plotError(maxJntError, image_directory + '/maxJntError.txt')

    	  hpe = Evaluation(np.array(gt3D), np.array(joints))
    	  print("Mean error: {}mm, max error: {}mm".format(hpe.getMeanError(), hpe.getMaxError()))

      # Save network weights
      if (iterations+1) % (4*config.snapshot_save_iterations) == 0:
	  trainer.save_vae(config.snapshot_prefix, iterations, 2+frac)

      iterations += 1
      if iterations >= max_iterations:
        return

if __name__ == '__main__':
  main(sys.argv)

