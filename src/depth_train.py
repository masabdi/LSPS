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
import cv2
import torchvision
from itertools import izip
import tensorboardX
from tensorboardX import summary
from optparse import OptionParser
import time

from utils.util import Camera
import utils.util
from utils.handpose_evaluation import NYUHandposeEvaluation, ICVLHandposeEvaluation


parser = OptionParser()
parser.add_option('--gpu', type=int, help="gpu id", default=0)
parser.add_option('--resume', type=int, help="resume training?", default=0)
parser.add_option('--frac', type=float,
                  help="fraction of real labels to use", default=1.)
parser.add_option('--idx', type=int, help="idx predtrain", default=-1)
parser.add_option('--config', type=str, help="net configuration")
parser.add_option('--mode', type=str, help="pretrain/estimate")

parser.add_option('--log', type=str, help="log path")

MAX_EPOCHS = 100000


def visPair(di, depth, pose=None, trans=None, com=None, cube=None, ratio=None):
    img = depth[0].copy()
    img = img.reshape((128, 128, 1))
    img = (img+1)*127.5
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    if pose is None:
        return img

    pose = pose.reshape((-1, 3))
    gtorig = di.joints3DToImg((pose) * (cube[0]/2.) + com)
    gtcrop = transformPoints2D(gtorig, trans)

    skel2 = []
    for idx, pt2 in enumerate(gtcrop):
        skel2.append((pt2[0], pt2[1]))
        cv2.circle(img, (pt2[0], pt2[1]), 2,
                   utils.util.figColor[colorPlatte[idx]], -1)
    if len(skel2) > 1:
        for b in bones:
            pt1 = skel2[b[0]]
            pt2 = skel2[b[1]]
            color = b[2]
            cv2.line(img, pt1, pt2, color, 1)
    return img


def main(argv):
    (opts, args) = parser.parse_args(argv)
    if 'estimate' in opts.mode:
        mode_idx = int(opts.mode[-1])

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

    batch_size = config.hyperparameters['batch_size'] if 'estimate' in opts.mode else 1
    test_batch_size = batch_size * 32
    max_iterations = config.hyperparameters['max_iterations']
    frac = opts.frac

    dataset_a = get_dataset(config.datasets['train_a'])
    dataset_b = get_dataset(config.datasets['train_b'])
    dataset_test = get_dataset(config.datasets['test_b'])

    train_loader_a = get_data_loader(dataset_a, batch_size, shuffle=True)
    train_loader_b = get_data_loader(dataset_b, batch_size, shuffle=True)
    test_loader_real = get_data_loader(
        dataset_test, test_batch_size, shuffle=False)

    cmd = "trainer=%s(config.hyperparameters)" % config.hyperparameters['trainer']
    local_dict = locals()
    exec(cmd, globals(), local_dict)
    trainer = local_dict['trainer']

    di_a = dataset_a.di
    di_b = dataset_b.di

    # Check if resume training
    iterations = 0
    if opts.resume == 1:
        iterations = trainer.resume(
            config.snapshot_prefix, idx=-1, load_opt=True)
        for i in range(iterations//1000):
            trainer.dis_sch.step()
            trainer.gen_sch.step()
    trainer.cuda(opts.gpu)

    print('using %.2f percent of the labeled real data' % frac)
    try:
        if 'estimate' in opts.mode and (mode_idx == 3 or mode_idx == 4):
            trainer.load_vae(config.snapshot_prefix, 2+frac)
        else:
            trainer.load_vae(config.snapshot_prefix, frac)
    except:
        print('Failed to load the parameters of vae')

    if 'estimate' in opts.mode:
        if opts.idx != 0:
            trainer.resume(config.snapshot_prefix,
                           idx=opts.idx, est=mode_idx == 5)
        if frac > 0. and frac < 1.:
            dataset_b.set_nmax(frac)
        # trainer.dis.freeze_layers()

    ###############################################################################################
    # Setup logger and repare image outputs
    train_writer = tensorboardX.FileWriter(
        "%s/%s" % (opts.log, os.path.splitext(os.path.basename(opts.config))[0]))
    image_directory, snapshot_directory = prepare_snapshot_and_image_folder(
        config.snapshot_prefix, iterations, config.image_save_iterations)

    best_err, best_acc = 100., 0.
    start_time = time.time()
    for ep in range(0, MAX_EPOCHS):
        for it, ((images_a, labels_a, com_a, M_a, cube_a, _), (images_b, labels_b, com_b, M_b, cube_b, _)) in \
                enumerate(izip(train_loader_a, train_loader_b)):
            if images_a.size(0) != batch_size or images_b.size(0) != batch_size:
                continue
            images_a = Variable(images_a.cuda(opts.gpu))
            images_b = Variable(images_b.cuda(opts.gpu))
            labels_a = Variable(labels_a.cuda(opts.gpu))
            labels_b = Variable(labels_b.cuda(opts.gpu))
            com_a = Variable(com_a.cuda(opts.gpu))
            com_b = Variable(com_b.cuda(opts.gpu))

            trainer.dis.train()
            if opts.mode == 'pretrain':
                if (iterations+1) % 1000 == 0:
                    trainer.dis_sch.step()
                    trainer.gen_sch.step()
                    print('lr %.8f' % trainer.dis_sch.get_lr()[0])

                trainer.dis_update(images_a, labels_a, images_b,
                                   labels_b, com_a, com_b, config.hyperparameters)
                image_outputs = trainer.gen_update(
                    images_a, labels_a, images_b, labels_b, config.hyperparameters)
                assembled_images = trainer.assemble_outputs(
                    images_a, images_b, image_outputs)
            else:
                if (iterations+1) % 100 == 0:
                    trainer.dis_sch.step()
                    image_outputs = trainer.post_update(
                        images_a, labels_a, images_b, labels_b, com_a, com_b, mode_idx, config.hyperparameters)
                assembled_images = trainer.assemble_outputs(
                    images_a, images_b, image_outputs)

            # Dump training stats in log file
            if (iterations+1) % config.display == 0:
                elapsed_time = time.time() - start_time
                write_loss(iterations, max_iterations,
                           trainer, train_writer, elapsed_time)
                start_time = time.time()

            if (iterations + 1) % config.image_display_iterations == 0:
                img_filename = '%s/gen.jpg' % (image_directory)
                torchvision.utils.save_image(
                    assembled_images.data / 2 + 0.5, img_filename, nrow=1)

            if (iterations+1) % config.image_save_iterations == 0:

                # and (iterations+1) % (2*config.image_save_iterations) != 0:
                if opts.mode == 'pretrain':
                    img_filename = '%s/gen_%08d.jpg' % (
                        image_directory, iterations + 1)
                    torchvision.utils.save_image(
                        assembled_images.data / 2 + 0.5, img_filename, nrow=1)
                    write_html(snapshot_directory + "/index.html", iterations + 1,
                               config.image_save_iterations, image_directory)
                else:
                    trainer.dis.eval()
                    score, maxerr = 0, 0
                    num_samples = 0
                    maxJntError = []
                    meanJntError = 0
                    img2sav = None
                    gt3D = []
                    joints = []
                    joints_imgcord = []
                    codec = cv2.VideoWriter_fourcc(*'XVID')
                    vid = cv2.VideoWriter(os.path.join(
                        image_directory, 'gen.avi'), codec, 25, (128*2, 128))
                    for tit, (test_images_b, test_labels_b, com_b, trans_b, cube_b, fn) in enumerate(test_loader_real):
                        test_images_b = Variable(test_images_b.cuda(opts.gpu))
                        test_labels_b = Variable(test_labels_b.cuda(opts.gpu))
                        if mode_idx == 0:
                            pred_pose, pred_post, _ = trainer.dis.regress_a(
                                test_images_b)
                        else:
                            pred_pose, pred_post, _ = trainer.dis.regress_b(
                                test_images_b)

                        if True:
                            pred_pose = trainer.vae.decode(pred_post)

                        n = test_labels_b.size(0)

                        gt_pose = test_labels_b.data.cpu().numpy().reshape((n, -1, 3))
                        pr_pose = pred_pose.data.cpu().numpy().reshape((n, -1, 3))

                        if tit < 20:
                            for i in range(0, n, 4):
                                real_img = visPair(di_b, test_images_b[i].data.cpu().numpy(), gt_pose[i].reshape((-1)),
                                                   trans_b[i].numpy(), com_b[i].numpy(), cube_b[i].numpy(), 50.0)
                                est_img = visPair(di_b, test_images_b[i].data.cpu().numpy(), pr_pose[i].reshape((-1)),
                                                  trans_b[i].numpy(), com_b[i].numpy(), cube_b[i].numpy(), 50.0)

                                vid.write(
                                    np.hstack((real_img, est_img)).astype('uint8'))

                        both_img = np.vstack((real_img, est_img))

                        if True and tit < 8:
                            if img2sav is None:
                                img2sav = both_img
                            else:
                                img2sav = np.hstack((img2sav, both_img))

                        if 'nyu' in opts.config:
                            restrictedJointsEval = np.array(
                                [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32])
                            gt_pose = gt_pose[:, restrictedJointsEval]
                            pr_pose = pr_pose[:, restrictedJointsEval]

                        for i in range(n):
                            gt3D.append(
                                gt_pose[i]*(cube_b.numpy()[0]/2.) + com_b[i].numpy())
                            joints.append(
                                pr_pose[i]*(cube_b.numpy()[0]/2.) + com_b[i].numpy())
                            joints_imgcord.append(di_b.joints3DToImg(
                                pr_pose[i]*(cube_b.numpy()[0]/2.) + com_b[i].numpy()))

                        score += meanJntError
                        num_samples += test_images_b.size(0)

                    cv2.imwrite(image_directory + '/_test.jpg',
                                img2sav.astype('uint8'))
                    vid.release()

                    hpe = Evaluation(np.array(gt3D), np.array(joints))
                    mean_err = hpe.getMeanError()
                    over_40 = 100. * \
                        hpe.getNumFramesWithinMaxDist(40) / len(gt3D)
                    best_err = np.minimum(best_err, mean_err)
                    best_acc = np.maximum(best_acc, over_40)
                    print("------------ Mean err: {:.4f} ({:.4f}) mm, Max over 40mm: {:.2f} ({:.2f}) %".format(
                        mean_err, best_err, over_40, best_acc))

            # Save network weights
            if (iterations+1) % config.snapshot_save_iterations == 0:
                if opts.mode == 'pretrain':
                    trainer.save(config.snapshot_prefix, iterations)
                elif 'estimate' in opts.mode:
                    trainer.save(config.snapshot_prefix+'_est', iterations)

            iterations += 1
            if iterations >= max_iterations:
                return


if __name__ == '__main__':
    main(sys.argv)
