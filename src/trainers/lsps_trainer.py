"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from .lsps_nets import *
from .helpers import get_model_list, _compute_fake_acc, _compute_true_acc
from .init import *
import torch
import torch.nn as nn
import os
import itertools
from utils.evaluation import Evaluation



class LSPSTrainer(nn.Module):
  def __init__(self, hyperparameters):
    super(LSPSTrainer, self).__init__()
    lr = hyperparameters['lr']
    # Initiate the networks
    exec( 'self.dis = %s(hyperparameters[\'dis\'])' % hyperparameters['dis']['name'])
    exec( 'self.gen = %s(hyperparameters[\'gen\'])' % hyperparameters['gen']['name'])
    exec( 'self.vae = %s(hyperparameters[\'vae\'])' % hyperparameters['vae']['name'])
    exec( 'self.map = %s(hyperparameters[\'map\'])' % hyperparameters['map']['name'])
    # Setup the optimizers
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(list(self.gen.parameters()) + list(self.map.parameters()), \
				 				lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=lr*10., betas=(0.5, 0.999), weight_decay=0.001)

    # lr Scheduler 
    self.dis_sch = torch.optim.lr_scheduler.MultiStepLR(self.dis_opt, milestones=[200,300,400,450], gamma=0.5)
    self.gen_sch = torch.optim.lr_scheduler.MultiStepLR(self.gen_opt, milestones=[200,300,400,450], gamma=0.5)
    self.vae_sch = torch.optim.lr_scheduler.MultiStepLR(self.vae_opt, milestones=[125,175], gamma=0.1)

    # Network weight initialization
    self.dis.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    self.vae.apply(gaussian_weights_init)
    self.map.apply(gaussian_weights_init)
    # Setup the loss function for training
    self.ll_loss_criterion_a = torch.nn.L1Loss()
    self.ll_loss_criterion_b = torch.nn.L1Loss()
    self.ll_loss_criterion = torch.nn.L1Loss() # We use MSELoss here



  def _compute_ll_loss(self,a,b):
    return self.ll_loss_criterion(a, b) 

  def _compute_l2_loss(self,a,b):
    return torch.pow(a-b, 2).mean()


  def _compute_kl(self, mu, sd=None):
    mu_2 = torch.pow(mu, 2)
    if sd is None:
	return torch.mean(mu_2)
    sd_2 = torch.pow(sd, 2)
    return (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)

  def vae_update(self, y, hyperparameters):
    self.vae.zero_grad()
    dec, z, mu, sd = self.vae(y)

    enc_loss = self._compute_kl(mu, sd)
    ll_loss = self._compute_ll_loss(dec, y)
    total_loss =  hyperparameters['kl_loss_vae'] * enc_loss + \
		  hyperparameters['ll_loss_vae'] * ll_loss

    total_loss.backward()
    self.vae_opt.step()
    self.vae_total_loss = total_loss.data.cpu().numpy()
    return dec

  def gen_update(self, images_a, labels_a, images_b, labels_b, hyperparameters):
    self.gen.zero_grad()

    x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a, images_b)
    x_bab, shared_bab = self.gen.forward_a2b(x_ba)
    x_aba, shared_aba = self.gen.forward_b2a(x_ab)

    matching_loss_z, matching_loss_a, matching_loss_b = 0., 0., 0.
    if hyperparameters['train_map']:
      self.map.zero_grad()

      labels = torch.cat((labels_a, labels_b), 0)
      enc_pose, _, _ = self.vae.encode(labels)
      z_pose2depth = self.map(enc_pose)

      decode_A, decode_B = self.gen.decode(z_pose2depth)
      decode_A, _ = torch.split(decode_A, decode_A.size(0) // 2, 0)
      _, decode_B = torch.split(decode_B, decode_B.size(0) // 2, 0)

      data_a = torch.cat((x_ba, decode_A), 0)
      data_b = torch.cat((x_ab, decode_B), 0)

      matching_loss_z = self._compute_l2_loss(shared, z_pose2depth)
      matching_loss_a = self.ll_loss_criterion_a(decode_A, images_a)
      matching_loss_b = self.ll_loss_criterion_b(decode_B, images_b)
    else:
      data_a, decode_A = x_ba, x_ba
      data_b, decode_B = x_ab, x_ab

    outs_a, outs_b, _, _ = self.dis(data_a,data_b)

    outputs_a = nn.functional.sigmoid(outs_a)
    outputs_b = nn.functional.sigmoid(outs_b)
    all_ones = Variable(torch.ones((outputs_a.size(0))).cuda(self.gpu))

    ad_loss_a = nn.functional.binary_cross_entropy(outputs_a, all_ones)
    ad_loss_b = nn.functional.binary_cross_entropy(outputs_b, all_ones)

    enc_loss  = self._compute_kl(shared)
    enc_bab_loss = self._compute_kl(shared_bab)
    enc_aba_loss = self._compute_kl(shared_aba)
    ll_loss_a = self.ll_loss_criterion_a(x_aa, images_a)
    ll_loss_b = self.ll_loss_criterion_b(x_bb, images_b)
    ll_loss_aba = self.ll_loss_criterion_a(x_aba, images_a)
    ll_loss_bab = self.ll_loss_criterion_b(x_bab, images_b)
    total_loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss) + \
                 hyperparameters['ll_map_z_w'] * (matching_loss_z) + \
		 hyperparameters['ll_map_w'] * (matching_loss_a + matching_loss_b)


    total_loss.backward()
    self.gen_opt.step()
    self.gen_enc_loss = enc_loss.data.cpu().numpy()
    self.gen_enc_loss2 = (enc_aba_loss+enc_bab_loss).data.cpu().numpy()
    self.gen_ad_loss = (ad_loss_a+ad_loss_b).data.cpu().numpy()
    self.gen_ll_loss = (ll_loss_a+ll_loss_b).data.cpu().numpy()
    self.gen_ll_loss2 = (ll_loss_bab+ll_loss_aba).data.cpu().numpy()
    if hyperparameters['train_map']: 
      self.gen_map_loss = matching_loss_z.data.cpu().numpy()
      self.gen_map_loss2 = (matching_loss_a + matching_loss_b).data.cpu().numpy()
    self.gen_total_loss = total_loss.data.cpu().numpy()
    return (x_aa, x_ba, x_ab, x_bb, x_aba, x_bab, decode_A, decode_B)

  def dis_update(self, images_a, labels_a, images_b, labels_b, com_a, com_b, hyperparameters, feat_mat=True):
    self.dis.zero_grad()
    x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a, images_b)

    if hyperparameters['train_map']:
      labels = torch.cat((labels_a, labels_b), 0)
      enc_pose, _, _ = self.vae.encode(labels)
      z_pose2depth = self.map(enc_pose)

      decode_A, decode_B = self.gen.decode(z_pose2depth)
      decode_A, _ = torch.split(decode_A, decode_A.size(0) // 2, 0)
      _, decode_B = torch.split(decode_B, decode_B.size(0) // 2, 0)

      data_a = torch.cat((images_a, x_ba, x_aa, decode_A), 0)
      data_b = torch.cat((images_b, x_ab, x_bb, decode_B), 0)
      ndiv = 4
    elif feat_mat:
      data_a = torch.cat((images_a, x_ba, x_aa), 0)
      data_b = torch.cat((images_b, x_ab, x_bb), 0)
      ndiv = 3
    else:
      data_a = torch.cat((images_a, x_ba), 0)
      data_b = torch.cat((images_b, x_ab), 0)
      ndiv = 2

    res_a, res_b, feats_a, feats_b = self.dis(data_a,data_b)


    feature_loss_a, feature_loss_b = 0., 0.
    if feat_mat: # feature matching 
      feat_as = torch.split(feats_a, feats_a.size(0) // ndiv, 0)
      feat_bs = torch.split(feats_b, feats_a.size(0) // ndiv, 0)
      dummy_variable = Variable(torch.zeros(feat_as[2].size()).cuda(self.gpu))
      feature_loss_a = self._compute_ll_loss(feat_bs[1] - feat_as[2], dummy_variable)
      feature_loss_b = self._compute_ll_loss(feat_as[1] - feat_bs[2], dummy_variable)

    out_a = nn.functional.sigmoid(res_a)
    out_b = nn.functional.sigmoid(res_b)

    outs_a = torch.split(out_a, out_a.size(0) // ndiv, 0)
    outs_b = torch.split(out_b, out_b.size(0) // ndiv, 0)


    all1 = Variable(torch.ones((outs_a[0].size(0))).cuda(self.gpu))
    all0 = Variable(torch.zeros((outs_a[1].size(0))).cuda(self.gpu))

    ad_true_loss_a = nn.functional.binary_cross_entropy(outs_a[0], all1)
    ad_true_loss_b = nn.functional.binary_cross_entropy(outs_b[0], all1)
    ad_fake_loss_a = nn.functional.binary_cross_entropy(outs_a[1], all0)
    ad_fake_loss_b = nn.functional.binary_cross_entropy(outs_b[1], all0)

    true_a_acc = _compute_true_acc(outs_a[0])
    true_b_acc = _compute_true_acc(outs_b[0])
    fake_a_acc = _compute_fake_acc(outs_a[1])
    fake_b_acc = _compute_fake_acc(outs_b[1])
    exec( 'self.dis_true_acc = 0.5 * (true_a_acc + true_b_acc)')
    exec( 'self.dis_fake_acc = 0.5 * (fake_a_acc + fake_b_acc)')

    ad_fake_dec_a , ad_fake_dec_b = 0., 0.
    if hyperparameters['train_map']:
      ad_fake_dec_a = nn.functional.binary_cross_entropy(outs_a[3], all0)
      ad_fake_dec_b = nn.functional.binary_cross_entropy(outs_b[3], all0)

    ad_loss_a = ad_true_loss_a + ad_fake_loss_a + ad_fake_dec_a
    ad_loss_b = ad_true_loss_b + ad_fake_loss_b + ad_fake_dec_b

    loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
           hyperparameters['feature_w'] * (feature_loss_a + feature_loss_b)

    loss.backward()
    self.dis_opt.step()
    self.dis_ad_loss = (ad_loss_a + ad_loss_b).data.cpu().numpy()
    if feat_mat:
      self.dis_feat_loss = (feature_loss_a + feature_loss_b).data.cpu().numpy()
    self.dis_loss = loss.data.cpu().numpy()
    return 

  def post_update(self, images_a, labels_a, images_b, labels_b, com_a, com_b, mode, hyperparameters):
    self.dis.zero_grad()

    x_aa, x_ba, x_ab, x_bb = images_a, images_a, images_b, images_b
    feature_loss_a, feature_loss_b = 0., 0.
    reg_loss_a, reg_loss_b = 0., 0.

    if mode == 0: # Synthetic-Only
        _, pred_post_a, _ = self.dis.regress_a(images_a)
	enc_pose_a,_,_ = self.vae.encode(labels_a)
	reg_loss_a = self._compute_l2_loss(pred_post_a, enc_pose_a) 

    elif mode == 1: # Real-Only
        _, pred_post_b, _ = self.dis.regress_b(images_b)
	enc_pose_b,_,_ = self.vae.encode(labels_b)
	reg_loss_b = self._compute_l2_loss(pred_post_b, enc_pose_b) 

    else: # Synthetic + Unlabeled
        x_aa, x_ba, x_ab, x_bb, lt_codes = self.gen(images_a[0:4], images_b[0:4])
        f_x_aa, f_x_ba, f_x_ab, f_x_bb = self.dis.feats(x_aa, x_ba, x_ab, x_bb)

        dummy_variable = Variable(torch.zeros(f_x_aa.size()).cuda(self.gpu))
        feature_loss_a = self._compute_ll_loss(f_x_ab - f_x_aa, dummy_variable)
        feature_loss_b = self._compute_ll_loss(f_x_ba - f_x_bb, dummy_variable)

	_, pred_post_a, _ = self.dis.regress_a(images_a)
	enc_pose_a,_,_ = self.vae.encode(labels_a)
	reg_loss_a = self._compute_l2_loss(pred_post_a, enc_pose_a) 

	if mode == 4: # Synthetic + semi-supervised
	   _, pred_post_b, _ = self.dis.regress_b(images_b)
	   enc_pose_b,_,_ = self.vae.encode(labels_b)
	   reg_loss_b = self._compute_l2_loss(pred_post_b, enc_pose_b)  

    total_loss = hyperparameters['reg_w'] * (reg_loss_a + reg_loss_b) + \
                 hyperparameters['feature_w_reg'] * (feature_loss_a + feature_loss_b)

    total_loss.backward()
    self.dis_opt.step()

    self.dis_reg_loss = (reg_loss_a + reg_loss_b).data.cpu().numpy()
    self.dis_total_loss = total_loss.data.cpu().numpy()
    return (x_aa, x_ba, x_ab, x_bb, x_aa, x_bb, x_aa, x_bb) #decode_A, decode_B)

  def assemble_outputs(self, images_a, images_b, network_outputs):
    images_a = self.normalize_image(images_a)
    images_b = self.normalize_image(images_b)
    x_aa = self.normalize_image(network_outputs[0])
    x_ba = self.normalize_image(network_outputs[1])
    x_ab = self.normalize_image(network_outputs[2])
    x_bb = self.normalize_image(network_outputs[3])
    x_aba = self.normalize_image(network_outputs[4])
    x_bab = self.normalize_image(network_outputs[5])
    dec_a = self.normalize_image(network_outputs[6])
    dec_b = self.normalize_image(network_outputs[7])
    return torch.cat((images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::], dec_a[0:1, ::], dec_b[0:1, ::],
                      images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::]), 3)

  def resume(self, snapshot_prefix, idx=-1, load_opt=False, est=False):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,"est_gen" if est else "gen", idx)
    if last_model_name is None:
      return 0
    self.gen.load_state_dict(torch.load(last_model_name), strict=False)
    iterations = int(last_model_name[-12:-4])
    last_model_name = get_model_list(dirname, "est_dis" if est else "dis", idx)
    self.dis.load_state_dict(torch.load(last_model_name), strict=False)

    if load_opt:
      try:
        last_model_name = get_model_list(dirname, "optg", idx)
        self.gen_opt.load_state_dict(torch.load(last_model_name))
        last_model_name = get_model_list(dirname, "optd", idx)
        self.dis_opt.load_state_dict(torch.load(last_model_name))
        print('-----optimizer parameters loaded!')
      except:
        print('-----Failed to load optimizer parameters!')

    try:
      last_model_name = get_model_list(dirname, "map", idx)
      self.map.load_state_dict(torch.load(last_model_name), strict=False)
    except:
      print('-----Failed to load map parameters!')

    print('Resume from iteration %d' % iterations)
    return iterations

  def save(self, snapshot_prefix, iterations):
    gen_filename = '%s_gen_%08d.pkl' % (snapshot_prefix, iterations + 1)
    dis_filename = '%s_dis_%08d.pkl' % (snapshot_prefix, iterations + 1)
    map_filename = '%s_map_%08d.pkl' % (snapshot_prefix, iterations + 1)

    gen_opt_filename = '%s_optg_%08d.pkl' % (snapshot_prefix, iterations + 1)
    dis_opt_filename = '%s_optd_%08d.pkl' % (snapshot_prefix, iterations + 1)
    #torch.save(self.gen_opt.state_dict(), gen_opt_filename)
    #torch.save(self.dis_opt.state_dict(), dis_opt_filename)

    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.dis.state_dict(), dis_filename)
    #torch.save(self.map.state_dict(), map_filename)

  def save_vae(self, snapshot_prefix, iterations, frac):
    vae_filename = '%s_vae_%.2f_%08d.pkl' % (snapshot_prefix, frac, iterations + 1)
    torch.save(self.vae.state_dict(), vae_filename)

  def load_vae(self, snapshot_prefix, frac):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,'vae_%.2f' % frac)
    if last_model_name is None:
      return 0
    self.vae.load_state_dict(torch.load(last_model_name))
    print('Loading pretrained VAE parameters from %s' % last_model_name)
    return 0

  def cuda(self, gpu):
    self.gpu = gpu
    self.dis.cuda(gpu)
    self.gen.cuda(gpu)
    self.vae.cuda(gpu)
    self.map.cuda(gpu)
    for state in self.gen_opt.state.values():
      for k, v in state.items():
        if torch.is_tensor(v):
          state[k] = v.cuda(gpu)
    for state in self.dis_opt.state.values():
      for k, v in state.items():
        if torch.is_tensor(v):
          state[k] = v.cuda(gpu)

  def normalize_image(self, x):
    return x[:,0:3,:,:]






