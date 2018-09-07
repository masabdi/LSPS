"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from .common_net import *
import sklearn.decomposition

class Mapping(nn.Module):
  def __init__(self, params):
    super(Mapping, self).__init__()
    self.input_dim = params['input_dim']
    dim = params['output_dim']
    ch = params['output_ch']
    self.output_dim = (ch, dim, dim)

    model = []
    model += [LeakyReLUConvTranspose2d(self.input_dim, 4 * ch, kernel_size=4, stride=1, padding=0)]
    #model += [INSResBlock(4 * ch, 4 * ch, dropout=0.)]
    model += [LeakyReLUConvTranspose2d(4 * ch, 4 * ch, kernel_size=4, stride=2, padding=1)]
    #model += [INSResBlock(4 * ch, 4 * ch, dropout=0.)]
    model += [LeakyReLUConvTranspose2d(4 * ch, 2 * ch, kernel_size=4, stride=2, padding=1)]
    #model += [INSResBlock(2 * ch, 2 * ch, dropout=0.)]
    model += [nn.ConvTranspose2d(2 * ch, ch, kernel_size=4, stride=2, padding=1)]

    self.model = nn.Sequential(*model)

  def cuda(self,gpu):
    self.model.cuda(gpu)

  def forward(self, x):
    return self.model(x.unsqueeze(2).unsqueeze(3))


class poseVAE(nn.Module):
  def __init__(self, params):
    super(poseVAE, self).__init__()
    self.input_dim = params['input_dim']
    self.z_dim = params['z_dim']
    self.h_dim = params['h_dim']

    #encoder
    self.en_fc1 = nn.Linear(self.input_dim, self.h_dim)
    self.en_mu = nn.Linear(self.h_dim, self.z_dim)
    self.en_sigma = nn.Linear(self.h_dim, self.z_dim)

    #decoder
    self.de_fc1 = LeakyReLULinear(self.z_dim, self.h_dim)
    self.de_fc2 = nn.Linear(self.h_dim, self.input_dim)

    self.lrelu = nn.LeakyReLU(inplace=True)
    self.softplus = nn.Softplus()

    self.preset_parameters()

  def preset_parameters(self):
    self.en_mu.weight.data.normal_(0, 0.002)
    self.en_mu.bias.data.normal_(0, 0.002)
    self.en_sigma.weight.data.normal_(0, 0.002)
    self.en_sigma.bias.data.normal_(0, 0.002)

  def cuda(self,gpu):
    self.en_fc1.cuda(gpu)
    self.en_mu.cuda(gpu)
    self.en_sigma.cuda(gpu)
    self.de_fc1.cuda(gpu)
    self.de_fc2.cuda(gpu)

  def forward(self, y):
    z, mu, sd = self.encode(y)
    recons = self.decode(z)
    return recons, z, mu, sd

  def encode(self, y):
    en_h0 = self.lrelu(self.en_fc1(y))
    mu = self.en_mu(en_h0)
    sd = self.softplus(self.en_sigma(en_h0))
    noise = Variable(torch.normal(torch.zeros(mu.size()), std=0.05)).cuda(y.data.get_device())
    return mu + sd.mul(noise), mu, sd

  def decode(self, z):
    de_h0 = self.de_fc1(z)
    de_h1 = self.de_fc2(de_h0)
    return de_h1


class SharedDis(nn.Module):
  def __init__(self, params):
    super(SharedDis, self).__init__()
    ch = params['ch']
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    n_front_layer = params['n_front_layer']
    n_expand_layer = params['n_expand_layer'] if 'n_expand_layer' in params.keys() else 0
    n_shared_layer = params['n_shared_layer']
    self.post_dim = params['post_dim'] 
    self.reg_dim = params['reg_dim']
    self.model_A, tch = self._make_front_net(ch, input_dim_a, n_front_layer)
    self.model_B, tch = self._make_front_net(ch, input_dim_b, n_front_layer)
    self.model_S, self.D, self.Post = self._make_shared_net(tch, n_shared_layer, n_expand_layer)
    self.dropout = None#nn.Dropout(0.4)

  def _make_front_net(self, ch, input_dim, n_layer):
    model = []
    model += [LeakyReLUConv2d(input_dim, ch, kernel_size=7, stride=2, padding=3)] #16
    tch = ch
    for i in range(1, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2
    return nn.Sequential(*model), tch

  def _make_shared_net(self, ch, n_layer, n_expand_layer):
    model = []
    if n_layer == 0:
      return nn.Sequential(*model)
    tch = ch
    for i in range(0, n_expand_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=1, padding=1)] # 8
      tch *= 2
    for i in range(0, n_layer):
      model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)] # 8
      tch *= 2

    post = nn.Conv2d(tch, self.post_dim, kernel_size=2, stride=1, padding=0)  # 1
    discrim = nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)  # 1

    return nn.Sequential(*model), discrim, post

  def cuda(self,gpu):
    self.model_A.cuda(gpu)
    self.model_B.cuda(gpu)
    self.model_S.cuda(gpu)
    self.D.cuda(gpu)
    self.Post.cuda(gpu)

  def regress_a(self, x_A):
    feats_A = self.model_S(self.model_A(x_A))
    feats_A = feats_A if self.dropout is None else self.dropout(feats_A)
    post = self.Post(feats_A)
    return post.squeeze(), post.squeeze(), post.squeeze()

  def regress_b(self, x_B):
    feats_B = self.model_S(self.model_B(x_B))
    feats_B = feats_B if self.dropout is None else self.dropout(feats_B)
    post = self.Post(feats_B)
    return post.squeeze(), post.squeeze(), post.squeeze()

  def feats(self, x_aa, x_ba, x_ab, x_bb):
    x_A = torch.cat((x_aa, x_ba), 0)
    x_B = torch.cat((x_ab, x_bb), 0)
    feats = torch.cat((self.model_A(x_A), self.model_B(x_B)), 0)
    feats = self.model_S(feats)
    return torch.split(feats, feats.size(0)//4, dim=0)

  def forward(self, x_A, x_B, second_feats=False):
    feats = torch.cat((self.model_A(x_A), self.model_B(x_B)), 0)
    feats = self.model_S(feats)
    out_D = self.D(feats)
    feats_A, feats_B = torch.split(feats, feats.size(0)//2, dim=0)
    out_D_A, out_D_B = torch.split(out_D, out_D.size(0)//2, dim=0)
    return out_D_A.view(-1), out_D_B.view(-1), feats_A, feats_B



class SharedResGen(nn.Module):
  def __init__(self, params):
    super(SharedResGen, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2 
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [LeakyINSResBlock(tch, tch, dropout=res_dropout_ratio)]
      encB += [LeakyINSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [LeakyINSResBlock(tch, tch, dropout=res_dropout_ratio)]
    enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [LeakyINSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    decA = []
    decB = []
    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      decA += [LeakyINSResBlock(tch, tch, dropout=res_dropout_ratio)]
      decB += [LeakyINSResBlock(tch, tch, dropout=res_dropout_ratio)]
    # Convolutional back-end
    for i in range(1, n_gen_front_blk):
      decA += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      decB += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      tch = tch//2
    decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def decode(self, z):
    out = self.dec_shared(z)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    return out_A, out_B

  def encode(self, x_A, x_B):
    out_A = self.enc_shared(self.encode_A(x_A))
    out_B = self.enc_shared(self.encode_B(x_B))
    return out_A, out_B

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared



# In COCOResGen4, 
class SharedResXGen(nn.Module):
  def __init__(self, params):
    super(SharedResXGen, self).__init__()
    input_dim_a = params['input_dim_a']
    input_dim_b = params['input_dim_b']
    ch = params['ch']
    n_enc_front_blk  = params['n_enc_front_blk']
    n_enc_res_blk    = params['n_enc_res_blk']
    n_enc_shared_blk = params['n_enc_shared_blk']
    n_gen_shared_blk = params['n_gen_shared_blk']
    n_gen_res_blk    = params['n_gen_res_blk']
    n_gen_front_blk  = params['n_gen_front_blk']
    n_resnext_k  = params['n_resnext_k'] if 'n_resnext_k' in params.keys() else 1
    n_resnext_c  = params['n_resnext_c'] if 'n_resnext_c' in params.keys() else 4
    if 'res_dropout_ratio' in params.keys():
      res_dropout_ratio = params['res_dropout_ratio']
    else:
      res_dropout_ratio = 0

    ##############################################################################
    # BEGIN of ENCODERS
    # Convolutional front-end
    encA = []
    encB = []
    encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=1, padding=3)]
    encB += [LeakyReLUConv2d(input_dim_b, ch, kernel_size=7, stride=1, padding=3)]
    tch = ch
    for i in range(1,n_enc_front_blk):
      encA += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      encB += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
      tch *= 2 
    # Residual-block back-end
    for i in range(0, n_enc_res_blk):
      encA += [LeakyINSResNeXtBlock(tch, tch, k=n_resnext_k, cardinality=n_resnext_c, dropout=res_dropout_ratio)]
      encB += [LeakyINSResNeXtBlock(tch, tch, k=n_resnext_k, cardinality=n_resnext_c, dropout=res_dropout_ratio)]
    # END of ENCODERS
    ##############################################################################

    ##############################################################################
    # BEGIN of SHARED LAYERS
    # Shared residual-blocks
    enc_shared = []
    for i in range(0, n_enc_shared_blk):
      enc_shared += [LeakyINSResNeXtBlock(tch, tch, k=n_resnext_k, cardinality=n_resnext_c, dropout=res_dropout_ratio)]
    enc_shared += [GaussianNoiseLayer()]
    dec_shared = []
    for i in range(0, n_gen_shared_blk):
      dec_shared += [LeakyINSResNeXtBlock(tch, tch, k=n_resnext_k, cardinality=n_resnext_c, dropout=res_dropout_ratio)]
    # END of SHARED LAYERS
    ##############################################################################

    ##############################################################################
    # BEGIN of DECODERS
    decA = []
    decB = []
    # Residual-block front-end
    for i in range(0, n_gen_res_blk):
      decA += [LeakyINSResNeXtBlock(tch, tch, k=n_resnext_k, cardinality=n_resnext_c, dropout=res_dropout_ratio)]
      decB += [LeakyINSResNeXtBlock(tch, tch, k=n_resnext_k, cardinality=n_resnext_c, dropout=res_dropout_ratio)]
    # Convolutional back-end
    for i in range(1, n_gen_front_blk):
      decA += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      decB += [LeakyReLUConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
      tch = tch//2
    decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
    decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
    decA += [nn.Tanh()]
    decB += [nn.Tanh()]
    # END of DECODERS
    ##############################################################################
    self.encode_A = nn.Sequential(*encA)
    self.encode_B = nn.Sequential(*encB)
    self.enc_shared = nn.Sequential(*enc_shared)
    self.dec_shared = nn.Sequential(*dec_shared)
    self.decode_A = nn.Sequential(*decA)
    self.decode_B = nn.Sequential(*decB)

  def decode(self, z):
    out = self.dec_shared(z)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    return out_A, out_B

  def encode(self, x_A, x_B):
    out_A = self.enc_shared(self.encode_A(x_A))
    out_B = self.enc_shared(self.encode_B(x_B))
    return out_A, out_B

  def forward(self, x_A, x_B):
    out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out_A = self.decode_A(out)
    out_B = self.decode_B(out)
    x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
    x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
    return x_Aa, x_Ba, x_Ab, x_Bb, shared

  def forward_a2b(self, x_A):
    out = self.encode_A(x_A)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_B(out)
    return out, shared

  def forward_b2a(self, x_B):
    out = self.encode_B(x_B)
    shared = self.enc_shared(out)
    out = self.dec_shared(shared)
    out = self.decode_A(out)
    return out, shared



