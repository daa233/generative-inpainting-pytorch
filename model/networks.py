import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils

from utils.tools import extract_image_patches, flow_to_image, pt_flow_to_image, highlight_flow,\
    reduce_mean, reduce_sum, default_loader, same_padding


class Generator(nn.Module):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, 3, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # 5 x 256 x 256
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = torch.clamp(x, -1., 1.)

        return x_stage1


class FineGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, 3, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2, offset_flow


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=True, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate, self.rate])      # b*hw*c*k*k
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1 / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1 / self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension

        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride])   # b*hw*c*k*k
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4.*self.rate), mode='nearest')
        m_groups = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                strides=[self.stride, self.stride])     # b*hw*c*k*k

        # m = m[0]  # hw*c*k*k
        # m = reduce_mean(m, axis=[1, 2, 3])  # hw*1*1*1
        # m = m.permute(1, 0, 2, 3).contiguous()  # 1*hw*1*1
        # mm = (m==0).to(torch.float32)   # 1*hw*1*1

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale * 255    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, m_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # hw*c*k*k
            wi_normed = wi / torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2), axis=[1, 2, 3])), escape_NaN)
            xi_normed = same_padding(xi, [self.ksize, self.ksize], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi_normed, wi_normed, stride=1)   # 1*hw*H*W

            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)

                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, int_bs[2]* int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)

            # mi: hw*c*k*k
            mi = reduce_mean(mi, axis=[1, 2, 3])  # hw*1*1*1
            mi = mi.permute(1, 0, 2, 3).contiguous()  # 1*hw*1*1
            mm = (mi == 0).to(torch.float32)  # 1*hw*1*1

            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # 1*hw*H*W

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W
            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)   # b*2*H*W
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest')

        return y, flow


def test_contextual_attention(args):
    import cv2
    import os
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    def float_to_uint8(img):
        img = img * 255
        return img.astype('uint8')

    rate = 2
    stride = 1
    grid = rate*stride

    b = default_loader(args.imageA)
    w, h = b.size
    b = b.resize((w//grid*grid//2, h//grid*grid//2), Image.ANTIALIAS)
    # b = b.resize((w//grid*grid, h//grid*grid), Image.ANTIALIAS)
    print('Size of imageA: {}'.format(b.size))

    f = default_loader(args.imageB)
    w, h = f.size
    f = f.resize((w//grid*grid, h//grid*grid), Image.ANTIALIAS)
    print('Size of imageB: {}'.format(f.size))

    f, b = transforms.ToTensor()(f), transforms.ToTensor()(b)
    f, b = f.unsqueeze(0), b.unsqueeze(0)
    if torch.cuda.is_available():
        f, b = f.cuda(), b.cuda()

    contextual_attention = ContextualAttention(ksize=3, stride=stride, rate=rate, fuse=True)

    if torch.cuda.is_available():
        contextual_attention = contextual_attention.cuda()

    yt, flow_t = contextual_attention(f, b)
    vutils.save_image(yt, 'vutils' + args.imageOut, normalize=True)
    vutils.save_image(flow_t, 'flow' + args.imageOut, normalize=True)
    # y = tensor_img_to_npimg(yt.cpu()[0])
    # flow = tensor_img_to_npimg(flow_t.cpu()[0])
    # cv2.imwrite('flow' + args.imageOut, flow_t)


class LocalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class GlobalDis(nn.Module):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, weight_norm='none', norm='in', activation='relu',
                 pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, weight_norm=weight_norm,
                              norm=norm, activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, weight_norm=weight_norm,
                              norm=norm, activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out



# Patch discriminator
class PatchD(nn.Module):
    def __init__(self, config):
        super(PatchD, self).__init__()

        self.input_dim = config['input_nc']
        self.dim = config['ndf']
        self.weight_norm = config['weight_norm']
        self.norm = config['norm']
        self.gan_type = config['gan_type']
        self.activ = config['activ']
        self.pad_type = config['pad_type']

        # 256 x 256
        self.layer1 = Conv2dBlock(self.input_dim, self.dim, 4, 2, 1, weight_norm=self.weight_norm,
                                  norm='none', activation=self.activ, pad_type=self.pad_type)
        # 128 x 128
        self.layer2 = Conv2dBlock(self.dim, self.dim*2, 4, 2, 1, weight_norm=self.weight_norm,
                                  norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        # 64 x 64
        self.layer3 = Conv2dBlock(self.dim*2, self.dim*4, 4, 2, 1, weight_norm=self.weight_norm,
                                  norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        # 32 x 32
        self.layer4 = Conv2dBlock(self.dim*4, self.dim*8, 4, 2, 1, weight_norm=self.weight_norm,
                                  norm=self.norm, activation=self.activ, pad_type=self.pad_type)
        # 31 x 31
        self.layer5 = nn.Conv2d(self.dim*8, 1, kernel_size=4, stride=1, padding=1)
        # 30 x 30

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, config):
        super(MsImageDis, self).__init__()
        self.input_dim = config['input_nc']
        self.dim = config['ndf']
        self.weight_norm = config['weight_norm']
        self.norm = config['norm']
        self.gan_type = config['gan_type']
        self.activ = config['activ']
        self.pad_type = config['pad_type']
        self.n_layer = 4
        self.num_scales = 3     # number of scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, weight_norm=self.weight_norm,
                              norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim*2, 4, 2, 1, weight_norm=self.weight_norm,
                                  norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.flayer = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1, stride=1)
        self.glayer = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1, stride=1)
        self.hlayer = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1)
        self.softmax = nn.Softmax()
        self.lambd = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        batchsize, C, width, height = input.size()
        # B * CX(N)
        f = self.flayer(input).view(batchsize, -1, width * height).permute(0, 2, 1)
        g = self.glayer(input).view(batchsize, -1, width * height)
        # s = f.transpose * g
        s = torch.bmm(f, g)
        beta = self.softmax(s)
        o = torch.bmm(self.hlayer(input).view(batchsize, -1, width * height), beta.permute(0, 2, 1))
        out = self.lambd * o.view(batchsize, C, width, height) + input

        return out


####################################################################
#--------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u
    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))
    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn

def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


##################################################################################
# VGG network definition
##################################################################################
class VggFeatureExtractor(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    # VGG16 Features
    # Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))             # conv1_1
    #     (1): ReLU(inplace)                                                                # relu1_1
    #     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))            # conv1_2
    #     (3): ReLU(inplace)                                                                # relu1_2
    #     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   # pool1
    #     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))           # conv2_1
    #     (6): ReLU(inplace)                                                                # relu2_1
    #     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))          # conv2_2
    #     (8): ReLU(inplace)                                                                # relu2_2
    #     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   # pool2
    #     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_1
    #     (11): ReLU(inplace)                                                               # relu3_1
    #     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_2
    #     (13): ReLU(inplace)                                                               # relu3_2
    #     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_3
    #     (15): ReLU(inplace)                                                               # relu3_3
    #     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # pool3
    #     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_1
    #     (18): ReLU(inplace)                                                               # relu4_1
    #     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_2
    #     (20): ReLU(inplace)                                                               # relu4_2
    #     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_3
    #     (22): ReLU(inplace)                                                               # relu4_3
    #     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # pool4
    #     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_1
    #     (25): ReLU(inplace)                                                               # relu5_1
    #     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_2
    #     (27): ReLU(inplace)                                                               # relu5_2
    #     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_3
    #     (29): ReLU(inplace)                                                               # relu5_3
    #     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # pool5
    # )

    # VGG19 Features
    # Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))             # conv1_1
    #     (1): ReLU(inplace)                                                                # relu1_1
    #     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))            # conv1_2
    #     (3): ReLU(inplace)                                                                # relu1_2
    #     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   # pool1
    #     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))           # conv2_1
    #     (6): ReLU(inplace)                                                                # relu2_1
    #     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))          # conv2_2
    #     (8): ReLU(inplace)                                                                # relu2_2
    #     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   # pool2
    #     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_1
    #     (11): ReLU(inplace)                                                               # relu3_1
    #     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_2
    #     (13): ReLU(inplace)                                                               # relu3_2
    #     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_3
    #     (15): ReLU(inplace)                                                               # relu3_3
    #     (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv3_4
    #     (17): ReLU(inplace)                                                               # relu3_4
    #     (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # pool3
    #     (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_1
    #     (20): ReLU(inplace)                                                               # relu4_1
    #     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_2
    #     (22): ReLU(inplace)                                                               # relu4_2
    #     (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_3
    #     (24): ReLU(inplace)                                                               # relu4_3
    #     (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv4_4
    #     (26): ReLU(inplace)                                                               # relu4_4
    #     (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # pool4
    #     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_1
    #     (29): ReLU(inplace)                                                               # relu5_1
    #     (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_2
    #     (31): ReLU(inplace)                                                               # relu5_2
    #     (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_3
    #     (33): ReLU(inplace)                                                               # relu5_3
    #     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))         # conv5_4
    #     (35): ReLU(inplace)                                                               # relu5_4
    #     (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  # pool5
    # )

    def __init__(self, model='vgg16', use_cuda=True, device_ids=None):
        super(VggFeatureExtractor, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        if model == 'vgg16':
            self.vgg_layers = vgg.vgg16(pretrained=True).features
            self.layer_name_uni_mapping = {
                '0': 'conv1_1',
                '1': 'relu1_1',
                '2': 'conv1_2',
                '3': 'conv1_2',
                '4': 'pool1',
                '5': 'conv2_1',
                '6': 'relu2_1',
                '7': 'conv2_2',
                '8': 'relu2_2',
                '9': 'pool2',
                '10': 'conv3_1',
                '11': 'relu3_1',
                '12': 'conv3_2',
                '13': 'relu3_2',
                '14': 'conv3_3',
                '15': 'relu3_3',
                '16': 'pool3',
                '17': 'conv4_1',
                '18': 'relu4_1',
                '19': 'conv4_2',
                '20': 'relu4_2',
                '21': 'conv4_3',
                '22': 'relu4_3',
                '23': 'pool4',
                '24': 'conv4_4',
                '25': 'relu4_4',
                '26': 'conv5_1',
                '27': 'pool4',
                '28': 'conv5_1',
                '29': 'relu5_1',
                '30': 'conv5_2',
                '31': 'relu5_2',
                '32': 'pool5'
            }

        elif model == 'vgg19':
            self.vgg_layers = vgg.vgg19(pretrained=True).features
            self.layer_name_uni_mapping = {
                '0': 'conv1_1',
                '1': 'relu1_1',
                '2': 'conv1_2',
                '3': 'conv1_2',
                '4': 'pool1',
                '5': 'conv2_1',
                '6': 'relu2_1',
                '7': 'conv2_2',
                '8': 'relu2_2',
                '9': 'pool2',
                '10': 'conv3_1',
                '11': 'relu3_1',
                '12': 'conv3_2',
                '13': 'relu3_2',
                '14': 'conv3_3',
                '15': 'relu3_3',
                '16': 'pool3',
                '17': 'conv4_1',
                '18': 'relu4_1',
                '19': 'conv4_2',
                '20': 'relu4_2',
                '21': 'conv4_3',
                '22': 'relu4_3',
                '23': 'pool4',
                '24': 'conv4_4',
                '25': 'relu4_4',
                '26': 'conv5_1',
                '27': 'pool4',
                '28': 'conv5_1',
                '29': 'relu5_1',
                '30': 'conv5_2',
                '31': 'relu5_2',
                '32': 'conv5_3',
                '33': 'relu5_3',
                '34': 'conv5_4',
                '35': 'relu5_4',
                '36': 'pool5'
            }
        else:
            raise NotImplementedError('Not support model: {}. Choices: ["vgg16", "vgg19"]'.format(model))

        # Construct `layer_name_uni_mapping` a to bidirectional mapping
        self.layer_name_mapping = {}
        for key, value in self.layer_name_uni_mapping.items():
            self.layer_name_mapping[key] = value
            self.layer_name_mapping[value] = key

    def preprocess(self, x):
        tensor_type = type(x.data)
        mean = tensor_type(x.data.size())
        std = tensor_type(x.data.size())
        if self.use_cuda:
            mean = mean.cuda()
            std = std.cuda()
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        x = x.sub(mean).div(std)  # subtract mean and divided by std
        return x

    def forward(self, x, feat_layer_items):
        """
        Extract VGG features
        :param x: input image
        :param feat_layer_items: the activation layer to be extracted and its
                                weight, e.g. {'conv3_1': 1.0, 'conv4_2': 1.0}
        :return: the extracted activation list
        """
        x = self.preprocess(x)
        if self.use_cuda:
            self.vgg_layers.cuda()
        outputs = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            layer_name = self.layer_name_mapping[name]
            if layer_name in feat_layer_items:
                outputs[layer_name] = x * feat_layer_items[layer_name]
        return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
