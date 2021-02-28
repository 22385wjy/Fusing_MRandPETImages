import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
from pytorch_msssim import ssim
import os
import os.path
import multiprocessing
import scipy.io as scio
from PIL import Image
import cv2

def load_mri_data():
    # load the train mri data
    dataset = os.path.join(os.getcwd(), './data_add_t2pet/MR-T2')
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    #print(len(data))
    train_mri = np.zeros((len(data), image_width, image_length))
    for i in range(len(data)):
        train_mri[i, :, :] = (imageio.imread(data[i]))
        train_mri[i, :, :] = (train_mri[i, :, :] - np.min(train_mri[i, :, :])) / (
                    np.max(train_mri[i, :, :]) - np.min(train_mri[i, :, :]))
        train_mri[i, :, :] = np.float32(train_mri[i, :, :])

    # expand dimension to add the channel
    train_mri = np.expand_dims(train_mri, axis=1)

    # verify the shape matches the pytorch standard
    #print(train_mri.shape)
    # convert the MRI training data to pytorch tensor
    train_mri_tensor = torch.from_numpy(train_mri).float()
    #print(train_mri_tensor.shape)

    return train_mri_tensor

def load_pet_data():
    # load the train pet data
    dataset = os.path.join(os.getcwd(), './data_add_t2pet/FDG')
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    train_other = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    train_pet = np.zeros((len(data), image_width, image_length), dtype=float)
    for i in range(len(data)):
        train_other[i, :, :, :] = (imageio.imread(data[i]))
        train_pet[i, :, :] = 0.2989 * train_other[i, :, :, 0] + 0.5870 * train_other[i, :, :, 1] + 0.1140 * train_other[
                                                                                                            i, :, :, 2]
        train_pet[i, :, :] = (train_pet[i, :, :] - np.min(train_pet[i, :, :])) / (
                    np.max(train_pet[i, :, :]) - np.min(train_pet[i, :, :]))

    # expand the dimension to add the channel
    train_pet = np.expand_dims(train_pet, axis=1)

    # verify the shape matches the pytorch standard
    #print(train_pet.shape)

    # convert the PET training data to pytorch tensor
    train_pet_tensor = torch.from_numpy(train_pet).float()
    #print(train_pet_tensor.shape)

    return train_pet_tensor

class _CFConv1(nn.Module):
    def __init__(self):
        super(_CFConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _CFConv2(nn.Module):
    def __init__(self):
        super(_CFConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 4, 7, 1, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 7, 1, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 7, 1, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _FFConv1(nn.Module):
    def __init__(self):
        super(_FFConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _FFConv2(nn.Module):
    def __init__(self):
        super(_FFConv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        #self.up = nn.Upsample(scale_factor=2)

    def upsample(self, x, size):
        return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1=self.conv1(x)
        #print(x1.shape)
        x2=self.pool(x1)
        #print(x2.shape)
        x3=self.conv2(x2)
        #print(x3.shape)
        x4=self.conv3(x3)
        #print(x4.shape)
        x5=self.upsample(x4,(256,256))
        #print(x5.shape)
        x1_2=self.conv2(x1)
        #print(x1_2.shape)
        x5_2=self.conv6(x5)
        #print(x5_2.shape)
        cat=torch.cat((x1_2,x5_2),1)
        #print(cat.shape)
        x6=self.conv4(cat)
        #print(x6.shape)

        return x6

def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

class forOriginal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(forOriginal, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class DsConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DsConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class downsample(nn.Module):
    def __init__(self, dw_channels1=16, dw_channels2=32, out_channels=64, **kwargs):
        super(downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dw_channels1, 3, 2),
            nn.BatchNorm2d(dw_channels1),
            nn.ReLU(True)
        )
        self.dsconv1 = DsConv(dw_channels1, dw_channels2, 2)
        #self.dsconv2 = DsConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        #print(x.shape)
        x1 = self.conv(x)
        #print(x1.shape)
        x2 = self.dsconv1(x1)
        #print(x2.shape)
        # x2 = self.dsconv2(x1)
        # print(x2.shape)
        return x1,x2

class SegConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(SegConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class SegConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, **kwargs):
        super(SegConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class PoolandUp(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PoolandUp, self).__init__()
        inter_channels = int(in_channels / 2)
        self.conv1 = SegConv(in_channels, inter_channels, 1, **kwargs)
        #self.out = SegConv(in_channels * 2, out_channels, 1)
        self.out = SegConv(in_channels, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        #print(size)
        p1 = self.pool(x, int(size[0]/2))
        #print(p1.shape)
        cp1 = self.conv1(p1)
        #print(cp1.shape)
        ucp1 = self.upsample(cp1, size)
        #print(ucp1.shape)

        x=self.upsample(self.conv1(x), size)

        x = torch.cat([x, ucp1], dim=1)
        #print(x.shape)
        x = self.out(x)
        #print('PoolandUp ',x.shape)
        return x

class SeFeature(nn.Module):
    def __init__(self, in_channels, out_channels, t=2, stride=2, **kwargs):
        super(SeFeature, self).__init__()
        self.block = nn.Sequential(
            SegConv(in_channels, in_channels * t, 1),
            SegConv2(in_channels * t, in_channels * t ),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.pool_up=PoolandUp(32,32)

    def forward(self, x):
        out1 = self.block(x)
        #print(out1.shape)
        out=self.pool_up(out1)
        #print('SeFeature ',out.shape)
        return out

class segment(nn.Module):
    def __init__(self, in_channels, num_classes, stride=1, **kwargs):
        super(segment, self).__init__()
        self.conv1 = DsConv(in_channels, in_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, x):
        size = x.size()[2:]
        #print('size= ',size)
        x1 = self.conv1(x)
        #print(x1.shape)
        x2 = self.conv(x1)
        #print(x2.shape)

        return x2

def net():
    # define the network
    class DeepTreeFuse(nn.Module):
        def __init__(self,num_classes=3):
            super(DeepTreeFuse, self).__init__()

            #####one cf layer 1#####
            self.one_cf1 =_CFConv1()
            self.one_cf2 = _CFConv2()
            #####one ff layers#####
            self.one_ff1 =_FFConv1()
            self.one_ff2 = _FFConv2()

            #####two cf layer 1#####
            self.two_cf1 = _CFConv1()
            self.two_cf2 = _CFConv2()
            #####two ff layers#####
            self.two_ff1 = _FFConv1()
            self.two_ff2 = _FFConv2()

            self.c_o=forOriginal(1,16,7,1,3)
            self.f_o = forOriginal(1, 16, 3, 1, 3)

            # cf reconstruction
            self.cf_recons = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )  # output shape (,16,256,256)

            # ff reconstruction
            self.ff_recons = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
            )  # output shape (,16,256,256)

            # final reconstruction
            self.recon = nn.Sequential(  # input shape (,16, 256, 256)
                nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=1,
                          padding=2))  # output shape (,3,256,256)
            self.ff2con = nn.Sequential(  # input shape (,16, 256, 256)
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1,
                          padding=2))
            self.reconse = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1,
                          padding=2))
            self.re = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1,
                          padding=1))

            # mulscale feature extract
            self.downsample = downsample()
            self.globalfeature = SeFeature(32, 32)
            self.otherconv = SegConv2(32, 32)
            self.merge = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                          padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False)
            )

            self.segclass = segment(32, num_classes=3)
            self.dropandplt = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(16, num_classes, 1)
            )
            self.x0 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
            self.x1 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
            self.x2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
            self.xxx = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True)
                                     )

        def upsample(self, x, size):
            return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

        def tensor_max(self, tensors):
            max_tensor = None
            for i, tensor in enumerate(tensors):
                if i == 0:
                    max_tensor = tensor
                else:
                    max_tensor = torch.max(max_tensor, tensor)
            return max_tensor

        def obtain_to_CFF(self, tensor):
            # cf
            x=tensor
            cf1 = self.one_cf1(x)
            cf2 = self.one_cf2(x)
            x1 = torch.cat((cf1, cf2), 1)
            cf_x = self.cf_recons(x1)

            # ff
            ff1 = self.one_ff1(x)
            ff2 = self.one_ff2(x)
            x2 = torch.cat((ff1, ff2), 1)
            ff_x = self.ff_recons(x2)

            x_change_c_oc = self.c_o(x)
            x_change_c_of = self.f_o(x)
            #outx =(lf_x+x_change_c_ol)/2+(hf_x+x_change_c_oh)/2
            #outx = self.tensor_max([lf_x, hf_x])
            outx =cf_x+ff_x
            #print(outx.shape)

            return cf_x,ff_x,outx,x_change_c_oc,x_change_c_of

        def mulscale_one(self,tensor):
            x=tensor
            sx, se = self.downsample(x)
            s1 = self.globalfeature(se)
            s2 = self.otherconv(se)
            s = s1 + s2
            out_s = self.merge(s)
            out_up = self.upsample(out_s, (128, 128))
            sclass = self.segclass(out_up)
            size = x.size()[2:]
            sclass = self.upsample(sclass, size)
            result_S = sclass
            x0 = self.x0(x)
            sx = self.x1(sx)
            se2 = self.x2(se)
            sx = self.upsample(sx, size)
            se2 = self.upsample(se2, size)
            x0 = self.dropandplt(x0)  # self.segclass(x0)
            sx = self.segclass(sx)
            se2 = self.segclass(se2)
            x3_out = x0 + sx + se2
            #print('x3_out ', x3_out.shape)

            total_seg1 = result_S + x3_out
            #print(total_seg1.shape)
            return total_seg1

        def doFuse(self,t1,t2,t3,t4,xl,xh,yl,yh,cx_l,cx_h,cy_l,cy_h):
            lhf_x=t1
            lhf_y=t2
            se_x=t3
            se_y=t4
            HF = xh + yh
            fu_lf_hf = (HF + xl + yl) / 3
            lf_hf = self.recon(fu_lf_hf)
            out_se = (se_x + se_y) / 2
            fuseout = (lf_hf + out_se) / 2
            fuseout = torch.tanh(fuseout)

            return fuseout

        def forward(self, x, y):
            # __________________ one type __________________
            cf_x,ff_x,cff_x,cx_c,cx_f=self.obtain_to_CFF(x)
            se_x=self.mulscale_one(x)  # mulscale

            # __________________ two type __________________
            cf_y, ff_y, cff_y,cy_c,cy_f = self.obtain_to_CFF(y)
            se_y = self.mulscale_one(y)

            ## to do fusing
            fuseout=self.doFuse(cff_x,cff_y,se_x,se_y,cf_x,ff_x,cf_y, ff_y,cx_c,cx_f,cy_c,cy_f)
            fuseout=self.re(fuseout)
            #print(fuseout.shape)

            return fuseout

    dtn = DeepTreeFuse().to(device)
    dtn = dtn.float()
    #print(dtn)

    return dtn

def train_for_model(dtn,train_mri_tensor,train_pet_tensor):
    # define the optimizers and loss functions
    optimizer = torch.optim.Adam(dtn.parameters(), lr=learning_rate)  # optimize all dtn parameters
    l2_loss = nn.MSELoss()  # MSEloss

    # perform the training
    counter = 0
    lamda = 0.7
    loss_history=[]
    output=torch.empty(2,1,256,256)
    nn.init.constant_(output, 0.3)
    for epoch in range(EPOCH):
        batch_idxs = 555 // batch_size
        for idx in range(0, batch_idxs):
            b_x = train_mri_tensor[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)
            b_y = train_pet_tensor[idx * batch_size: (idx + 1) * batch_size, :, :, :].to(device)
            counter += 1
            output = dtn(b_x, b_y)  # dtn output
            #print('output.shape ',output.shape)

            ssim_loss_mri = 1 - ssim(output, b_x, data_range=1)
            ssim_loss_pet = 1 - ssim(output, b_y, data_range=1)
            ssim_total = ssim_loss_mri + ssim_loss_pet
            l2_total = l2_loss(output, b_x) + l2_loss(output, b_y)
            loss_total = (1 - lamda) * ssim_total + lamda * l2_total
            #print('loss_total: ', loss_total)

            optimizer.zero_grad()
            #loss_total.backward(retain_graph=True)
            loss_total.backward()
            optimizer.step()
            loss_history.append(loss_total.item())

            if counter % 25 == 0:
                print(
                    "Epoch: [%2d],step: [%2d], mri_ssim: [%.8f], pet_ssim: [%.8f],  total_ssim: [%.8f], total_l2: [%.8f], total_loss: [%.8f]"
                    % (epoch, counter, ssim_loss_mri, ssim_loss_pet, ssim_total, l2_total, loss_total))

            if (epoch == EPOCH - 1):
                # Save a checkpoint
                torch.save(dtn.state_dict(), './fusionDFP.pth', _use_new_zipfile_serialization=False)

                return loss_history


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_length = 256
    image_width = 256
    mr_channels = 1
    gray_channels = 1
    pet_channels = 4
    rgb_channels = 2
    batch_size = 3
    EPOCH = 61
    learning_rate = 0.0001  

    train_mri_tensor = load_mri_data()
    train_pet_tensor = load_pet_data()
    new_cnn = net()
    for key in new_cnn.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                nn.init.constant_(new_cnn.state_dict()[key], 0.3)
            if 'bn' in key:
                nn.init.constant_(new_cnn.state_dict()[key][...], 1)
        elif key.split('.')[-1] == 'bias':
            nn.init.constant_(new_cnn.state_dict()[key][...],0)

    loss_history=train_for_model(new_cnn, train_mri_tensor, train_pet_tensor)

    plt.plot(loss_history,label='loss for every epoch')
    fig=plt.gcf()
    #fig.savefig('./loss.png')
    plt.show()
    print('finish fusing')