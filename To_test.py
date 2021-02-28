import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
import os
import os.path
import multiprocessing
from Fusing_mrt2pet import net
from PIL import Image
import matplotlib
import scipy.misc


def load_mri_to_do_test():
    # load the test input MRI dataset
    dataset = os.path.join(os.getcwd(), './testImage/MRI/')
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    test_mri = np.zeros((len(data), image_width, image_length))
    for i in range(len(data)):
        test_mri[i, :, :] = (imageio.imread(data[i]))
        test_mri[i, :, :] = (test_mri[i, :, :] - np.min(test_mri[i, :, :])) / (
                    np.max(test_mri[i, :, :]) - np.min(test_mri[i, :, :]))
        test_mri[i, :, :] = np.float32(test_mri[i, :, :])

    # expand dimension to add the channel
    test_mri = np.expand_dims(test_mri, axis=1)
    # verify the shape matches the pytorch standard
    print('test_mri',test_mri.shape)

    #plt.imshow(test_mri[0, 0, :, :], 'gray')
    # plt.savefig('MRI.png', bbox_inches = 'tight',pad_inches = 0,dpi=200)

    # convert the MRI Testing data to pytorch tensor
    test_mri_tensor = torch.from_numpy(test_mri).float()
    test_mri_tensor = test_mri_tensor.to(device)
    print('test_mri_tensor ',test_mri_tensor.shape)
    test_mri_tensor.requires_grad = True

    return test_mri_tensor


def load_pet_to_do_test():
    # load the test input PET dataset
    dataset = os.path.join(os.getcwd(), './testImage/PET/')
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    test_pet = np.zeros((len(data), image_width, image_length))
    for i in range(len(data)):
        test_pet[i, :, :] = (imageio.imread(data[i]))
        test_pet[i, :, :] = (test_pet[i, :, :] - np.min(test_pet[i, :, :])) / (
                    np.max(test_pet[i, :, :]) - np.min(test_pet[i, :, :]))
        test_pet[i, :, :] = np.float32(test_pet[i, :, :])

    # expand dimension to add the channel
    test_pet = np.expand_dims(test_pet, axis=1)
    # verify the shape matches the pytorch standard
    print('test_pet ',test_pet.shape)

    #plt.imshow(test_pet[0, 0, :, :], 'gray')
    # plt.savefig('PET.png', bbox_inches = 'tight',pad_inches = 0,dpi=200)

    # convert the PET Testing data to pytorch tensor
    test_pet_tensor = torch.from_numpy(test_pet).float()
    test_pet_tensor = test_pet_tensor.to(device)
    print('test_pet_tensor ',test_pet_tensor.shape)
    test_pet_tensor.requires_grad = True

    return test_pet_tensor

if __name__=="__main__":
    # device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    image_length = 256
    image_width = 256

    # load the model
    dtn = net()
    dtn.load_state_dict(torch.load('./fusionDFP.pth'))
    dtn.eval()

    # predicted the fused image
    test_mri_tensor = load_mri_to_do_test()
    test_pet_tensor = load_pet_to_do_test()
    fused = dtn(test_mri_tensor.to(device), test_pet_tensor.to(device))
    fused_numpy = fused.data.cpu().numpy()

    one_pic_out=fused_numpy[0, 0, :, :]
    imageio.imsave('./fusedImage/g0.png', one_pic_out)
    matplotlib.image.imsave('./fusedImage/c0.png', one_pic_out)

    print('Congratulations on finishing fusion!')







