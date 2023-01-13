#!/usr/bin/python3

# Copyright (C) 2019-2021 Xilinx, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may
# not use this file except in compliance with the License. A copy of the
# License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import ctypes
import os
import sys
import uuid
import re
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import torch
# Following found in PYTHONPATH setup by XRT
import pyxrt
import cv2

from utils_binding import *

mnist_test = datasets.MNIST('./data',
                   train=False,
                   download=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

test_data = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)



faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

def runKernel(opt):
    n = 0
    count = 0
    while True:

        data, target = next(iter(test_data))  # 迭代器
        new_data = data[0][0].clone().numpy()  # 拷贝数据
        # plt.imsave('MNIST_images/'+str(i)+str(target.numpy())+'.png', new_data)
        img_flatten = ((new_data.flatten() * 255).astype(int))
        #print(img_flatten)
        #print(target.numpy())

        d = pyxrt.device(opt.index)
        xbin = pyxrt.xclbin(opt.bitstreamFile)
        uuid = d.load_xclbin(xbin)

        kernellist = xbin.get_kernels()

        rule = re.compile("lenet*")
        kernel = list(filter(lambda val: rule.match(val.get_name()), kernellist))[0]
        kHandle= pyxrt.kernel(d, uuid, kernel.get_name(), pyxrt.kernel.shared)

        zeros = bytearray(opt.DATA_SIZE)
        #print("Allocate and initialize buffers")
        boHandle1 = pyxrt.bo(d, opt.DATA_SIZE, pyxrt.bo.normal, kHandle.group_id(0))
        boHandle1.write(zeros, 0)
        bo1 = boHandle1.map()

        boHandle2 = pyxrt.bo(d, opt.DATA_SIZE, pyxrt.bo.normal, kHandle.group_id(1))
        boHandle2.write(zeros, 0)
        bo2 = boHandle2.map()

        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray,(28,28))
        ret,gray = cv2.threshold (gray,90,255,cv2.THRESH_BINARY)
        gray_flatten = gray.flatten()

        for i in range(opt.DATA_SIZE):
            # bo1[i] = img_flatten[i]         ##  MNIST test
            bo1[i] = (255 - gray_flatten[i])  ##  cv test


        boHandle1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, opt.DATA_SIZE, 0)

        #print("Start the kernel")
        run = kHandle(boHandle1, boHandle2, opt.DATA_SIZE)
        #print("Now wait for the kernel to finish")
        state = run.wait()

        #print("Get the output data from the device and validate it")
        boHandle2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, opt.DATA_SIZE, 0)

        ##  MNIST test
        #########################################
        # print(bo2[0] == target.numpy())
        # if (bo2[0] == target.numpy()):
        #     count = count + 1
        # print(count)
        #########################################

        ##  cv test
        #########################################
        cv2.imshow('video', gray)
        print("The number is " + str(bo2[0]), end='\r')
        #########################################
        n = n + 1

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

def main(args):
    opt = Options()
    Options.getOptions(opt, args)

    try:
        runKernel(opt)
        print('')
        print("TEST PASSED")
        return 0

    except OSError as o:
        print(o)
        print("TEST FAILED")
        return -o.errno

    except AssertionError as a:
        print(a)
        print("TEST FAILED")
        return -1
    except Exception as e:
        print(e)
        print("TEST FAILED")
        return -1

if __name__ == "__main__":
    os.environ["Runtime.xrt_bo"] = "false"
    result = main(sys.argv)
    sys.exit(result)
