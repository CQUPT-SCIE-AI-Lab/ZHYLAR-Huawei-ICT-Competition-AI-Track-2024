import os
import argparse
import logging
import numpy as np
import cv2
import mindspore
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from src.unet_nested import UNet
from src.config import cfg_unet as cfg
import matplotlib.pyplot as plt

transpose_fn = P.Transpose()
reshape_fn = P.Reshape()
argmax_fn = P.Argmax()
softmax_fn = P.Softmax()

resize_w, resize_h = tuple(cfg['img_size'])


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='data directory')
    parser.add_argument('-s', '--save_url', dest='save_url', type=str, default='pred_visualization/',
                        help='data directory')
    parser.add_argument('-p', '--ckpt_path', dest='ckpt_path', type=str, default='best.ckpt',
                        help='checkpoint path')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    ckpt_path = args.ckpt_path  # "pred_visualization/best-eval-model.ckpt"
    image = args.data_url  # "pred_visualization/image.png"  # (3648, 5472, 3)
    image_path = os.path.join(image, "image.png")
    mask_path = os.path.join(image, "mask.png")  # "pred_visualization/mask.png"  # (3648, 5472)
    save_path = args.save_url  # "pred_visualization/crop_plot.png"

    net = UNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])  # 3,2
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    resize_w, resize_h = tuple(cfg['img_size'])
    img = cv2.imread(image_path)  # np
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 画图，原图，裁剪的
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_crop = img.copy()[1000:2800, 2200:4000, :]
    mask_crop = mask.copy()[1000:2800, 2200:4000]
    img_crop_plot = img_crop.copy()
    plt.figure()
    plt.subplot(1, 2, 1)
    img_crop_plot[np.where(mask_crop == 1)] = [127, 255, 0]
    plt.imshow(cv2.cvtColor(img_crop_plot, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(mask_crop)
    plt.savefig(os.path.join(save_path, "crop_plot.png"))
    print("num of mask crop=1 pixel: ", np.all(img_crop_plot==[127, 255, 0], axis=2).sum())

    ori_h, ori_w, ori_c = img_crop.shape  # hwc = 3k*5k*3
    img_crop = (img_crop.astype(np.float32) - 127.5) / 127.5
    img_crop = cv2.resize(img_crop, (resize_w, resize_h))  # 3k*5k*3 -> 512*512*c
    img_crop = img_crop.transpose(2, 0, 1)  # hwc->chw, 3,512,512
    img_crop = img_crop[np.newaxis, ...]  # chw->1chw, 1,3,512,512
    img_crop = mindspore.Tensor(img_crop, dtype=mindspore.float32)  # ms
    # 送入模型
    prediction = net(img_crop)  # ms,1,2,512,512
    prediction = transpose_fn(prediction[0], (1, 2, 0))  # CHW->HWC, 1,2,512,512->512,512,2
    # argmax得到答案
    pred_h, pred_w, pred_c = prediction.shape  # HWC, 512*512*2
    prediction = reshape_fn(prediction, (-1, pred_c))  # HWC->HW*C //512*512,2
    softmax = softmax_fn(prediction)
    argmax = argmax_fn(softmax).asnumpy()  # H*W,C -> H*W
    print("(resized)num of pred=1 pixel: ", argmax.sum())

    # predict画到原图上，略
    argmax = argmax.reshape(pred_w, pred_h).astype(np.uint8)  # resizeh, resizew
    argmax = cv2.resize(argmax, (ori_w, ori_h))  # croph, cropw
    img_crop_plot[argmax == 1] = np.array([0, 0, 255], dtype=np.float32)  # bgr, 表现为红色
    print("num of mask pred=1 pixel: ", np.all(img_crop_plot==[0, 0, 255], axis=2).sum())

    img[1000:2800, 2200:4000, :] = img_crop_plot
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig(os.path.join(save_path, "predict_plot.png"))

    print("finished")
