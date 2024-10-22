import os
import pathlib
import argparse
import logging
import numpy as np
import cv2
import mindspore
from mindspore import context, Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from src.unet_nested import UNet
from src.config import cfg_unet

# device_id = int(os.getenv('DEVICE_ID'))
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=device_id)
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def draw_result(data_dir, ckpt_path, save_path, cfg):
    transpose_fn = P.Transpose()
    reshape_fn = P.Reshape()
    argmax_fn = P.Argmax()

    resize_w, resize_h = tuple(cfg['img_size'])

    # 固定用unet simple， 固定图片channel=3，类别n_class=2。如果后期用到其他模型，就把eval.py L35~43粘过来
    net = UNet(in_channel=cfg['num_channels'], n_class=cfg['num_classes'])
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)

    abs_path = os.path.abspath(data_dir)
    i = 0
    for dir in os.listdir(abs_path):
        path = os.path.join(abs_path, dir)
        for img in os.listdir(path):
            if img == 'image.png':
                pathlib.Path(os.path.join(save_path, dir)).mkdir(parents=True, exist_ok=True)

                img_path = os.path.join(path, img)

                img = cv2.imread(img_path)  # np
                ori_h, ori_w, ori_c = img.shape  # hwc = 3k*5k*3
                img = (img.astype(np.float32) - 127.5) / 127.5
                img = cv2.resize(img, (resize_w, resize_h))  # 3k*5k*3 -> re*512*c
                img = img.transpose(2, 0, 1)  # hwc->chw, 3,512,512
                img = img[np.newaxis, ...]  # chw->1chw, 1,3,512,512
                img = mindspore.Tensor(img, dtype=mindspore.float32)  # ms
                # 送入模型
                prediction = net(img)  # ms,1,2,512,512
                prediction = transpose_fn(prediction[0], (1, 2, 0))  # CHW->HWC, 1,2,512,512->512,512,2
                # argmax得到答案
                pred_h, pred_w, pred_c = prediction.shape  # HWC, 512*512*2
                prediction = reshape_fn(prediction, (-1, pred_c))  # HWC->HW*C //512*512,2
                argmax = argmax_fn(prediction).asnumpy()  # H*W,C -> H*W
                # 值为1的位置转化为其他颜色。 [x,x,x]，然后与原图相加
                img = transpose_fn(img[0], (1, 2, 0))  # 1CHW->HWC,  1,3,512,512->512,512,3
                img = img*127.5 + 127.5
                img = reshape_fn(img, (-1, ori_c)).asnumpy()  # np, HWC -> HW*C // HW,3
                img[argmax == 1] = np.array([127, 255, 0], dtype=np.float32)
                img = img.reshape(resize_h, resize_w, -1)
                img = cv2.resize(img, (ori_w, ori_h))
                # cv imwrite保存图片到 path 中。
                cv2.imwrite(os.path.join(save_path, dir, "predict.png"), img)
                print('save as {}'.format(os.path.join(save_path, "predict"+str(i)+".png")))
                break


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
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    print("Training setting:", args)
    draw_result(data_dir=args.data_url, ckpt_path=args.ckpt_path, save_path=args.save_url, cfg=cfg_unet)