"""
Preprocess dataset.
Images within one folder is an image, the image file named `"image.png"`, the mask file named `"mask.png"`.
"""
import os
import argparse
import cv2
import numpy as np
# from model_zoo.official.cv.unet.src.config import cfg_unet
from src.config import cfg_unet

def annToMask(ann, height, width):
    """Convert annotation to RLE and then to binary mask."""
    from pycocotools import mask as maskHelper
    segm = ann['segmentation']  # 前景边界点，对应coco rle格式
    if isinstance(segm, list):
        rles = maskHelper.frPyObjects(segm, height, width)
        rle = maskHelper.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskHelper.frPyObjects(segm, height, width)
    else:
        rle = ann['segmentation']
    m = maskHelper.decode(rle)
    return m

def preprocess_cell_nuclei_dataset(param_dict):
    """
    Preprocess for Cell Nuclei dataset.
    merge all instances to a mask, and save the mask at data_dir/img_id/mask.png.
    """
    print("========== start preprocess Cell Nuclei dataset ==========")
    data_dir = param_dict["data_dir"]
    img_ids = sorted(next(os.walk(data_dir))[1])
    for img_id in img_ids:
        path = os.path.join(data_dir, img_id)
        if (not os.path.exists(os.path.join(path, "image.png"))) or \
                (not os.path.exists(os.path.join(path, "mask.png"))):
            img = cv2.imread(os.path.join(path, "images", img_id + ".png"))
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.concatenate([img, img, img], axis=-1)
            mask = []
            for mask_file in next(os.walk(os.path.join(path, "masks")))[2]:
                mask_ = cv2.imread(os.path.join(path, "masks", mask_file), cv2.IMREAD_GRAYSCALE)
                mask.append(mask_)
            mask = np.max(mask, axis=0)
            cv2.imwrite(os.path.join(path, "image.png"), img)
            cv2.imwrite(os.path.join(path, "mask.png"), mask)

def preprocess_coco_dataset(param_dict):
    """
    Preprocess for coco dataset.
    Save image and mask at save_dir/img_name/image.png save_dir/img_name/mask.png
    """
    print("========== start preprocess coco dataset ==========")
    from pycocotools.coco import COCO
    
    
    #1、填写参数标签
    #------------------**************
    anno_json =   # annotaion json文件路径
    coco_cls =   # 数据类别80+1类的名字
    coco_dir =   # 数据集路径
    save_dir =   # 最终结果保存路径
    
    
    #--------------------***********************

    coco_cls_dict = {}  # key为类名，value为索引值
    for i, cls in enumerate(coco_cls):
        
         /2、补全该处代码
        #------------------**************
        
        coco_cls_dict[xxx] =   # eg:{'backgroud':0, 'person':1',...}
        #------------------**************
        
    coco = COCO(anno_json)
    classs_dict = {}  # key为idx，value为类名，这里不报错background
    cat_ids = coco.loadCats(coco.getCatIds())  # 长度80，元素形如{'supercategory':'vihicle','id':2, "name": "bick"}
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]  # {1:"collid"}
    image_ids = coco.getImgIds()  # [1,...,300]
    images_num = len(image_ids)  # 300
    for ind, img_id in enumerate(image_ids):
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]  # xxx.jpg
        img_name, _ = os.path.splitext(file_name)
        image_path = os.path.join(coco_dir, file_name)
        if not os.path.isfile(image_path):
            print("{}/{}: {} is in annotations but not exist".format(ind + 1, images_num, image_path))
            continue
        if not os.path.exists(os.path.join(save_dir, img_name)):
            os.makedirs(os.path.join(save_dir, img_name))  # 保存在 /xxx/图片名/
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)  # 每个元素为一个前景物体，{'segmentation':[[...]], 'area':x ,}
        h = coco.imgs[img_id]["height"]
        w = coco.imgs[img_id]["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        
        #3、补全该处代码
        #------------------**************
        for instance in anno:
            m = annToMask(   )  # h*w的array
            c = coco_cls_dict[ ]  # 最里层为此分割物体属于哪类。class_dict把idx转成类名，cls_dict把类名转回idx
            if len(m.shape) < 3:
                mask[:, :] += () * ( * )  # 当前属于背景的mask=0，与得到的物体的矩阵m，对应位置上标记为类别c
            else:
                mask[:, :] += () * (((  * c).astype(np.uint8)  # 将3d转成2d,做上面类似的操作
        #------------------**************
        
        img = cv2.imread(image_path)
        cv2.imwrite(os.path.join(save_dir, img_name, "image.png"), img)
        cv2.imwrite(os.path.join(save_dir, img_name, "mask.png"), mask)

def preprocess_dataset(cfg, data_dir):
    """Select preprocess function."""
    if cfg['dataset'].lower() == "cell_nuclei":
        preprocess_cell_nuclei_dataset({"data_dir": data_dir})
    elif cfg['dataset'].lower() == "coco":
        if 'split' in cfg and cfg['split'] == 1.0:
            train_data_path = os.path.join(data_dir, "train")
            val_data_path = os.path.join(data_dir, "val")
            train_param_dict = {"anno_json": cfg["anno_json"], "coco_classes": cfg["coco_classes"],
                                "coco_dir": cfg["coco_dir"], "save_dir": train_data_path}
            preprocess_coco_dataset(train_param_dict)
            val_param_dict = {"anno_json": cfg["val_anno_json"], "coco_classes": cfg["coco_classes"],
                              "coco_dir": cfg["val_coco_dir"], "save_dir": val_data_path}
            preprocess_coco_dataset(val_param_dict)
        else:
            param_dict = {"anno_json": cfg["anno_json"], "coco_classes": cfg["coco_classes"],
                          "coco_dir": cfg["coco_dir"], "save_dir": data_dir}
            preprocess_coco_dataset(param_dict)
    else:
        raise ValueError("Not support dataset mode {}".format(cfg['dataset']))
    print("========== end preprocess dataset ==========")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_url', dest='data_url', type=str, default='data/',
                        help='save data directory')
    args = parser.parse_args()
    preprocess_dataset(cfg_unet, args.data_url)
