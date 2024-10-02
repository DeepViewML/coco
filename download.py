import os
import json
import requests
import cv2
import numpy as np
from tqdm import tqdm
import lmdb
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def download_coco(dataset_type):
    url = f"http://images.cocodataset.org/zips/{dataset_type}2017.zip"
    annotation_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Download and extract images
    os.system(f"wget {url} -O {dataset_type}2017.zip")
    os.system(f"unzip {dataset_type}2017.zip")
    
    # Download and extract annotations
    os.system(f"wget {annotation_url} -O annotations.zip")
    os.system("unzip annotations.zip")

def convert_bbox_to_darknet(bbox, img_width, img_height):
    x, y, w, h = bbox
    return [
        0,
        (x + w/2) / img_width,
        (y + h/2) / img_height,
        w / img_width,
        h / img_height
    ]

def create_lmdb(dataset_type):
    coco = COCO(f'annotations/instances_{dataset_type}2017.json')
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    env = lmdb.open(
        f'dataset/{dataset_type}2017.lmdb', 
        map_size=(32 * 1024 * 1024 * 1024), 
        max_dbs=10,
        subdir=False,
        lock=False
    )
    
    images_db = env.open_db(b'image')
    masks_db = env.open_db(b'mask')
    bboxes_db = env.open_db(b'boxes')  

    with env.begin(write=True) as txn:
        for img_id in tqdm(img_ids):
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            img_width, img_height = img_info['width'], img_info['height']

            # store boxes in bboxes_db
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            bboxes = [convert_bbox_to_darknet(ann['bbox'], img_width, img_height) for ann in anns]
            bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 5)
            txn.put(file_name.encode(), bboxes.tobytes(), db=bboxes_db)

            # store image in images_db
            with open(os.path.join(f'{dataset_type}2017', file_name), 'rb') as f:       
                txn.put(file_name.encode(), f.read(), db=images_db)

            # Store mask in masks_db with the same key as the image and boxes, also mask has to be 1 where people class is present  
            
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            for ann in anns:
                mask = np.maximum(mask, coco.annToMask(ann))

            txn.put(file_name.encode(), mask.tobytes(), db=masks_db)



def main():
    os.makedirs('dataset', exist_ok=True)
    for dataset_type in ['train', 'val']:
        print(f"Processing {dataset_type} dataset...")
        
        # Check if files already exist
        annotation_file = f'annotations/instances_{dataset_type}2017.json'
        image_dir = f'{dataset_type}2017'
        
        if not os.path.exists(annotation_file) or not os.path.exists(image_dir):
            download_coco(dataset_type)
        else:
            print(f"Files for {dataset_type} dataset already exist. Skipping download.")
        
        create_lmdb(dataset_type)
        print(f"Finished processing {dataset_type} dataset.")

if __name__ == "__main__":
    main()
