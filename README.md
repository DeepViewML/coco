# Object Detection Project

This project implements an object detection model using DVC (Data Version Control) for experiment tracking and pipeline management.

## Setup

1. Clone the repository:
   ```
   git clone git@github.com:DeepViewML/coco.git
   cd coco
   git checkout coco-people
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up the dataset:
   - Place your training and validation datasets in the `dataset` directory.
   - Ensure the dataset is in LMDB format as specified in `dataset.yaml`.

## Usage
5. Download and prepare the dataset:
   ```
   python download.py
   ```
   This script will download the COCO dataset (train and validation sets) and convert it to LMDB format as specified in `dataset.yaml`. It will only download people class images and annotations.

6. (Optional) Verify the LMDB dataset:
   You can use the following Python script to sample and visualize images from the LMDB dataset:

   ```python
   import lmdb
   import cv2
   import numpy as np
   import random

   def sample_lmdb(lmdb_path, num_samples=5):
       env = lmdb.open(lmdb_path, readonly=True, lock=False)
       
       with env.begin() as txn:
           cursor = txn.cursor()
           keys = [key for key, _ in cursor]
           
       for _ in range(num_samples):
           key = random.choice(keys)
           with env.begin() as txn:
               img_bytes = txn.get(key, db=txn.open_db(b'image'))
               mask_bytes = txn.get(key, db=txn.open_db(b'mask'))
               bbox_bytes = txn.get(key, db=txn.open_db(b'boxes'))
           
           img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
           mask = np.frombuffer(mask_bytes, np.uint8).reshape(img.shape[:2])
           bboxes = np.frombuffer(bbox_bytes, np.float32).reshape(-1, 5)
           
           # Visualize
           for bbox in bboxes:
               _, x, y, w, h = bbox
               x1, y1 = int((x - w/2) * img.shape[1]), int((y - h/2) * img.shape[0])
               x2, y2 = int((x + w/2) * img.shape[1]), int((y + h/2) * img.shape[0])
               cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
           
           cv2.imshow('Image', img)
           cv2.imshow('Mask', mask * 255)
           cv2.waitKey(0)
       
       cv2.destroyAllWindows()
       env.close()

   # Usage
   sample_lmdb('dataset/train2017.lmdb')
   ```

   This script will display 5 random images from the training set along with their bounding boxes and masks.

### Training

To train the model:


