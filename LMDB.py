import lmdb
import numpy as np
import cv2
import os
from turbojpeg import TurboJPEG, TJPF_RGB
from PIL import Image
from io import BytesIO

jpeg = TurboJPEG()

def read_lmdb(lmdb_path, num_samples=5):
    env = lmdb.open(
        lmdb_path, 
        readonly=True, 
        max_dbs=10,
        lock=False,
        subdir=False
    )
    

    images_db = env.open_db(b'image')
    bboxes_db = env.open_db(b'boxes')
    masks_db = env.open_db(b'mask')
        
    with env.begin(buffers=True) as txn:
        image_cursor = txn.cursor(db=images_db)
        image = image_cursor.get(b'000000337561.jpg')
        image_stream = BytesIO(image)
        image = Image.open(image_stream)
        image = np.asarray(image)
        image = image.copy()
        h, w, _ = image.shape

    with env.begin(buffers=True) as txn:
        boxes_cursor = txn.cursor(db=bboxes_db)
        boxes = boxes_cursor.get(b'000000337561.jpg')
        boxes = np.frombuffer(boxes, dtype=np.float32).reshape(-1, 5)

    with env.begin(buffers=True) as txn:
        masks_cursor = txn.cursor(db=masks_db)
        masks = masks_cursor.get(b'000000337561.jpg')
        masks = np.frombuffer(masks, dtype=np.uint8).reshape(h, w)
    
    print(boxes.shape, masks.shape, image.shape)
    plot_image_with_annotations(image, boxes, masks, "output.jpg")
        
def plot_image_with_annotations(img, bboxes, mask, filename):
    h, w = img.shape[:2]
    
    # Draw bounding boxes
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Apply mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img_with_mask = cv2.addWeighted(img, 0.7, mask_rgb * 255, 0.3, 0)
    
    # Display image
    cv2.imwrite(filename, img_with_mask)
    

if __name__ == "__main__":
    lmdb_path = "dataset/train2017.lmdb"  # Update this path if necessary
    read_lmdb(lmdb_path)
