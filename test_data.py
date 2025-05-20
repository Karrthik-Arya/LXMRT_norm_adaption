import sys
import csv
import base64
import time
import json
import pandas as pd
import numpy as np

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

if __name__ == '__main__':
    q_path = 'data/test/train_questions.csv'
    df_questions = pd.read_csv(q_path)
    df_questions = (
        df_questions
        .assign(
            prefix=lambda x: x['image'].str.split('_').str[0],
            numeric_id=lambda x: x['image'].str.split('_').str[1].str.split('.').str[0]
        )
        .query("prefix in ['GQA', 'VG']")
        .dropna(subset=['numeric_id'])
    )
    question_img_ids = set(df_questions['numeric_id'].unique())
    print(len(question_img_ids))
    feature_data1 = load_obj_tsv('data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv')
    feature_img_ids1 = {item['img_id'] for item in feature_data1}
    print(feature_data1[0]['features'].shape)
    feature_data2 = load_obj_tsv('data/vg_gqa_imgfeat/vg_gqa_obj36.tsv')
    feature_img_ids2 = {item['img_id'] for item in feature_data2}
    feature_img_ids = feature_img_ids1.union(feature_img_ids2)
    
    # Check for missing images
    missing_images = question_img_ids - feature_img_ids2

    # with open('feature_ids.txt', 'w') as f:
    #     for img_id in feature_img_ids:
    #         f.write(f"{img_id}\n")

    # print("Saved", len(feature_img_ids), "feature IDs to feature_ids.txt")
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images are missing from the features file.")
        print("First 10 missing images:", list(missing_images)[:10])
    else:
        print("All images referenced in the questions have corresponding features!")
