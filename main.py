import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import cv2
import logging
import json
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
import gdown

logging.disable(logging.WARNING)
tf.get_logger().setLevel('ERROR')

def normalize_image(image, offset=(0.485, 0.456, 0.406), scale=(0.229, 0.224, 0.225)):
    with tf.name_scope('normalize_image'):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        offset = tf.constant(offset)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        image -= offset

        scale = tf.constant(scale)
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        image /= scale
        return image


def load_image_into_numpy_array(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))
    (im_width, im_height) = image.size
    image = image.convert('RGB')
    np_array = np.array(image.getdata())
    reshaped = np_array.reshape((1, im_height, im_width, 3))
    return reshaped.astype(np.uint8)


model_handle = r'C:\GreenAI\material_form_model\saved_model\saved_model'
PATH_TO_LABELS = './models/official/projects/waste_identification_ml/pre_processing/config/data/material_form_labels.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print('loading model...')
hub_model = hub.load(model_handle)
print('model loaded!')

while True:
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile('mycreds.txt')
    gauth.Authorize()
    drive = GoogleDrive(gauth)
    fileList = drive.ListFile({'q': "'1guqMw71nQgfMgY5KRD1o7CrlUD-0WGk2' in parents and trashed=false"}).GetList()

    if fileList:
        starttime = time.time()
        url = "https://drive.google.com/drive/folders/1guqMw71nQgfMgY5KRD1o7CrlUD-0WGk2"
        gdown.download_folder(url, use_cookies=False)

        for file in fileList:
            print('Title: %s, ID: %s' % (file['title'], file['id']))
            fileID = file['id']
            file2 = drive.CreateFile({'id': fileID})
            file2.Delete()  # Permanently delete the file.

        files = os.listdir("C:\GreenAI\Images")
        image_path = fr"C:\GreenAI\Images\{files[0]}"
        image_np = load_image_into_numpy_array(image_path)
        print("loaded image")

        hub_model_fn = hub_model.signatures["serving_default"]
        height = hub_model_fn.structured_input_signature[1]['inputs'].shape[1]
        width = hub_model_fn.structured_input_signature[1]['inputs'].shape[2]
        input_size = (height, width)

        image_np_cp = cv2.resize(image_np[0], input_size[::-1], interpolation=cv2.INTER_AREA)
        image_np = normalize_image(image_np_cp)
        image_np = tf.expand_dims(image_np, axis=0)
        image_np.get_shape()

        print("Inference")
        results = hub_model_fn(image_np)

        result = {key: value.numpy() for key, value in results.items()}

        label_id_offset = 0
        min_score_thresh = 0.4

        result['detection_boxes'][0][:, [0, 2]] /= height
        result['detection_boxes'][0][:, [1, 3]] /= width

        if 'detection_masks' in result:
            detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
            detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes,
                image_np.shape[1], image_np.shape[2])

            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               np.uint8)

            result['detection_masks_reframed'] = detection_masks_reframed.numpy()

        image_np_cp,dict = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_cp,
            result['detection_boxes'][0],
            (result['detection_classes'][0] + label_id_offset).astype(int),
            result['detection_scores'][0],
            category_index=category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thresh,
            agnostic_mode=False,
            instance_masks=result.get('detection_masks_reframed', None),
            line_thickness=2)


        cv2.imwrite("output/imageimage.jpg", image_np_cp)
        mask_count = np.sum(result['detection_scores'][0] >= min_score_thresh)

        print('Total number of objects found are:', mask_count)
        print(dict)

        with open('output/data.json', 'w', encoding='utf-8') as f:
            json.dump(dict, f, ensure_ascii=False, indent=4)

        mask = np.zeros_like(detection_masks_reframed[0])
        for i in range(mask_count):
            if result['detection_scores'][0][i] >= min_score_thresh:
                mask += detection_masks_reframed[i]

        mask = tf.clip_by_value(mask, 0, 1)

        path = r"C:\GreenAI\output"
        file_list = drive.ListFile({'q': "'1KF0z2k5zbAeunKMIyshJ5O_7baWychik' in parents and trashed=False"}).GetList()

        for x in os.listdir(path):
            try:
                for file1 in file_list:
                    if file1['title'] == x:
                        file1.Delete()
            except:
                pass

        for x in os.listdir(path):
            file2 = drive.CreateFile({'title': x, 'parents': [{'id': '1KF0z2k5zbAeunKMIyshJ5O_7baWychik'}]})
            file2.SetContentFile(os.path.join(path, x))
            file2.Upload()
            file2 = None

        for file in os.scandir('Images'):
            os.remove(file.path)

        print('Finished')
        endtime = time.time()
        print(endtime - starttime)
