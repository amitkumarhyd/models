# USAGE: python inference.py

'''
TODO:
 Find lighter models or other version of DeepLab
 Plot labels legend

REFERENCES: 
https://github.com/tensorflow/models/tree/master/research/deeplab
https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb
https://www.learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation/

'''
import sys

import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

sys.path.append('utils')
import get_dataset_colormap

# Select and download models

_MODEL_URLS = {
    #'xception_coco_voctrainaug': 'http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    #'xception_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    'mobilenetv2_coco_voctrainval': 'http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
}


#_TARBALL_NAME = 'deeplab_model.tar.gz'
_TARBALL_NAME = 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz'


model_dir = 'D:\\open_projects\\person_segmentation\\models\\research\\deeplab\\model'

download_path = os.path.join(model_dir, _TARBALL_NAME)
#print('downloading model to %s, this might take a while...' % download_path)
#urllib.request.urlretrieve('http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz', download_path)
#print('download completed!')

# Load model in TensorFlow

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if _FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()
        
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():      
            tf.import_graph_def(graph_def, name='')
            graph_nodes=[n for n in graph_def.node]

        self.sess = tf.compat.v1.Session(graph=self.graph)
            
    def run(self, image):
        """Runs inference on a single image.
        
        Args:
            image: A PIL.Image object, raw input image.
            
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of 'resized_image'.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


#print(download_path)
model = DeepLabModel(download_path)

cap = cv2.VideoCapture('C:\\Users\\AMIT KUMAR\\Pictures\\Camera Roll\\WIN_20200515_09_22_33_Pro.mp4')

while True:
    ret, frame = cap.read()
    
    # From cv2 to PIL
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    
    # Run model
    resized_im, seg_map = model.run(pil_im)
    #print(seg_map)
    
    # Get color of mask labels
    seg_image = get_dataset_colormap.label_to_color_image(
        seg_map, get_dataset_colormap.get_pascal_name()).astype(np.uint8)
    #print(get_dataset_colormap.get_pascal_name())
    
    # Convert PIL image back to cv2 and resize
    frame = np.array(pil_im)
    r = seg_image.shape[1] / frame.shape[1]
    dim = (int(frame.shape[0] * r), seg_image.shape[1])[::-1]
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite("seg_map.jpg", seg_map)

    # Stack horizontally color frame and mask
    color_and_mask = np.hstack((resized, seg_image))

    cv2.imwrite("mask.jpg", seg_image)

    image_copy = seg_image.copy()

    # Person pixel color for Pascal VOC BGR[192,128,128] light pink
    person_pixels_mask = np.all(seg_image == [192, 128, 128], axis=-1)

    non_person_pixels_mask = ~person_pixels_mask

    # Mask person to white
    image_copy[person_pixels_mask] = [255, 255, 255]
    cv2.imwrite("Mask_person_white.jpg", image_copy)

    # Retain only person
    image_copy[non_person_pixels_mask] = [0, 0, 0]
    cv2.imwrite("Masked_non_person.jpg", image_copy)

    mask_out=cv2.subtract(image_copy,resized)
    cv2.imwrite("mask_out_1.jpg", mask_out)
    mask_out=cv2.subtract(image_copy,mask_out)
    cv2.imwrite("mask_out_2.jpg", mask_out)

    # Display the final result
    numpy_horizontal_concat = np.concatenate((resized, seg_image), axis=1)
    numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, mask_out), axis=1)
    cv2.imwrite("Person Segmentation.jpg", numpy_horizontal_concat)

    #cv2.imshow('Person Segmentation',numpy_horizontal_concat)

    
    #cv2.imshow('frame', color_and_mask)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

