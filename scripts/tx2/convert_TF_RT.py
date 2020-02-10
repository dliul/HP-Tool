import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
import re
import random
import shutil
import os


def getCalibImg(calib_img_dir, normalization, height, width, channels):
    pathDir = os.listdir(calib_img_dir)
    sample = random.sample(pathDir, 1)
    image = Image.open(calib_img_dir+sample[0])
    image = image.resize((height, width), resample=0)
    image = np.array(image)
    try:
        image = image.reshape((height, width, channels))
    except ValueError:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.reshape((height, width, channels))
    if normalization:
        image = image / 255.0
    images = []
    images.append(image)
    return images


def convert_main(model_path, calibration_image, normalize, precision):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.gfile.GFile(model_path, 'rb') as f:
            frozen_graph = tf.GraphDef()
            frozen_graph.ParseFromString(f.read())
            tf.import_graph_def(frozen_graph, name='')

        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        tensor_values_list = [tensor.values() for tensor in tf.get_default_graph().get_operations()]
        input_tensor = tensor_name_list[0] + ":0"
        output_tensor = tensor_name_list[-1]
        input_shape = tensor_values_list[0]
        a = str(input_shape)
        parse_items = a.split(' ')
        id_name1 = re.findall(r"\d+\.?\d*", parse_items[3])
        id_name2 = re.findall(r"\d+\.?\d*", parse_items[4])
        id_name3 = re.findall(r"\d+\.?\d*", parse_items[5])

        (height, width, channels) = (int(id_name1[0]), int(id_name2[0]), int(id_name3[0]))
        print(input_tensor, output_tensor, height, width, channels)

        if precision == "FP32":
            # FP32
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph, nodes_blacklist=[output_tensor],
                                              precision_mode="FP32")
            trt_graph = converter.convert()

            with tf.gfile.GFile(model_path[0:-3] + "_FP32.pb", 'wb') as f:
                f.write(trt_graph.SerializeToString())

        if precision == "FP16":
            # FP16
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph, nodes_blacklist=[output_tensor],
                                              precision_mode="FP16")
            trt_graph = converter.convert()

            with tf.gfile.GFile(model_path[0:-3] + "_FP16.pb", 'wb') as f:
                f.write(trt_graph.SerializeToString())

        if precision == "INT8":
            # INT8
            images = getCalibImg(calibration_image, normalize, height, width, channels)
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph, nodes_blacklist=[output_tensor],
                                              precision_mode="INT8", use_calibration=True)
            trt_graph = converter.convert()

            trt_graph = converter.calibrate(fetch_names=[output_tensor], num_runs=1,
                                            feed_dict_fn=lambda: {input_tensor: images})

            for n in trt_graph.node:
                if n.op == "TRTEngineOp":
                    print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
                    with tf.gfile.GFile("%s.calib_table" % (n.name.replace("/", "_")), 'wb') as f:
                        f.write(n.attr["calibration_data"].s)

            with tf.gfile.GFile(model_path[0:-3] + "_INT8.pb", 'wb') as f:
                f.write(trt_graph.SerializeToString())

    print("TensorRT model is successfully stored!")


if __name__ == "__main__":

    model_path = '../models/inception_v3_frozen_graph.pb'
    calibration_image = '../dataSet5000/'
    normalize = 1
    precision = "FP16"

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelpath", help="The path of the frozen model")
    parser.add_argument("-c", "--calib_img", help="The image for calibration")
    parser.add_argument("-n", "--normalization", help="If need normalization")
    parser.add_argument("-p", "--precision", help="precision mode")

    args = parser.parse_args()

    if args.modelpath:
        model_path = args.modelpath
    if args.calib_img:
        calibration_image = args.calib_img
    if args.normalization:
        normalize = int(args.normalization)
    if args.precision:
        precision = args.precision

    convert_main(model_path, calibration_image, normalize, precision)
