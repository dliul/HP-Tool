# coding: utf-8
import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import time
import json
import cv2
import re
from PIL import Image
import csv
import threading


def powerGet():
    global power_tmp
    power_tmp = []
    while 1 and con_thread:
        with os.popen('cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input') as process:
            output = process.read()
            power = float(int(output)) / 1000
            power_tmp.append(power)
            time.sleep(0.1)


def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    return t


def openImgs(img_dir_path, test_image_txt, normalize, height, width, channels, batch_size, load_number, current_number,
             total_numner):
    images = []
    image_batch = []
    (h, w, c) = (height, width, channels)
    have_load = current_number * load_number
    count = 0
    batch_count = 0
    if total_numner - have_load > load_number:
        current_load = load_number
    else:
        current_load = total_numner - have_load
    for i in range(current_load):
        image = Image.open(img_dir_path + test_image_txt[i + have_load])
        image = image.resize((w, h), resample=0)
        image = np.array(image)
        try:
            image = image.reshape((h, w, c))
        except ValueError:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = image.reshape((h, w, c))
        if normalize:
            image = image / 255.0

        if count % batch_size == 0:
            image_batch = []
        image_batch.append(image)
        if count % batch_size == (batch_size - 1):
            images.append(image_batch)
            batch_count = batch_count + 1
        count = count + 1

    if batch_count * batch_size < count:
        images.append(image_batch)

    return images


def run_main(model_path, img_dir_path, normalize, label_path, batch_size, test_number, results_name):
    test_image_txt = []
    temp_path = os.path.join(img_dir_path)
    for line in os.listdir(temp_path):
        test_image_txt.append(line)
    total_images_num = len(test_image_txt)
    total_iteration = int(total_images_num / test_number) + 1
    result_all = []
    global power_all
    global con_thread
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
            tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            tensor_values_list = [tensor.values() for tensor in tf.get_default_graph().get_operations()]
            input_tensor = tensor_name_list[0] + ":0"
            output_tensor = tensor_name_list[-1] + ":0"
            input_shape = tensor_values_list[0]

            a = str(input_shape)
            parse_items = a.split(' ')
            id_name1 = re.findall(r"\d+\.?\d*", parse_items[3])
            id_name2 = re.findall(r"\d+\.?\d*", parse_items[4])
            id_name3 = re.findall(r"\d+\.?\d*", parse_items[5])

            (height, width, channels) = (int(id_name1[0]), int(id_name2[0]), int(id_name3[0]))
    times = []
    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=detection_graph, config=config) as sess:
            image_tensor = detection_graph.get_tensor_by_name(output_tensor)
            for iteration in range(total_iteration):
                images_load = openImgs(img_dir_path, test_image_txt, normalize, height, width, channels, batch_size,
                                       test_number, iteration, total_images_num)
                for images in images_load:
                    start = time.time()
                    power_t = thread_it(powerGet)
                    con_thread = True
                    power_t.start()
                    predictions = sess.run(image_tensor, {input_tensor: images})
                    con_thread = False
                    power_all.append(power_tmp)
                    times.append(time.time() - start)
                    for precision in predictions:
                        result_all.append(precision)
                del images_load

    print(times)
    for item in power_all:
        print(sum(item)/len(item))
    anns = []
    class1 = []
    class5 = []
    id = []
    for i in range(len(result_all)):
        a = test_image_txt[i]
        parse_items = a.split('_')
        id_name1 = parse_items[2]
        par_id = id_name1.split('.')
        id_name = int(par_id[0])
        top_1 = result_all[i].argsort()[-1:][::-1]
        top_5 = result_all[i].argsort()[-5:][::-1]
        class1.append(top_1)
        class5.append(top_5)
        id.append(id_name)
        ann_i = {'id': id_name, 'class1': top_1.tolist(), 'class5': top_5.tolist()}
        anns.append(ann_i)

    labels = [2000]
    true_num = 0
    true_num5 = 0
    with open(label_path, 'rb') as file:
        for line in file:
            a = line.decode().strip().split(' ')
            labels.append(int(a[1]))
    for i in range(len(id)):
        if labels[id[i]] == class1[i][0]:
            true_num = true_num + 1
        for j in range(0, 5):
            if labels[id[i]] == class5[i][j]:
                true_num5 = true_num5 + 1
                break

    print('Top-1 Accuracy: ', true_num / len(id))
    print('Top-5 Accuracy: ', true_num5 / len(id))
    print("Execution time: ", sum(times[1:]))

    with open(results_name, 'w+', encoding='utf-8') as file:
        json.dump(anns, file, ensure_ascii=False)


if __name__ == "__main__":
    model_path = '../models/inception_v3_frozen_graph.pb'
    img_dir_path = '../dataSet5000/'
    normalize = 1
    precision = "GPU"
    results_name = 'results.json'
    label_path = '../models/CaffeImagenetIS2012Label.txt'
    test_number = 200
    batch_size = 2
    normalize = 1
    power_all = []
    power_tmp = []
    con_thread = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelpath", help="The path of the frozen model")
    parser.add_argument("-d", "--dataset", help="The path of the dataset needed to be detected")
    parser.add_argument("-n", "--normalization", help="If normalization")
    parser.add_argument("-p", "--precision", help="The precision mode")
    parser.add_argument("-l", "--labelpath", help="The path of the true label file")
    parser.add_argument("-b", "--batchsize", help="The size of a batch")
    parser.add_argument("-a", "--loadnumber", help="The number of pictures in every load")
    parser.add_argument("-r", "--resultname", help="The path of the result save file")

    args = parser.parse_args()

    if args.modelpath:
        model_path = args.modelpath
    if args.dataset:
        img_dir_path = args.dataset
    if args.normalization:
        normalize = int(args.normalization)
    if args.precision:
        precision = args.precision
    if args.labelpath:
        label_path = args.labelpath
    if args.batchsize:
        batch_size = int(args.batchsize)
    if args.loadnumber:
        test_number = int(args.loadnumber)
    if args.resultname:
        results_name = args.resultname

    if precision == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if precision != "CPU" and precision != "GPU":
        import tensorflow.contrib.tensorrt as trt

    run_main(model_path, img_dir_path, normalize, label_path, batch_size, test_number, results_name)
