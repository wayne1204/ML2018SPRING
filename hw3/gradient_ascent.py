#!/usr/bin/env python

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
#from marcos import *

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step, input_image_data, iter_func, learning_rate=.05, record_freq=50):

    filter_images = []
    for i in range(num_step):
        target, grads_val = iter_func([input_image_data, 0])
        input_image_data += grads_val * learning_rate

        #print('current target: {}'.format(target))

        if i % record_freq == 0:
            filter_images.append((input_image_data, target))

    return filter_images

def main():
    parser = argparse.ArgumentParser(prog='gradient_ascent.py',
            description='ML-Assignment3 filter visualization via gradient ascent.')
    parser.add_argument('--model', type=str, metavar='<#model>', required=True)
    # parser.add_argument('--attr', type=str, metavar='<#attr>', required=True)
    parser.add_argument('--step', type=int, metavar='<#step>', default=300)
    parser.add_argument('--record', type=int, metavar='<#record>', default=50)
    parser.add_argument('--filter_dir', type=str, metavar='<#filter_dir>', default='./image/filter')
    args = parser.parse_args()

    num_steps = args.step
    record_freq = args.record
    # attr_name = args.attr
    model_name = args.model
    filter_dir = args.filter_dir

    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)

    # print('load attr')
    # attr = np.load(attr_name)
    # mean, std = attr[0], attr[1]

    print('load model')
    emotion_classifier = load_model(model_name)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
    input_img = emotion_classifier.input

    name_ls = ['conv2d_1','conv2d_3','conv2d_5']
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    nb_filter = 32

    for cnt, c in enumerate(collect_layers):
        print('Process layer {}'.format(name_ls[cnt]))
        filter_imgs = []
        for filter_idx in range(nb_filter):
            print('-> Process filter {}'.format(filter_idx))
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise

            # input_img_data = (input_img_data - mean) / (std + 1e-20)
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img, K.learning_phase()], [target, grads])

            _filter_imgs = grad_ascent(
                    num_steps, 
                    input_img_data, 
                    iterate,
                    record_freq=record_freq)

            filter_imgs.append(_filter_imgs)

        for it in range(num_steps//record_freq):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/8, 8, i+1)
                origin_img = filter_imgs[i][it][0].reshape(48, 48, 1)
                # origin_img = (origin_img * std + mean) * 255
                origin_img = np.clip(origin_img, 0, 255).astype('uint8')
                ax.imshow(origin_img.squeeze(), cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[i][it][1]))
                plt.tight_layout()
            #fig.suptitle('Filters of layer {} (# Ascent Epoch {})'.format(name_ls[cnt], it*record_freq))
            img_path = os.path.join(filter_dir, '{}'.format(name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*record_freq)))

if __name__ == "__main__":
    main()
