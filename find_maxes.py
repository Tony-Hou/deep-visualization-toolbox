#! /usr/bin/python
import argparse
import cPickle as pickle
import sys

import lmdb
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

sys.path.insert(0, 'find_maxes')
from max_tracker import NetMaxTracker
from caffe_misc import RegionComputer

sys.path.insert(0, '/home/ubuntu/new/caffe/python')
import caffe


def lmdb_reader(fpath):
    env = lmdb.open(fpath)
    txn = env.begin()
    cursor = txn.cursor()

    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum).astype(np.uint8)
        yield (key, image, label)


def read_mean(fpath):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(fpath, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))[0]
    return arr.mean(1).mean(1)


def preprocess_image(image, net, mean):
    in_dims = net.blobs['data'].data.shape[1:]
    caffe_in = resize(image, in_dims)  # resize to input dimensions
    caffe_in = caffe_in[(2, 1, 0), :, :]  # reorder channels (for instance color to BGR)
    caffe_in *= 255  # scale raw input
    caffe_in -= mean[:, np.newaxis, np.newaxis]  # subtract mean
    return caffe_in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/home/ubuntu/new/caffe/data/marketplace/googlenet/deploy.prototxt')
    parser.add_argument('--weights', type=str,
                        default='/home/ubuntu/new/caffe/data/marketplace/googlenet/snapshots/googlenet_train_quick_iter_5000.caffemodel')
    parser.add_argument('--lmdb', type=str,
                        default='/home/ubuntu/new/caffe/data/marketplace/images/marketplace_val_lmdb')
    parser.add_argument('--mean', type=str,
                        default='/home/ubuntu/new/caffe/data/marketplace/images/mean.binaryproto')
    parser.add_argument('--layer', type=str, default='prob')
    parser.add_argument('--channel', type=int, default=0)
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    reader = lmdb_reader(args.lmdb)
    mean = read_mean(args.mean)

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    try:
        max_tracker = pickle.load(open('maxes.pkl', 'rb'))
    except:
        layers = map(str, net._blob_names)
        is_conv = [False for ll in layers]
        max_tracker = NetMaxTracker(layers=layers, is_conv=is_conv, n_top=10)

        for idx, (i, image, label) in enumerate(reader):
            print idx, i, label

            caffe_in = preprocess_image(image, net, mean)
            net.blobs['data'].data[...] = caffe_in

            net.forward()
            max_tracker.update(net, idx, label)

        with open('maxes.pkl', 'wb') as f:
            pickle.dump(max_tracker, f, -1)

    reader = lmdb_reader(args.lmdb)
    files = [i.split('_', 1)[1] for (i, _, _) in reader]

    layer = args.layer
    channel = args.channel
    mt = max_tracker.max_trackers[layer]
    rc = RegionComputer()

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(3, 3, wspace=0, hspace=0)
    for i, im in enumerate(mt.max_locs[channel, -9:, 0]):
        img = mpimg.imread('/home/ubuntu/new/caffe/data/marketplace/images/' + files[im])
        if mt.is_conv:
            ii, jj = mt.max_locs[channel, i+1, 2:]
            layer_indices = (ii, ii + 1, jj, jj + 1)
            data_indices = rc.convert_region(layer, 'data', layer_indices)
            data_ii_start, data_ii_end, data_jj_start, data_jj_end = data_indices
            img = img[data_ii_start:data_ii_end, data_jj_start:data_jj_end, :]
        ax = plt.subplot(gs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, aspect='auto')
        fig.add_subplot(ax)
    plt.tight_layout(pad=0)
    plt.show()


if __name__ == "__main__":
    main()
