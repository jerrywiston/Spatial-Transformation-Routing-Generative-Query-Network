"""
tfrecord-converter

Takes a directory of tf-records with Shepard-Metzler data
and converts it into a number of gzipped PyTorch records
with a fixed batch size.

Thanks to l3robot and versatran01 for providing initial
scripts.

Requires tensorflow 1.13.1 + pytorch 1.0.0 Environment
For Conda:
conda install tensorflow==1.13.1
conda install pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 cpuonly -c pytorch
"""
import os, gzip, torch
import tensorflow as tf, numpy as np, multiprocessing as mp
from functools import partial
from itertools import islice, chain
from argparse import ArgumentParser
import os

# disable logging and gpu
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#POSE_DIM, IMG_DIM, SEQ_DIM = 5, 64, 15
#POSE_DIM, IMG_DIM, SEQ_DIM = 5, 64, 10
#POSE_DIM, IMG_DIM, SEQ_DIM = 5, 128, 4
#POSE_DIM, IMG_DIM, SEQ_DIM = 5, 128, 10
default_args = lambda: None
default_args.pose_dim = 5
default_args.img_dim = 64
default_args.seq_dim = 10

def chunk(iterable, size=10):
    """
    Chunks an iterator into subsets of
    a given size.
    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))

def process(record, args):
    """
    Processes a tf-record into a numpy (image, pose) tuple.
    """
    kwargs = dict(dtype=tf.uint8, back_prop=False)
    for data in tf.python_io.tf_record_iterator(record):
        instance = tf.parse_single_example(data, {
            'frames': tf.FixedLenFeature(shape=args.seq_dim, dtype=tf.string),
            'cameras': tf.FixedLenFeature(shape=args.seq_dim * args.pose_dim, dtype=tf.float32)
        })

        # Get data
        images = tf.concat(instance['frames'], axis=0)
        poses  = instance['cameras']

        # Convert
        images = tf.map_fn(tf.image.decode_jpeg, tf.reshape(images, [-1]), **kwargs)
        images = tf.reshape(images, (-1, args.seq_dim, args.img_dim, args.img_dim, 3))
        poses  = tf.reshape(poses,  (-1, args.seq_dim, args.pose_dim))

        # Numpy conversion
        images, poses = images.numpy(), poses.numpy()
        yield np.squeeze(images), np.squeeze(poses)

def convert(record, batch_size, out_path, args=default_args):
    """
    Processes and saves a tf-record.
    """
    path, filename = os.path.split(record)
    basename, *_ = os.path.splitext(filename)
    print(basename)

    batch_process = lambda r: chunk(process(r, args), batch_size)

    for i, batch in enumerate(batch_process(record)):
        p = os.path.join(out_path, "{0:}-{1:02}.pt.gz".format(basename, i))
        with gzip.open(p, 'wb') as f:
            torch.save(list(batch), f)

if __name__ == '__main__':
    tf.enable_eager_execution()
    parser = ArgumentParser(description='Convert gqn tfrecords to gzip files.')
    parser.add_argument('base_dir', nargs=1,
                        help='base directory of gqn dataset')
    parser.add_argument('dataset', type=str, default="shepard_metzler_5_parts",
                        help='datasets to convert, eg. shepard_metzler_5_parts')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='number of sequences in each output file')
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='whether to convert train or test')
    args = parser.parse_args()

    # Find path
    base_dir = os.path.expanduser(args.base_dir[0])
    data_dir = os.path.join(base_dir, args.dataset, args.mode)

    # Find all records
    records = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
    records = [f for f in records if "tfrecord" in f]
    print(records)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        f = partial(convert, batch_size=args.batch_size)
        pool.map(f, records)
