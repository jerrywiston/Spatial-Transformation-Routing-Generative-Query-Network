import os
import argparse
import configparser
from tfrecord_converter import *

parser = argparse.ArgumentParser(description='Convert tfrecord to pt file.')
parser.add_argument('--dataset', nargs='?', type=str, default="rooms_ring_camera", help='Dataset name.')
parser.add_argument('--download_path', nargs='?', type=str, default="GQN-Datasets" ,help='Download path.')
parser.add_argument('--convert_path', nargs='?', type=str, default="GQN-Datasets-pt" ,help='Convert path.')
parser.add_argument('--batch', nargs='?', type=int, default=32, help='Batch size.')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read("datasets_handle/dataset.conf")
dataset_full_name = config.get(args.dataset, 'name')
dataset_args = {}
dataset_args['image_size'] = config.getint(args.dataset, 'image_size')
dataset_args['cam_params'] = config.getint(args.dataset, 'cam_params')
dataset_args['sequence_size'] = config.getint(args.dataset, 'sequence_size')
print(dataset_args)
tf.enable_eager_execution()
###############################################
print("Convert training data.")
convert_dir_train = os.path.join(args.convert_path, dataset_full_name, "train")
if not os.path.exists(convert_dir_train):
    os.makedirs(convert_dir_train)

data_dir = os.path.join(args.download_path, dataset_full_name, "train")
records = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
records = [f for f in records if "tfrecord" in f and "gstmp" not in f]
for r in records:
    convert(r, batch_size=args.batch, out_path=convert_dir_train, args=dataset_args)
#with mp.Pool(processes=mp.cpu_count()) as pool:
#    f = partial(convert, batch_size=args.batch, out_path=convert_dir_train)
#    pool.map(f, records)
print("Done")

###############################################
print("Convert testing data.")
convert_dir_test = os.path.join(args.convert_path, dataset_full_name, "test")
if not os.path.exists(convert_dir_test):
    os.makedirs(convert_dir_test)

data_dir = os.path.join(args.download_path, dataset_full_name, "test")
records = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]
records = [f for f in records if "tfrecord" in f and "gstmp" not in f]
for r in records:
    convert(r, batch_size=args.batch, out_path=convert_dir_train, args=dataset_args)
#with mp.Pool(processes=mp.cpu_count()) as pool:
#    f = partial(convert, batch_size=args.batch, out_path=convert_dir_test)
#    pool.map(f, records)
print("Done")