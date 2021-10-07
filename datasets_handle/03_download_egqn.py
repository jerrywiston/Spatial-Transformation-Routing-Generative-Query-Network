import os
import argparse

parser = argparse.ArgumentParser(description='Download EGQN datasets.')
parser.add_argument('--dataset', nargs='?', type=str, default="disco", help='Dataset name.')
parser.add_argument('--download_path', nargs='?', type=str, default="EGQN-Datasets" ,help='Download path.')
parser.add_argument('--train_test_split', nargs='?', type=int, default=1944 ,help='Split of training and testing.')
args = parser.parse_args()

print("Create downloading folder.")
if not os.path.exists(args.download_path):
    os.mkdir(args.download_path)
print("Done")

print("Download data.")
command_str = "gsutil -m cp -R -n gs://egqn-datasets/" + args.dataset + " " + args.download_path
print(command_str)
os.system(command_str)
print("Done")

for i in range(2,1944):
    id = str(i).zfill(4)
    fn = id + "-of-1943.tfrecord"
    train_test = "train/"
    if i>1600:
        train_test="test/"
    os.system("mv "+args.download_path+fn+" "+args.download_path+train_test+fn)

