import os
import argparse
import configparser

parser = argparse.ArgumentParser(description='Download GQN datasets.')
parser.add_argument('--dataset', nargs='?', type=str, default="rooms_ring_camera", help='Dataset name.')
parser.add_argument('--download_path', nargs='?', type=str, default="GQN-Datasets" ,help='Download path.')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read("datasets_handle/dataset.conf")
dataset_full_name = config.get(args.dataset, 'name')

print("Create downloading folder.")
if not os.path.exists(args.download_path):
    os.mkdir(args.download_path)
print("Done")


print("Download data.")
command_str = "gsutil -m cp -R -n gs://gqn-dataset/" + dataset_full_name + " " + args.download_path
print(command_str)
os.system(command_str)
print("Done")

#chmod -R 777 GQN-Datasets/
