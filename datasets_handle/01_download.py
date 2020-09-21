import os
import argparse

parser = argparse.ArgumentParser(description='Download GQN datasets.')
parser.add_argument('--dataset', nargs='?', type=str, default="rooms_ring_camera", help='Dataset name.')
parser.add_argument('--download_path', nargs='?', type=str, default="GQN-Datasets" ,help='Download path.')
args = parser.parse_args()

print("Create downloading folder.")
if not os.path.exists(args.download_path):
    os.mkdir(args.download_path)
print("Done")

print("Download data.")
command_str = "gsutil -m cp -R -n gs://gqn-dataset/" + args.dataset + " " + args.download_path
print(command_str)
os.system(command_str)
print("Done")

#chmod -R 777 GQN-Datasets/