########## Model Parameters ##########
w = 2000            # Number of world cells.
c = 128             # Number of channels of the cell.
ch = 64             # Number of network channels.
down_size = 4       # Down-sampling size of view cell.
draw_layers = 12     # Number of layers of ConvDRAW.
share_core= True    # If share the core of ConvDRAW.

########## Experimental Parameters ##########
exp_name = "rrc"
data_path = "../GQN-Datasets-pt/rooms_ring_camera"
frac_train = 0.01
frac_test = 0.01
max_obs_size = 5
train_steps = 1600000
train_epochs = 400
