config_args = lambda: None

########## Model Parameters ##########
config_args.w = 2000            # Number of world cells.
config_args.c = 128             # Number of channels of the cell.
config_args.ch = 64             # Number of network channels.
config_args.down_size = 4       # Down-sampling size of view cell.
config_args.draw_layers = 6     # Number of layers of ConvDRAW.
config_args.share_core= True    # If share the core of ConvDRAW.

########## Experimental Parameters ##########
config_args.exp_name = "rrc"
config_args.data_path = "../GQN-Datasets-pt/rooms_ring_camera"
config_args.frac_train = 0.01
config_args.frac_test = 0.01
config_args.max_obs_size = 5
config_args.train_steps = 1600000
config_args.train_epochs = 400
