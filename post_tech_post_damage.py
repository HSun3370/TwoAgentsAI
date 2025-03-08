##################################################################
##################################################################
#### This program solves post-tech post-damage models
##################################################################
##################################################################
##################################################################

import model 
import numpy as np
import tensorflow as tf 
import sys 
import pathlib
import json 

#############################################
#############################################
#############################################
#############################################
#############################################

## Load parameters
export_folder                    = sys.argv[1] 
batch_size                       = int(sys.argv[2])
num_iterations                   = int(sys.argv[3])
pretrained_path                  = sys.argv[4]
logging_frequency                = int(sys.argv[5])
learning_rates                   = [float(x) for x in sys.argv[6].split(",")]
hidden_layer_activations         = sys.argv[7].split(",")
output_layer_activations         = sys.argv[8].split(",")
num_hidden_layers                = int(sys.argv[9])
num_neurons                      = int(sys.argv[10])
learning_rate_schedule_type      = sys.argv[11]
export_folder_output             = sys.argv[12]
 
 
tensorboard="True"



## Take care of pretrained path
if pretrained_path == "None":
    pretrained_path = None
else:
    pretrained_path = pretrained_path 

## Take care of activation functions 
hidden_layer_activations   = [None if x == "None" else x for x in hidden_layer_activations]
output_layer_activations   = [None if x == "None" else x for x in output_layer_activations]

#############################################
## Part 1
## Solve post tech post damage model
#############################################

## This model has three state variables

v_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_nn"}
v_nn_config["final_activation"] = output_layer_activations[0]

i_g_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[1], "dim" : 1, "nn_name" : "i_g_nn"}
i_g_nn_config["final_activation"] = output_layer_activations[1]

i_a_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[2], "dim" : 1, "nn_name" : "i_a_nn"}
i_a_nn_config["final_activation"] = output_layer_activations[2]

i_d_nn_config = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[3], "dim" : 1, "nn_name" : "i_d_nn"}
i_d_nn_config["final_activation"] = output_layer_activations[3]

## Network for Labor Allocation 
## 3 outputs (L_g,L_a, N) in order
L_nn_config   = {"num_hiddens": [num_neurons for _ in range(num_hidden_layers)], "use_bias": True, "activation": "relu", "dim": 3,  "nn_name": "Labor_nn", "final_activation": "softmax" }



## Prices network
v_g_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_g_nn"}
v_g_nn_config["final_activation"] = output_layer_activations[0]

v_a_nn_config   = {"num_hiddens" : [num_neurons for _ in range(num_hidden_layers)], "use_bias" : True, "activation" : hidden_layer_activations[0], "dim" : 1, "nn_name" : "v_a_nn"}
v_a_nn_config["final_activation"] = output_layer_activations[0]



## Create params struct 
params = {"batch_size" : batch_size,  
"v_nn_config" : v_nn_config, "v_g_nn_config" : v_g_nn_config, "v_a_nn_config" : v_a_nn_config, 
"i_g_nn_config" : i_g_nn_config,"i_a_nn_config" : i_a_nn_config,  "i_d_nn_config" : i_d_nn_config,  "L_nn_config" : L_nn_config ,
"num_iterations" : num_iterations, "logging_frequency": logging_frequency, "verbose": True, "load_parameters" : None, "norm_weight" : 0.9,
"pretrained_path" : pretrained_path, 'tensorboard' : tensorboard, "learning_rate_schedule_type" : learning_rate_schedule_type }

 



if params["learning_rate_schedule_type"] == "None":
    lr_schedulers = learning_rates
    params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "piecewiseconstant":
    boundaries            = [int(round(x)) for x in np.linspace(0,num_iterations,5)][1:-1]
    values_list           = [[learning_rate / np.power(2,x) for x in range(len(boundaries)+1)] for learning_rate in learning_rates]
    lr_schedulers         = [ tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
    params["optimizers"] = [tf.keras.optimizers.Adam( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "sgd+piecewiseconstant":
    boundaries            = [int(round(x)) for x in np.linspace(0,num_iterations,5)][1:-1]
    values_list           = [[learning_rate / np.power(2,x) for x in range(len(boundaries)+1)] for learning_rate in learning_rates]
    lr_schedulers         = [ tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values) for values in values_list]
    params["optimizers"] = [tf.keras.optimizers.legacy.SGD( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]
elif params["learning_rate_schedule_type"] == "sgd":
    lr_schedulers = learning_rates
    params["optimizers"] = [tf.keras.optimizers.legacy.SGD( learning_rate = lr_scheduler) for lr_scheduler in lr_schedulers]

 
params["export_folder"]  = export_folder +  "/TrainingResults"

params["phi_g"] = 16.7
params["phi_d"] = 16.7
if output_layer_activations[1] == "custom" or output_layer_activations[2] == "custom":
    params["i_g_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_g"]) / (tf.exp(2 * x) + 1.0)
    params["i_d_nn_config"]["final_activation"] = lambda x: 1.0 - (1.0 + 1.0/ params["phi_d"]) / (tf.exp(2 * x) + 1.0)


  
  
test_model = model.model(params)
test_model.export_parameters()
test_model.train()
 

