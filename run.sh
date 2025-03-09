#!/bin/bash

# pre_tech_pre_damage_export_folder="model_original"
 


prefix="/project/lhansen/Capital_NN_variant/TwoAgentsDataFeedback/output"

batch_size="128"
num_iterations="200000" 
# num_iterations="1000000" 
# num_iterations="100000" 
# num_iterations="300000" 
logging_frequency="1000"
learning_rates="40e-5,40e-5,40e-5,40e-5"  # Larger learning rate is much efficient
# learning_rates="40e-3,40e-3,40e-3,40e-3" 
# learning_rates="10e-3,10e-3,10e-3,10e-3"
learning_rates="10e-4,10e-4,10e-4,10e-4"
# learning_rates="40e-7,40e-7,40e-7,40e-7"
# learning_rates="40e-6,40e-6,40e-6,40e-6"
hidden_layer_activations="swish,tanh,tanh,tanh"
# hidden_layer_activations="relu,relu,relu,relu"
# hidden_layer_activations="tanh,tanh,tanh,tanh"
# output_layer_activations="softplus,tanh,tanh,softplus"
# output_layer_activations="softplus,tanh,tanh,tanh"
output_layer_activations="softplus,sigmoid,sigmoid,sigmoid"
# output_layer_activations="softplus,custom,custom,softplus"
# output_layer_activations="softplus,custom,custom,sigmoid"
# output_layer_activations="softplus,softplus,softplus,softplus"
num_hidden_layers="4"
num_neurons="32"
learning_rate_schedule_type="piecewiseconstant"

 
tensorboard='True'

foldername="WithoutScalers_${learning_rate_schedule_type}_${learning_rates}_num_iterations${num_iterations}_hidden_layer_activations_${hidden_layer_activations}_output_layer_activations_${output_layer_activations}"
pretrained_export_folder="None"



job_name="${prefix}/${foldername}"
jobout_name="${foldername}"

echo "Export folder: $pre_tech_pre_damage_export_folder"

job_file="Two_Agents.job"
job_output_file="Two_Agents.out"

echo "#!/bin/bash
#SBATCH --job-name=AI_Economy
#SBATCH --output=${job_output_file}
#SBATCH --error=run.err
#SBATCH --time=0-36:00:00
#SBATCH --account=pi-lhansen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
module load tensorflow/2.1
module unload cuda
module unload python
module load cuda/11.2
module load python/anaconda-2021.05
 
python3 post_tech_post_damage.py $job_name  $batch_size $num_iterations $pretrained_export_folder $logging_frequency $learning_rates $hidden_layer_activations $output_layer_activations $num_hidden_layers $num_neurons $learning_rate_schedule_type $foldername" > $job_file

sbatch $job_file


