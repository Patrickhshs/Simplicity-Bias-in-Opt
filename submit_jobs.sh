#!/bin/bash
#SBATCH --job-name=simplicity_bias
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=44G
#SBATCH --gres=gpu:1



# Define the different configurations
# declare -a num_conv_layers=(2 3)
# declare -a num_filters=("6,16" "6,16,32")
declare -a num_fc_layers=(1 10 100)
declare -a num_neurons=("1000,1000,1000" "100,100,100" "10,10,10")
declare -a num_overlay=(1 2 3 4 5 6)

# Iterate over each configuration and run the training script
overlay_index=0
for fc_layers in "${num_fc_layers[@]}"; do
  for neurons in "${num_neurons[@]}"; do
    overlay_file="/scratch/js12556/Simplicity-Bias-in-Opt/overlays/MMMB-SR-overlay/overlay-file_${num_overlay[$overlay_index]}.ext3"
    save_dir="/scratch/js12556/Simplicity-Bias-in-Opt/results/fc_layers_${fc_layers}_neurons_${neurons}"
    echo "Running with fc_layers=$fc_layers, neurons=$neurons, overlay_file=$overlay_file"
    singularity exec --nv --bind /scratch/$USER --overlay $overlay_file:rw /scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
    source /scratch/js12556/MMMB-SR/ext3/env.sh
    conda activate simplicity-bias
    cd /scratch/js12556/Simplicity-Bias-in-Opt
    python training.py --num_fc_layers $fc_layers --num_neurons \"$neurons\" --save_dir \"$save_dir\""
    overlay_index=$((overlay_index + 1))
  done
done