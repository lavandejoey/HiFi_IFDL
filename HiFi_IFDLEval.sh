#!/bin/bash
#SBATCH --job-name=HiFi_IFDLEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A40,L40S,A100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_HiFi_IFDL"
mkdir -p "${result_dir}"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMockBin"
data_entry_csv="/projects/hi-paris/DeepFakeDataset/frames_index.csv"
done_csv_list=("results")
#hrnet_path="weights/HRNet/225000.pth"
#nlcd_path="weights/NLCDetection/225000.pth"

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate hifi37

srun python3 -Wignore HiFi_IFDLEval.py \
--data_root "${data_root}" \
--results "${result_dir}" \
--data_csv ${data_entry_csv} \
--done_csv_list "${done_csv_list[@]}"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"