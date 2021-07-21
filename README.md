# CH-Zero
## Generating data
### To generate validation and test data:
1.  python generate_data.py --problem all --name validation --seed 1
2.  python generate_data.py --problem all --name test --seed 2
### Training (You can run different nodes by changing the task name)
python run.py --graph_size 20 --baseline rollout --run_name 'tsp20' --val_dataset data/tsp/tsp20_validation_seed1.pkl
### Multiple GPUs (Set the environment variable CUDA_VISIBLE_DEVICES to only use specific GPUs)
CUDA_VISIBLE_DEVICES=2,3 python run.py 
### Evaluation
python eval.py data/tsp/tsp20_test_seed2.pkl --model pretrained/tsp_20 --decode_strategy greedy

