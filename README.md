# Reward Model Routing in Alignment

## Train and Test Offline Router
cd gp_router
python train_offline_2encoder_100M2.py

## Train and eval BayesianRouter on MMLU:
gp_router
python train_bo_router4.py
