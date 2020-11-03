# Region-based Non-local operation for Video Classification

## Data Preparation

Please refer to [TSM](https://github.com/mit-han-lab/temporal-shift-module) repo for the details of data preparation.

## Training 

We provided several examples to train RNL network with this repo:

- To train on Something-Something V1 from ImageNet pretrained models, you can run the script bellow:
 ```bash
python main.py --dist-url 'tcp://localhost:6666' --dist-backend 'nccl' \
--multiprocessing-distributed --available_gpus 0,1,2,3 --world-size 1 --rank 0 \
--dataset something --gd 20 --shift --shift_div=8 --shift_place=blockres --npb \
--lr 0.02 --wd 1e-3 --dropout 0.8 --num_segments 8 --batch_size 16 --batch_multiplier 4\
--use_warmup --warmup_epochs 1 --lr_type cos --epochs 50 --non_local  --suffix 1

# The total batch size is equal to batch_size x batch_multiplier
#Notice that you should scale up the learning rate with batch size. For example, if you use a batch size of 128 you should set learning rate to 0.04.
  ```


