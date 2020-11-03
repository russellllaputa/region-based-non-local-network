# Region-based Non-local operation for Video Classification

## Data Preparation

Please refer to [TSM](https://github.com/mit-han-lab/temporal-shift-module) repo for the details of data preparation.


## Pretrained Models

Training video models is computationally expensive. Here we provide some of the pretrained models. The accuracy might vary a little bit compared to the paper, since we re-train some of the models.

### Kinetics-400

#### Dense Sample

In the latest version of our paper, we reported the results of TSM trained and tested with **I3D dense sampling** (Table 1&4, 8-frame and 16-frame), using the same training and testing hyper-parameters as in [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) paper to directly compare with I3D. 

We compare the I3D performance reported in Non-local paper:

| method          | n-frame      | Kinetics Acc. |
| --------------- | ------------ | ------------- |
| I3D-ResNet50 NL | 32 * 10clips | 74.9%         |
| TSM-ResNet50 RNL | 8 * 10clips  | 75.6%     |
| TSM-ResNet50 RNL | 16 * 10clips  | 77.2%     |
| TSM-ResNet50 RNL | (16+8) * 10clips  | 77.4%     |

RNL TSM models achieve better performance than NL I3D model with less computation (shorter video length).


| model             | n-frame     | Kinetics Acc. | checkpoint                                                   | download code|
| ----------------- | ----------- | ------------- | ------------------------------------------------------------ | -------------|
| TSM ResNet50 NL   | 8 * 10clips | 75.6%         | [link](https://pan.baidu.com/s/1WnepknYcxwGzzxHl52n0tQ) | 3573|


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
# Notice that you should scale up the learning rate with batch size. 
# For example, if you use a batch size of 128 you should set learning rate to 0.04.
  ```
  
  - To train on Kinetics from ImageNet pretrained models, you can run the script bellow:
 ```bash
python main.py --dataset kinetics  --dense_sample --dist-url 'tcp://localhost:6666' \
--dist-backend 'nccl' --multiprocessing-distributed --available_gpus 0,1,2,3 --world-size 1 \
--rank 0 --gd 20 --shift --shift_div=8 --shift_place=blockres --npb --lr 0.02 --wd 2e-4 \
--dropout 0.5 --num_segments 8 --batch_size 16 --batch_multiplier 4 --use_warmup --warmup_epochs 5 \
--lr_type cos --epochs 100 --non_local  --suffix 1

# The total batch size is equal to batch_size x batch_multiplier
# Notice that you should scale up the learning rate with batch size. 
# For example, if you use a batch size of 128 you should set learning rate to 0.04.
```

## Test 

For example, to test the downloaded pretrained models on Kinetics, you can run the scripts below. The scripts test RNL on 8-frame setting by running:

```bash
# test on Something
python test_models.py something \
--weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e50_cos_nl_lr0.02_wd6.0e-04_1.pth.tar \
--test_segments=8 --batch_size=2 -j 25 --test_crops=3  --twice_sample  --full_res

# test on kinetics
python test_models.py kinetics  \
--weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50_cos_nl_lr0.02_wd6.0e-04_1.pth.tar \
--test_segments=8 --batch_size=16 -j 25 --test_crops=3  --dense_sample --full_res
```
