
# Scalable Modular Network: A Framework for Adaptive Learning via Agreement Routing
This repository contains the reference source code for the paper ["Scalable Modular Network: A Framework for Adaptive Learning via Agreement Routing"](https://openreview.net/forum?id=pEKJl5sflp) in the International Conference on Learning Representations (ICLR 2024). In this project, we implement the Scalable Modular Network (SMN) and other comparative methods for toy Min-Max Digital Game.

## Dependencies
The code is built with following libraries:
- python 3.7
- PyTorch 1.13.1
- cv2
- tqdm
- matplotlib
- h5py

## Installation
```install
conda create -n smn python=3.7
source activate smn
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Out-of-distribution Experiment
The main script is in the file ``ood_generalization/main.py``. To train the SMN model on the out-of-distribution scenario, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python ood_generalization/main.py --model SMN --routing_iter 4
```

or train other comparative models (CNN, Transformer, SMN_topk or SMN_truncated), run this command:

```train
CUDA_VISIBLE_DEVICES=0 python ood_generalization/main.py --model {the choice of model}
```

For evaluation, you can simply use the same command as during training, but set the train option to ``False``:

```evaluation
CUDA_VISIBLE_DEVICES=0 python ood_generalization/main.py --model SMN --routing_iter 4 --train False
```

## Few-shot Adaptation Experiment
The main script is in the file ``fs_adaptation/main.py``. To train the SMN model on the few-shot adaptation scenario, run this command:

```train
CUDA_VISIBLE_DEVICES=0 python fs_adaptation/main.py --model SMN --routing_iter 4
```

During adaptation, you can simply use the following command by specifiying the number of finetuned samples (15/30/45/60/75/90):

```adaptation
CUDA_VISIBLE_DEVICES=0 python fs_adaptation/main.py --model SMN --routing_iter 4 --finetune_num {the number of finetuned samples} --finetune_iter 100 --train False
```

Note that our SMN supports integration of new modules after pre-training for better adaptation.
To evaluate this, run this command:
```adaptation
CUDA_VISIBLE_DEVICES=0 python fs_adaptation/main.py --model SMN --routing_iter 4 --finetune_num 15 --finetune_iter 100 --finetune_lr 0.001 --add_num {the number of new modules} --train False
```

### Fast Start
Here we provide some checkpoints of SMN model for fast start.
- [SMN (T=4)](https://drive.google.com/file/d/1vwxocqyoL38cPyhgp1nU8EAM15y_ie5z/view?usp=sharing)

The above SMN checkpoints are trained on the few-shot adaptation scenario with T=4.
Please download the pretrained model at file path ``checkpoint/fs_adaptation/``, and then run the corresponding commands.

For example, to evaluate the performance of SMN with ``2 new modules``, 
```
CUDA_VISIBLE_DEVICES=0 python fs_adaptation/main.py --model SMN --routing_iter 4 --finetune_num 15 --finetune_iter 100 --finetune_lr 0.001 --add_num 0 --train False --log_dir smn_t4

CUDA_VISIBLE_DEVICES=0 python fs_adaptation/main.py --model SMN --routing_iter 4 --finetune_num 15 --finetune_iter 100 --finetune_lr 0.001 --add_num 2 --train False --log_dir smn_t4
```

## Some Things You Need to Know
 - As the Min-Max Digital Game is only a toy example for evaluation, we didn't focus much on hyperparameter tuning.
 However, we found that setting appropriate hyperparameters can improve the performance of our SMN (and other comparative methods) further.
 For example, reducing the finetuned learning rate or iterations helps the adaptation process of SMN.
 - From the aspect of supporting intergration of new modules, the agreement routing is still far from satisfactory. Hence if you are interested in imporving our work, you may start with this part.

## Citation
If our paper or project inspires your work, please consider citing our work using the bibtex:

```citation
@inproceedings{hu2024smn,
  title={Scalable Modular Network: A Framework for Adaptive Learning via Agreement Routing},
  author={Hu, Minyang and Chang, Hong and Ma, Bingpeng and Shan, Shiguang and Xilin, CHEN},
  booktitle={The International Conference on Learning Representations},
  year={2024}
}
```