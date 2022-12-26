[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



# A²

Code for **"Automated Adversarial Training"**.

## Requisite

This code is implemented in PyTorch, and we have tested the code under the following environment settings:
- `python = 3.8.10`
- `pytorch = 1.9.0+cu111`
- `autoattack = 0.1`

## What is in this repository

- `auto_adv`: the core codes for A², contains the super-attacker in `adv_x.py`, the attacker space in `genotypes.py`, the attack cell in `cell.py`, the operations in `operations.py`, and the Gumbel Softmax in `sample.py`;
- `train_[cifar10, cifar100, svhn].py`: the codes for CIFAR-10, CIFAR-100, and SVHN respectively;
- other codes: the basic code from **AWP**. 

## How to run it

For A² with a PreAct ResNet-18 on CIFAR-10 under $L_{\infty}$ threat model (8/255), run codes as follows,
```
python train_cifar10.py --data-dir DATASET_DIR --model PreActResNet18 --awp-warmup 200 
```
where `$DATASET_DIR` is the path to the dataset.

For TRADES-A² with a WRN-34-10 on CIFAR10 under $L_{\infty}$ threat model (8/255), run codes as follows,
```
python train_cifar10.py --data-dir DATASET_DIR --model WideResNet --awp-warmup 200 --loss trades
```

For MART-A², just set `MART` to `$LOSS`.

For AWP-A² with a WRN-34-10 on CIFAR10 under $L_{\infty}$ threat model (8/255), run codes as follows,
```
python train_cifar10.py --data-dir DATASET_DIR --model WideResNet --awp-warmup 0
```


To verify the effectiveness of A² further, we run `attack.py` to evaluate the robustness of the defense model against FGSM, PGD, C&W and AutoAttack:
```
python attack.py --arch $ARCH --checkpoint $CKP --preprocess meanstd --attack fgsm pgd20 cw aa
```


## Reference Code:

- AT: https://github.com/locuslab/robust_overfitting
- TRADES: https://github.com/yaodongyu/TRADES/
- MART: https://github.com/YisenWang/MART
- AWP: https://github.com/csdongxian/AWP
- FAT: https://github.com/zjfheart/Friendly-Adversarial-Training
- AutoAttack: https://github.com/fra31/auto-attack# A2-efficient-automated-attacker-for-boosting-adversarial-training
