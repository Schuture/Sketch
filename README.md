# Cetegory-level Sketch-based Image Retrieval

### 0. 代码位置

Standard CSBIR的代码在`master`分支里，Zero-shot CSBIR的代码在`zs`分支里

### 1. 数据集

1. 目前数据集在`10.141.209.43`和`10.88.3.92`这两台服务器上有。代码如果在这两台上跑就不用传数据了。

   - `10.141.209.43`

     数据在`/home/lupeng/dataset/sketch/TUBerlin`和`/home/lupeng/dataset/sketch/sketchy`

   - `10.88.3.92`

     数据在`/home/lp_user/dataset/sketch/TUBerlin`和`/home/lp_user/dataset/sketch/sketchy`

### 2. Standard CSBIR的各种设定

- 网络结构

  - 基础结构
    - 由参数`model_type`控制
    - 包括
      - ResNeXt101(默认)(`--model_type=resnext101`)
      - ResNet101(`--model_type=resnet`)
      - DenseNet121(`--model_type=densenet`)
      - VGG16(`--model_type=vgg`)
      - AlexNet(`--model_type=alexnet`)
  - SE模块（只对ResNeXt101, ResNet101有效）
    - 不加SE模块: `--SE=False --with_domain=False`
    - 加SE模块(默认): `--SE=True --with_domain=False`
    - 加CSE模块: `--SE=True --with_domain=True`
  - Branching stage（只对ResNeXt101, ResNet101有效）
    - 按照paper，想要merging stage 是$n$的话，这里的命令是`--branching_stage=(n-1)`
    - 默认值为$n=0$, 即没有分支`--branching_stage=-1`
    - 和SE/CSE模块不兼容

- Loss

  | Loss      | `--loss_type=` | `--margin=` | `--scale=` | `--loss_ratio=` | `--test_distance` |
  | --------- | -------------- | ----------- | ---------- | --------------- | ----------------- |
  | EMS       | `sqreudmargin` | `4`         | /          | `1.0`           | `eud`             |
  | Softmax   | `softmax`      | /           | /          | `1.0`           | `eud`             |
  | A-Softmax | `sphere`       | `4`         | /          | `1.0`           | `cos`             |
  | LMCL      | `cos`          | `0.35`      | `30`       | `1.0`           | `cos`             |

  '/'表示是多少无所谓

- 其他设定

  - phase: `--phase=train (or test)` 训练完会自动test一次，所以一般设定成train就可以了
  - gpu_id: `--gpu_id=0 (or 1,2,3,...)` 目前只支持单卡训练
  - root: `--root=exprs/TUBerlin/rnx_cse-sqrem_m4_r1` 给每次的训练取一个名字，模型、结果都保留在相应文件夹里
  - obj: `--obj=TUBerlin (or sketchy)`

- 主要结果

  - CSE-ResNeXt101 + EMS with margin 4

    ```bash
    python main.py \
    --mode=std --phase=train --gpu_id=0 \
    --data_root=/home/lupeng/sketch/TUBerlin --obj=TUBerlin \
    --model_type=resnext101 --SE=True --with_domain=True \
    --loss_type=sqreudmargin --margin=4 --loss_ratio=1 \
    --root=exprs/TUBerlin/rnx_cse-sqrem_m4_r1
    ```

    ```bash
    python main.py \
    --mode=std --phase=train --gpu_id=0 \
    --data_root=/home/lupeng/sketch/sketchy --obj=sketchy \
    --model_type=resnext101 --SE=True --with_domain=True \
    --loss_type=sqreudmargin --margin=4 --loss_ratio=1 \
    --root=exprs/TUBerlin/rnx_cse-sqrem_m4_r1
    ```

- Hashing

  Hashing过程是在训练结束之后的. 代码在`hash.py`里。主要参数有

  - root: 必须要和上面的root保持一致
  - loss_ratio: 这里要输入三个小数，比如`'1,0.3,1'`, 分别表示文章中的$r, q, s$三个loss的比例
  - code_dim: binary code的长度

  例子

  ```bash
  python hash.py \
  --root=exprs/TUBerlin/rnx_cse-sqrem_m4_r1 \
  --gpu_id=0 \
  --code_dim=64 \
  --loss_ratio='1,0,1'
  ```

### 3. 现在的代码为了ablation study各种奇怪的参数还比较多，后面放出来的代码我会整理的简单一下，只放能跑主要结果的部分(CSE-ResNeXt101 + EMS with margin 4 + Hashing)