# 第三组：基于 EGMN 的视频观看时长预测与时长感知门控

本文档是全组的实施说明。工作按阶段交接：后一阶段只使用前一阶段已经达到完成标准的产出。某一项没有达到该节的「做到什么程度」，后面的实验和报告就不使用那一项的结果。

## 我们要做什么

复现论文 _Multi-Granularity Distribution Modeling for Video Watch Time Prediction via Exponential-Gaussian Mixture Network_（[arXiv:2508.12665](https://arxiv.org/html/2508.12665)）在 KuaiRec 上的观看时长预测，再只改混合权重的生成方式，加入视频时长感知门控。

官方代码：<https://github.com/BestActionNow/EGMN>

论文报告的 KuaiRec 结果是 MAE 4.204、XAUC 0.6093。这两个数只作为文献对照。本组表格里的数字必须来自本组实际运行，不能预先填成“门控更好”。

讲稿见仓库外的 `ML.md`，方法取舍见 `修改意见.md` 的方案一。

## 全组必须遵守的四条约定

1. 只使用阶段 1 冻结的那一份 `dataset/kuairec/kuairec_data.pkl`。不要各自重新抽样或重新划分。
2. 指标只使用阶段 2 的 `eval/evaluate.py`。不要在自己的笔记本里手算 MAE。
3. 官方 EGMN 的分布形式、高斯分量数 `K=10`、左截断高斯，以及 NLL、熵项、L1 这三项损失，保持与官方 `model/egmn.py` 一致。门控只允许改 softmax 之前的混合权重 logits。
4. 预测文件统一放在 `outputs/`。测试预测的列是 `row_id,y_true,y_pred`。`row_id` 是该样本在 pkl 测试集中的行号，所有模型必须对齐。门控分析文件的列是 `row_id,duration_sec,g_0,...,g_10,omega_0,...,omega_10`。

## 阶段总览

每个阶段先写「实现目标」（这一阶段交出什么），再写「实现目的」（为什么必须有这一阶段），最后写「做到什么程度」（怎样才算完成）。

```text
阶段 1  数据冻结
阶段 2  评价脚本          可与阶段 1 并行
阶段 3  环境与 EGMN 复现  调试可提前，正式训练等阶段 1
阶段 4  VR 点回归         等阶段 1，评价用阶段 2
阶段 5  时长门控          等阶段 1 和阶段 3 的训练流程
阶段 6  消融与重复实验    等阶段 5 的三种变体
阶段 7  图表与课程报告    等阶段 6 的结果表
```

---

## 阶段 1：冻结 KuaiRec 数据

**实现目标**

产出全组唯一的训练集和测试集，以及门控要用的时长统计。

交付文件：

- `dataset/kuairec/row_data/` 中的原始表
- `dataset/kuairec/kuairec_data.pkl`，内含 `train`、`test`、`description`
- `dataset/kuairec/duration_stats.json`，字段为 `m_d`、`s_d`、`play_duration_max_ms`、`sample_frac`、`filter_seed`、`split_seed`
- `dataset/kuairec/data_audit.md`

**实现目的**

论文 KuaiRec 表使用的是大矩阵，不是讲稿里用来举例的 `small_matrix`。大矩阵约 1253 万条曝光、7176 名用户、10728 个视频。官方预处理会抽样并随机切分。若每人各切一次，MAE 无法比较，也不能和论文的 4.204 放在一起讨论。

讲稿中的一条记录（`user_id=14`，`video_id=148`，观看约 4.381 秒，视频约 6.067 秒）只说明字段含义。正式实验不用小矩阵代替大矩阵。

**具体处理**

按官方 `dataset/kuairec/kuairec_process.py` 执行：

1. 读 `big_matrix.csv`，左连接 `user_features.csv`、`item_daily_features.csv` 中的 `video_type`、`music_id`、`video_tag_id`，以及 `item_categories.csv` 和文案类目文件中的三级类目 id。
2. 去掉广告，去掉 `play_duration <= 0` 或 `video_duration <= 0` 的行。
3. 用 `random_state=312` 抽取 10%。
4. 标签 `play_time = play_duration / max(play_duration)`。视频时长按同一个最大值缩放并截断。丢掉观看时长超过视频时长 10 倍的样本。
5. 用 `random_state=1234` 打乱后，前 80% 为训练集，后 20% 为测试集。
6. 另存原始视频时长的秒数 `d_sec = video_duration / 1000`。对数变换只用于阶段 5 的门控输入，不改变 `play_time` 标签。
7. 在训练集上计算 `log(1+d_sec)` 的均值 `m_d` 和标准差 `s_d`。

官方脚本用逐行循环建词表，大矩阵上会很慢。可以改成向量化，但过滤条件、两个随机种子、特征字段和 `description` 的含义要与官方一致。原始数据目录名使用官方脚本中的 `row_data`。

下载地址见 <https://github.com/chongminggao/KuaiRec>，压缩包约 432MB。需要的文件是 `big_matrix.csv`、`user_features.csv`、`item_daily_features.csv`、`item_categories.csv`、`kuairec_caption_category.csv`。

**做到什么程度**

- 后续阶段能用同一路径读出 `train`、`test`、`description`。
- `data_audit.md` 写明：过滤后行数、10% 抽样后行数、训练条数、测试条数、`play_duration` 最大值、视频时长分位数、缺失值如何填充。
- 内存不够时，可以另交 `kuairec_data_frac001.pkl` 供冒烟，并在审计中写明 `sample_frac=0.01`。调试文件不能覆盖正式的 `kuairec_data.pkl`。
- 与论文数字对照的最终表格，必须来自 10% 正式划分。若最终没用上 10%，报告里不能把该结果和 4.204 写成同一次实验。

本阶段不训练模型。

---

## 阶段 2：统一评价指标

**实现目标**

所有模型的 MAE、XAUC、KL 和时长分组误差由同一个脚本计算。

交付文件：

- `eval/evaluate.py`
- `dataset/kuairec/duration_tertiles.json`，由训练集视频时长的三分位数得到
- 一组用假数据就能运行的检查，证明脚本计算正确

**实现目的**

官方训练时的标签是缩放过的 `play_time`，测试打印前把 MAE 乘以 `999639/1000` 换回秒。若有人在缩放空间比较，有人在秒空间比较，表格会差三个数量级。时长分组若用测试集自己的分位数，分组边界会跟着测试样本变化，模型之间也不再可比。

**脚本必须输出**

对每个预测文件输出：

- 缩放空间 MAE
- 官方常数换算的秒制 MAE：缩放 MAE 乘 `999639/1000`。和论文对照时用这一列
- 本份数据换算的秒制 MAE：缩放 MAE 乘 `play_duration_max_ms/1000`。`play_duration_max_ms` 来自阶段 1 的 `duration_stats.json`
- XAUC
- KL
- 短、中、长三档各自的秒制 MAE（官方常数那一列）

三分位边界只在训练集的视频时长上计算，再应用到测试样本。边界写入 `duration_tertiles.json` 后不再改。

**做到什么程度**

- 预测值与标签完全一致时，MAE 为 0。
- 预测排序与真实排序完全一致时，XAUC 高于随机打乱的预测。
- 阶段 3、4、5、6 把 csv 路径传给这个脚本即可，不在各自代码里复制 MAE 公式。
- 本阶段不负责把某个模型调到更高分，也不改模型结构。

调用方式在脚本顶部用注释写清，例如：

```bash
python eval/evaluate.py --pred outputs/egmn_42_pred.csv --stats dataset/kuairec/duration_stats.json --tertiles dataset/kuairec/duration_tertiles.json
```

---

## 阶段 3：复现官方 EGMN

**实现目标**

在冻结数据上得到官方 EGMN 的检查点、测试预测和训练日志。这一列是后面所有改动的基线。

交付文件：

- `requirements.txt`，环境与 PyTorch 2.1、Python 3.10 兼容
- `notes/egmn_official.md`，一页纸写明实际使用的超参
- `checkpoints/egmn_42.pt`
- `outputs/egmn_42_pred.csv`
- 每个 epoch 的测试日志

**实现目的**

门控是否有用，只能相对同一个官方模型来判断。若复现时改了损失或分量数，后面的提升无法解释。

**保持不变的实现**

以官方 `model/egmn.py` 和 `run_egmn.py` 为准：

- 共享 MLP 为 `(256, 128, 64)`，embedding 维度 16，dropout 0.2
- 高斯分量数 `K=10`，另有 1 个指数分量，混合权重共 11 维
- `lambda = softplus(h) + 1e-6`
- 高斯均值 `mu = softplus(h) + 1/lambda`，不低于指数分量的均值
- 高斯标准差 `softplus(h) + 1e-6`；计算似然时，高斯在 0 以下的部分按官方代码做左截断
- 预测值是混合分布的期望
- 损失为 `NLL + 0.1 * 熵项 + 1.0 * L1`
- 优化器 Adagrad，学习率 0.1，batch 2048，训练 10 个 epoch，种子 42

官方前向已经把 `duration` 作为连续特征送入共享 MLP。复现时保留这一点，不要为了“做创新”把时长从基线里删掉。

**做到什么程度**

- 先在阶段 1 的调试 pkl 上训练 1 个 epoch。nll、reg、entropy 都是有限数值，`predict` 能写出 csv 并通过阶段 2 的脚本。
- 正式 pkl 到达后，用上面的默认超参完整训练。
- 同一检查点重新加载后，测试 MAE、XAUC、KL 与训练结束时一致。
- 不修改官方 `model/egmn.py` 的损失和分布参数。论文数字只抄到对照栏，不写成本次复现结果。

---

## 阶段 4：VR 点回归基线

**实现目标**

提供一个只输出一个观看时长数值的参照模型。

交付文件：

- `checkpoints/vr_42.pt`
- `outputs/vr_42_pred.csv`
- 经阶段 2 脚本得到的总指标和三档 MAE

**实现目的**

EGMN 的价值在于用指数分量和多个高斯分量描述观看时长分布，而不是只拟合一个点。VR 用同一批用户、视频和同一次划分做点回归，用来看分布模型相对于普通回归的差异。它不参与门控设计。

**实现方式**

优先使用仓库中已有的 Wide&Deep 回归入口。若该入口无法在当前 pkl 上直接训练，则使用与 EGMN 相同的 embedding 和 `share_mlp (256, 128, 64)`，后面接一个线性头，损失为 L1。优化器、学习率、batch、epoch、种子和数据与阶段 3 相同。

**做到什么程度**

- `outputs/vr_42_pred.csv` 的 `row_id` 与 `outputs/egmn_42_pred.csv` 一一对应，`y_true` 相同。
- 指标由 `eval/evaluate.py` 生成。
- VR 中不加入时长门控，也不实现混合分布。

---

## 阶段 5：视频时长感知门控

**实现目标**

在不改混合分布和三项损失的前提下，让视频时长通过独立分支修正 11 个分量的权重，并训练完整门控模型。

交付文件：

- `model/egmn_gate.py`，包含 `concat`、`gate`、`nogate` 三种变体
- `notes/model_notes.md`，说明公式和 `g=0` 检查结果
- `checkpoints/gate_42.pt`
- `outputs/gate_42_pred.csv`
- `outputs/gate_42_gate.csv`

**实现目的**

短视频的长度会改变观看时长的取值范围：很短的视频和两分钟的视频，快速划走与看到后面的比例可能不同。官方模型已经把时长送进共享 MLP，因此“再拼一次时长”本身不足以称为改进。本阶段要回答的是：时长信息若专门用来调整各分布分量的权重，是否比只放在共享网络里更有用。

对数变换只作用于输入的视频时长，不作用于观看时长标签，因此不改变 EGMN 对原始观看时长的分布假设。

**公式**

实现使用 logits，避免对已经 softmax 的权重取对数：

```text
a0   = W_omega h + b
z_d  = MLP_d( (log(1 + d_sec) - m_d) / s_d )
g    = sigmoid(W_g [h; z_d] + b_g)
delta = W_d [h; z_d] + b_d
a    = a0 + g * delta
```

`m_d` 和 `s_d` 只读阶段 1 的 `duration_stats.json`，不要在测试集上重新计算。`g` 和 `delta` 的长度都是 11。

这与修改意见中的 `Softmax(log ω + g⊙Δ)` 等价：`log ω` 比原始 logits 少一个与分量无关的常数，再做 softmax 时该常数会消掉。`notes/model_notes.md` 要写上这句，避免再实现一次 `log(softmax(a0))`。

三种变体：

- `gate`：使用上面的公式。
- `concat`：`a = W [h; z_d]`，没有门控向量。用来看额外的时长编码是否已经够用。
- `nogate`：仍计算 `delta`，但 `g` 固定为全 1。用来看收益来自门控的选择性，还是只来自多出来的参数。

**做到什么程度**

- `g` 全为 0 时，`gate` 的 logits 与原始 `a0` 逐元素一致。这个检查写进 `notes/model_notes.md`。
- `loss` 和 `predict` 仍对 logits 做 softmax，似然和 L1 的计算调用与官方 EGMN 相同的路径。
- 本阶段完成 `gate`、种子 42 的训练，并交出预测文件和 `g`、`omega` 文件。
- `concat` 与 `nogate` 的代码在本阶段一并提交，训练放在阶段 6。

---

## 阶段 6：消融与重复实验

**实现目标**

在同一份 pkl 上补齐对照实验，形成一张可直接写入报告的表。

交付文件：

- `outputs/concat_42_pred.csv`
- `outputs/nogate_42_pred.csv`
- `outputs/results.csv`
- 若 GPU 允许，再增加种子 43 和 44 的 VR、EGMN、gate 预测与对应结果行

**实现目的**

只比较“原始 EGMN”和“加上门控的 EGMN”，无法区分是门控有效，还是多了一个时长编码器、多了一些参数。`concat` 和 `nogate` 就是用来做这个区分的。多种子用来看种子 42 的高低是否稳定。

**results.csv 每行至少包含**

`model,variant,seed,epochs,lr,batch_size,pkl_name,sample_frac,mae_sec_official,xauc,kl,mae_short,mae_mid,mae_long,minutes`

这些指标全部来自阶段 2 的脚本输出，不手填。

**运行顺序**

阶段 3 的 EGMN 优先占用 GPU，其次是阶段 5 的 `gate`，然后才是 `concat`、`nogate` 和额外种子。不要为了赶表把 epoch 或学习率改成与基线不同的值。时间不够时，种子 43 和 44 留空，并在表中注明未运行。

**做到什么程度**

主表至少有 5 行种子 42 的结果：VR、EGMN、concat、gate、nogate。每行都有总 MAE、XAUC，以及短、中、长三档 MAE。本阶段不改 `model/egmn_gate.py` 里的公式。

---

## 阶段 7：图表与课程报告

**实现目标**

把阶段 1 到阶段 6 的设置和 `outputs/results.csv` 写成课程报告，并给出可核对的图。

**实现目的**

课程要呈现的是一个完整闭环：复现论文方法，提出一个只改权重生成器的模块，用消融和时长分组判断这个模块是否有额外作用。报告中的结论必须和表中的数字一致。

**第一周就要定下的报告结构**

1. 观看时长为什么要做成分布预测
2. EGMN 的指数分量、高斯分量和三项损失
3. 时长门控的公式，以及它和共享 MLP 中已有时长特征的区别
4. 数据划分、种子、抽样比例、优化器
5. 五模型结果
6. 短、中、长视频上的 MAE，以及 `gate` 的平均 `g` 和平均 `ω`

**图**

- 观看时长分布
- 训练集时长三分位的边界
- 五个模型的 MAE 与 XAUC
- 三个时长组的 MAE
- `gate` 在短、中、长视频上的平均门控值和平均混合权重，数据只来自 `outputs/gate_42_gate.csv`

**做到什么程度**

- 文献栏写 MAE 4.204、XAUC 0.6093，并标明是论文结果。
- 本组栏只填 `outputs/results.csv` 里有的数。每个数字能追溯到该表或 `data_audit.md`。
- 写明实际 `sample_frac`、种子 312 和 1234。
- 写明视频时长已经进入基线的共享 MLP，门控是额外的权重修正。
- 写明分布、`K=10` 和三项损失与官方实现一致。
- 若 `gate` 没有同时优于 EGMN 和 `concat`，正文写成：在本次划分上，没有观察到门控相对于原始模型和简单时长编码的稳定额外收益。
- 不另做一套数据划分，也不把未写入 `results.csv` 的实验放进正文。

---

## 阶段交付

| 阶段        | 主要交付                                         | 完成才算交卷                                    |
| ----------- | ------------------------------------------------ | ----------------------------------------------- |
| 1 数据冻结  | 正式 pkl、`duration_stats.json`、`data_audit.md` | 后续阶段能读同一份 train/test；审计写明抽样比例 |
| 2 评价指标  | `eval/evaluate.py`、三分位边界                   | 假数据上 MAE 为 0；后续阶段都用该脚本           |
| 3 EGMN 复现 | `egmn_42` 检查点、预测、超参说明                 | 重载检查点后测试指标不变；官方损失未改          |
| 4 VR 基线   | `vr_42` 检查点与预测                             | 与 EGMN 的 `row_id` 和 `y_true` 对齐            |
| 5 时长门控  | `egmn_gate.py`、`gate_42` 预测与 `g`/`omega`     | `g=0` 时 logits 等于 `a0`；损失路径与官方相同   |
| 6 消融      | `concat_42`、`nogate_42`、`results.csv`          | 种子 42 的五行齐全，含三档 MAE                  |
| 7 报告      | 课程报告和上表中的图                             | 每个数字都能在结果表或审计文件中找到            |

## 目录

```text
DataMining_team3/
  dataset/kuairec/row_data/          原始表，不提交到 git
  dataset/kuairec/kuairec_data.pkl   正式划分
  dataset/kuairec/duration_stats.json
  dataset/kuairec/duration_tertiles.json
  dataset/kuairec/data_audit.md
  model/egmn.py                      官方文件，复现时不改损失
  model/egmn_gate.py                 阶段 5 新增的门控变体
  eval/evaluate.py                   阶段 2 的统一评价脚本
  checkpoints/
  outputs/
  notes/egmn_official.md
  notes/model_notes.md
  requirements.txt
```

原始大矩阵和 pkl 体积大，不要提交到 git。提交代码、审计、统计 json、`results.csv` 和报告图。
