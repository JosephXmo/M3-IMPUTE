# 全量 UCI-HAR：M3 → Juicer 联合训练部署指南

入口为仓库根目录的 `train_joint.py`，使用主项目 uv 环境。
Juicer package 只在 XHAR-SDCN 的 `delivery/` 中维护；手动复制到 M3 的 `delivery/`
供安装使用，该副本不纳入 M3 的 Git 历史。模型与梯度数据流见 [DATA_FLOW.md](DATA_FLOW.md)。

## 1. 当前实验的定义

- 将官方 train 的 7,352 行与 test 的 2,947 行按此顺序拼接，得到 **10,299 × 561**。
  所有行参与训练；不使用旧 `train_edge=0.7`，没有永久留出的样本或单元。
- 每轮独立随机挖空 `mask_rate` 比例的特征单元，默认 10%。mask 会保留每行、每列
  至少一个可见单元。样本行序始终不变，掩码不依赖活动标签。
- 按全部输入行拟合逐特征 Min-Max，作为本次全量传导式实验的固定预处理。
  隐藏单元在 M3 的 dense input 中变为 `1e-4`，同时从双向图中删除。
- M3 编码当前可见图，ImputeNet **预测全部单元**，包括可见位置。送给 XHAR 的是
  完整 decoder 重构矩阵，不将可见原值覆盖回去。由此，即使改变挖空率，模型的
  重构分支仍是 XHAR 的真实上游。
- M3 的重构 MSE 只计算当轮人为隐藏的单元。它们的原值仅充当自监督目标。
- XHAR 输入先由 M3 的 `[0,1]` 尺度逆变换至原 UCI 特征尺度，再进入 Juicer。
- 活动标签 `y` **只用于聚类指标**。subject ID 只定义时间片段边界；不当成类别输入。
- 评估仍使用全部 10,299 行，但使用一个固定随机挖空 mask，便于逐轮比较。
  这不是独立测试集泛化评估，也不是被永久隐藏单元上的泛化评估。

## 2. 代码与手动复制

M3 仓库管理 `train_joint.py`、`joint/` 运行代码、`configs/remote_start.json`、本指南和
数据流说明。主项目 `pyproject.toml` 统一声明依赖，包括 `munkres`。
以下内容只保留在本机，需要部署时单独传输：

- 手动复制的 `delivery/`；更新应在 XHAR 完成，再同步副本。
- 官方 `uci/raw_data/har/original/UCI HAR Dataset/` 数据。
- 需要恢复的 `runs/joint/` checkpoint 和对应运行记录。

在两个仓库位于同一父目录时，从 M3 根目录手动复制或更新 package：

```bash
rsync -a --delete \
  --exclude='__pycache__/' --exclude='*.egg-info/' \
  --exclude='build/' --exclude='dist/' \
  ../XHAR-SDCN/delivery/ delivery/
```

该命令将 delivery 作为可替换的分发副本；源码修改应先回到 XHAR。
不维护单独的复制来源清单；每次训练的 `run.json` 会记录实际导入的 package 源码哈希。

远端先同步 M3 代码，再单独传输 delivery 和官方数据。若使用传输包，可在本机根目录执行：

```bash
tar -czf /tmp/m3-juicer-joint.tar.gz \
  --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='test_*.py' --exclude='tests' \
  --exclude='local_*.json' --exclude='LOCAL_*' \
  pyproject.toml README.md train_joint.py joint delivery models utils \
  'uci/raw_data/har/original/UCI HAR Dataset'
```

本地测试、机器配置与验证记录不属于部署所需文件。Git ignore 不会自动过滤手工
打包内容，因此上面的命令显式排除了这些文件。

## 3. 使用主项目环境

建议 Python 3.12，在目标机器的 M3 根目录执行：

```bash
uv sync
uv pip install --python .venv/bin/python --no-deps -e ./delivery
source .venv/bin/activate
python -c "import torch, xhar_juicer; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(xhar_juicer.__file__)"
```

不创建 joint 专用环境，也不维护重复依赖清单。`munkres` 已由主项目安装；Juicer 的
PyTorch/PyG 依赖复用现有环境。重新执行 `uv sync` 后应再次执行 delivery 安装命令；
运行时使用激活后的 `python`，或 `uv run --no-sync python`，保持手动安装的 package。

用 `nvidia-smi` 和上面的导入检查确认 GPU/驱动/CUDA wheel 匹配；需要调整 PyTorch
安装时参考 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)。
不要从 macOS 复制 `.venv`。当前 runner 支持 CPU/CUDA，不支持 MPS RNG 重放。
远端运行不依赖 M3 checkout 中的本地测试文件。

## 4. 启动实验

所有命令均在 M3 根目录运行。输出目录必须全新或为空，防止覆盖结果。
`remote_start.json` 是后续实验的起始配置，不代表已优化的参数，也尚未在 CUDA 上验收。
可先以一轮运行确认目标环境，再使用完整预算：

```bash
python -u train_joint.py \
  --config joint/configs/remote_start.json --epochs 1 \
  --data-dir '/data/UCI HAR Dataset' \
  --output runs/joint/remote_check_001

python -u train_joint.py \
  --config joint/configs/remote_start.json \
  --data-dir '/data/UCI HAR Dataset' \
  --output runs/joint/remote_001
```

此处始终使用全部 10,299 行；需要调整模型容量或内存占用时，复制起始配置并修改参数。

后台启动时将日志写在输出目录外，避免触发“目录非空”保护：

```bash
mkdir -p runs/joint
nohup python -u train_joint.py \
  --config joint/configs/remote_start.json \
  --data-dir '/data/UCI HAR Dataset' \
  --output runs/joint/remote_001 \
  > runs/joint/remote_001.log 2>&1 &
tail -f runs/joint/remote_001.log
```

## 5. 损失、图和阶段

每次联合更新的损失为：

```text
L = m3_weight * MSE(M3预测, 原始归一化值)[当轮隐藏单元]
  + xhar_weight * (
        MSE(Juicer重构, stop_gradient(XHAR输入))
      + 0.1  * KL(P || Q)
      + 0.01 * KL(P || Z)
      + temporal_weight * 时间邻居概率差
    )
```

Q 是 AE latent 到聚类中心的 Student-t 分配，Z 是 GNN 输出概率。
P 从本轮**全图 Q** 计算并停止梯度。未启用 InfoNCE；没有活动标签交叉熵。

阶段顺序固定且显式：

1. 按固定评估 mask 运行初始 M3 重构。
2. 用该重构结果构建初始 XHAR 图；在它上面预热 Juicer AE。
3. 仅用 AE embedding 做 KMeans，初始化聚类中心。
4. 每轮刷新训练挖空 mask，执行 M3 编码、全矩阵重构、Juicer forward 和联合更新。
5. 固定评估 mask 上评估全部行，记录 Q/Z/P 三种分布的指标并保存 checkpoint。

XHAR 图由 **M3 重构特征** 的正余弦 top-k 邻居、同片段时间边和自环组成，KNN
不读取隐藏原值或标签。时间边不跨 subject 切换点及官方 train/test 拼接点；
KNN 可连接不同 subject，因为本实验是全量无监督图。没有边 dropout。

`graph_refresh_interval=1` 每轮重建；`5` 表示第 1、6、11…轮重建；`0` 固定初始图。
图选择是离散操作，不求梯度。每次 forward 的边属性则从最新重构特征可微计算。
RoPE 使用全局行号差编码，表示行序距离，并非跨 subject 的真实时间差。

这里的图构造、全图 P、全矩阵 M3 重构和训练目标已经构成新的联合实验协议，
与旧 XHAR 的局部采样训练结果不构成仅改变单个因素的对照。

## 6. 回传与内存

代码采用经过数值对照的 decoder 重算流程：

1. M3 encoder 生成保持计算图的节点 embedding。
2. 分块运行 decoder，保存全矩阵预测和每块 CPU/CUDA RNG 状态，不保留全部 decoder 激活。
3. 在预测矩阵的梯度代理上计算 XHAR 与 M3 损失，得到 `dL/d重构矩阵`。
4. 逐块重放相同 dropout/peer 采样，按链式法则累计 decoder 参数梯度和 embedding 梯度。
5. 对原 encoder embedding 执行一次 backward，最后一个 Adam 更新所有模块。

中间使用 detach 是**显式梯度传递的一部分**，不是把 detached 预测直接当成
已连通的联合模型。训练日志同时记录 XHAR→重构矩阵梯度、四个模块的参数梯度以及
M3 encoder/decoder、Juicer 的实际参数更新量。

`decoder_batch_size` 只限制 decoder 激活。M3 全图 encoder、Juicer 全图与输入图仍需
同时容纳；增加 M3 层数、隐藏维度或边维度会显著增加显存。CUDA OOM 时先降低
`decoder_batch_size` 判断是否来自 decoder；若仍 OOM，需要减小模型或另行实现
encoder 的分块/重计算。当前没有自动退化为样本子集，也没有 DDP。

## 7. 恢复、复评与调参

```bash
# 在同一架构和协议下继续到总共 200 轮；新输出保留原有历史。
python -u train_joint.py \
  --resume runs/joint/remote_001/checkpoint.pt \
  --epochs 200 --device cuda:0 \
  --data-dir '/data/UCI HAR Dataset' \
  --output runs/joint/remote_001_continued

# 只复评 checkpoint，用相同的固定挖空 mask 和当前保存的 XHAR 图。
python -u train_joint.py \
  --resume runs/joint/remote_001/checkpoint.pt --evaluate \
  --device cuda:0 --data-dir '/data/UCI HAR Dataset' \
  --output runs/joint/remote_001_eval
```

checkpoint 含模型、边投影、Adam 状态、图、RNG、参数配置、数据/源码哈希和历史指标。
源码哈希覆盖 M3/joint 运行代码和实际导入的 Juicer package，不包含本地测试或文档。
手动更新 delivery 后若源码变化，需启动新实验，或使用对应旧版本继续原实验。
本次整理改变了源码及哈希范围；整理前 checkpoint 应配合当时的代码恢复，不能改写
旧哈希强行通过检查。
仅加载自己生成的可信 checkpoint。精确续跑允许改变总轮数、device 和 CPU 线程数；
修改学习率、损失权重、mask、结构或源码会被拒绝。需要改变协议时复制 JSON，使用
新的 output 启动新实验；当前 CLI 不提供“忽略不匹配参数的部分权重加载”。跨设备的
续跑可执行，但不承诺 CPU/CUDA 逐位相同。

优先调整的参数及含义：

| 配置 | 含义 |
|---|---|
| `seed` | 模型初始化、训练随机过程和每轮挖空的基准种子 |
| `mask_rate` / `eval_seed` | 当轮隐藏比例 / 固定评估 mask 与评估随机采样种子 |
| `m3_weight` / `xhar_weight` | M3 隐藏单元重构与 XHAR 目标的相对权重 |
| `fcu` / `sru` / `peers` | M3 上下文模块及 peer 数量；当前 SRU 使用随机采样 |
| `m3_hidden` / `m3_layers` / `m3_edge_dim` | M3 encoder 容量与全图内存主要因素 |
| `decoder_batch_size` | decoder 重算块大小；随机模块下改变块大小会改变随机序列 |
| `xhar_widths` / `latent` / `xhar_edge_dim` | Juicer AE、latent 和边投影容量 |
| `ae_epochs` / `kmeans_n_init` | Juicer 预热和初始聚类次数 |
| `topk` / `similarity_threshold` | XHAR KNN 稀疏程度 |
| `temporal_delta` / `temporal_weight` | 同片段时间邻居范围与平滑权重 |
| `graph_refresh_interval` | 离散 XHAR 图刷新周期 |
| `lr` / `epochs` | 联合 Adam 学习率 / 总训练轮数 |

做参数比较时保持固定评估 mask、数据文件和其余配置，并用多个 seed 报告结果。
不要仅以最优 epoch 代替固定训练协议；当前程序不根据活动标签指标挑选 checkpoint。

## 8. 结果文件

| 文件 | 内容 |
|---|---|
| `run.json` | 状态、数据和源码哈希、环境、运行时间、进程 RSS、CUDA 峰值（若适用） |
| `config.json` | 展开后的完整配置 |
| `metrics.csv` | 每轮 Q/Z/P 的 Accuracy、NMI、ARI、macro/micro F1、Precision、Recall |
| `training.json` | 损失、挖空数量、图大小、梯度范数、实际参数更新量 |
| `checkpoint.pt` | 最新一轮完整恢复状态 |
| `predictions.npz` | 全部行的 Q/Z/P 聚类 ID、评估标签、subject、行号 |
| `evaluation_known_mask.npy` | 固定评估可见性 mask，True 为可见 |
| `normalization.npz` | 全量输入拟合的特征 offset / scale |

主结果看 `Distribution=Z`。Accuracy/F1/Precision/Recall 用与原 XHAR 相同的
Munkres 最优标签匹配；NMI 使用 arithmetic 平均，ARI 使用原聚类 ID。
CSV 中为 0–1 比例，ARI 可为负数；`M3 hidden RMSE` 是额外诊断。

若 Accuracy 接近最大类别占比且 NMI/ARI 接近 0，应先检查是否只预测一个簇。
应先确认数据、梯度、参数更新、评估与恢复流程，再开展模型规模和超参数实验。
