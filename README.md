# M³-Impute：缺失值补全训练指南

本仓库是 M³-Impute 的实现。模型把一个表格转换为“样本节点—特征节点”的二部图，用图神经网络学习节点表示，再由 `ImputeNet` 预测缺失的样本-特征边。原有训练入口仍然是 `train_mdi.py`，核心实现位于：

```text
train_mdi.py                 命令行入口、随机种子、设备和输出目录
uci/uci_data.py              读取 data.txt，归一化并构造二部图
training/gnn_mdi.py          训练、验证、测试和结果保存
training/imputation_batches.py  分块预测、损失累积与编码器梯度回传
models/gnn_model.py          GNNStack 图编码器
models/egsage.py             边感知 GraphSAGE 层
models/egcn.py               边感知 GCN 层
models/prediction_model.py   FCU、SRU 和 ImputeNet 预测头
```

保留原有目录、类名和训练调用方式，并添加说明性注释。训练和评估的预测头现在按目标单元格分块，以限制 FCU/SRU 中间张量的显存；图编码仍采用全图方式。为了兼容仓库中已有的实验脚本，`--impute_hidden` 是 `--impute_hiddens` 的别名；两个参数写入同一个配置项。

![M3-Impute 模型结构](assets/m3-model.png)

## 1. 安装环境

主项目统一使用 `pyproject.toml` 和 uv 环境管理依赖；其中已经包含联合聚类指标所需的
`munkres`，不再维护 joint 专用依赖列表。

```bash
uv sync
source .venv/bin/activate
```

Juicer package 在 XHAR-SDCN 的 `delivery/` 中维护，使用联合训练时手动复制到本仓库
`delivery/` 并安装进同一环境。该副本被 Git 忽略，安装和远端部署步骤见
[joint/DEPLOYMENT.md](joint/DEPLOYMENT.md)。后续若重新执行 `uv sync`，需再次安装
这个不在主项目依赖清单中的本地 package。

核心训练不再直接依赖 `torch-scatter`：EGCN 的度数求和使用 PyG 的 `scatter(..., reduce='sum')`，EGSAGE 不需要导入该扩展。PyG 的这些求和路径可使用 PyTorch 原生算子；因此可以先只安装与 GPU 架构兼容的 PyTorch 和 PyG，避免被旧扩展 wheel 限制。先确认版本：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda :', torch.version.cuda)
print('available:', torch.cuda.is_available())
PY
```

如果其他工具需要 `torch-scatter`，其 wheel 必须同时匹配 PyTorch、CUDA 和 GPU 架构。例如，旧环境 PyTorch 2.1.0 + CUDA 12.1 对应如下页面（这不是 RTX PRO 6000 Blackwell 的安装推荐）：

```bash
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

CPU 扩展使用对应的 `+cpu` 页面。核心依赖安装后可以快速检查，不需要导入可选扩展：

```bash
python - <<'PY'
import torch
import torch_geometric
print(torch.__version__, torch_geometric.__version__)
from models.gnn_model import GNNStack
print('M3 core imports: ok')
PY
```

## 2. 数据格式

默认数据根目录是 `uci/raw_data`。每个数据集至少需要一个文件：

```text
uci/raw_data/<数据集名>/data/data.txt
```

`data.txt` 是无表头的纯数字文本，默认由 `numpy.loadtxt` 读取；每行一个样本，每列一个数值。最后一列被当作下游任务目标 `y`，前面的所有列才是要进行缺失值补全的特征 `X`：

```text
x_1  x_2  ...  x_m  y
```

因此，如果原始数据有 `m` 个输入特征和 1 个目标列，`data.txt` 必须有 `m + 1` 列。当前加载器要求 `X` 在开始训练前是完整的有限数值；不要把空字符串、表头或 `NaN` 直接写入文件。分类目标也可以保留为数字，但补全图仍只使用最后一列之前的 `X`。

加载时，`X` 默认按列进行 Min-Max 归一化到 `[0, 1]`。归一化后的值会作为样本-特征边属性和训练标签；`y` 单独保存在 `data.y`，不参与特征补全。

完整的数据目录示例：

```text
uci/raw_data/my_dataset/
└── data/
    └── data.txt
```

仓库中已有的一些数据集还包含 `index_target.txt`、`index_features.txt` 或 `split_data_train_test.py`。当前 `load_data()` 的训练主路径只读取 `data/data.txt`，这些辅助文件不是新数据集接入的必需项。

### mHealth 数据集

`mhealth/original` 包含 10 位受试者的 50 Hz 原始传感器记录。从仓库根目录分别生成两种特征表：

```bash
python mhealth/preprocess.py --mode full
python mhealth/preprocess.py --mode acc_gyro_only
python train_mdi.py --epochs 1 --log_dir mhealth_smoke mhealth --data mhealth --train_edge 0.7
```

`full` 写入 `uci/raw_data/mhealth/data`，`acc_gyro_only` 写入 `uci/raw_data/mhealth_acc_gyro_only/data`。两种模式共享 253 维加速度/陀螺仪特征：身体与重力加速度分离、连续信号求 jerk、向量模长、均值/标准差/中位绝对偏差/四分位距/能量/自相关、FFT 主频与频谱量、重力方向角，以及轴间和部位间相关性。全传感器模式再加入磁场单位方向、对数场强及其变化特征，以及 ECG 基本统计，共 295 维；磁场强度取对数以降低受试者间的尺度差异。短窗 ECG 不计算心率或 HRV。`acc_gyro_only` 仅对齐 UCI-HAR 的传感器类型和部分特征方法；mHealth 佩戴部位不同。具体列名及顺序保存在 `feature_names.txt`；这些列与 UCI-HAR 官方 561 列不等价，不能直接复用依赖其定义的权重。

窗口在每位受试者的原始时间轴上固定取 128 点（2.56 秒），步长 64 点（50% 重叠），**不使用标签确定窗口边界**。滤波在完整受试者序列上进行：加速度/陀螺仪先用 20 Hz 零相位 FIR 去噪，加速度再经 0.3 Hz 零相位 FIR 分离重力；输入单位转换为 `g` 与 `rad/s`。这是离线预处理，零相位滤波使用窗口后续的信号，不能直接作为实时推理流程。

`data.txt` **只含特征**，对应的活动标签按相同行序保存在 `labels.txt`；仅保留标签 1-12 且窗口内标签完全一致的样本，标签 0 是未标注区间。`windows.csv` 记录**全部候选窗口**的起止采样点、多数标签、纯度和筛选决定，便于报告筛选覆盖率或改变 `--min-purity` 后重新生成。`manifest.json` 保存双文件格式、参数和原始文件 SHA-256；`subject_ids.txt` 与 10 组 `index_train_<fold>.txt` / `index_test_<fold>.txt` 构成留一受试者划分，fold 0-9 分别留出受试者 1-10。活动标签和受试者编号都不进入特征矩阵。

`labels.txt` 仅供 XHAR-SDCN 等下游任务计算指标使用。`train_mdi.py mhealth` 只读取纯特征的 `data.txt`，不加载活动标签；M3 的 `train_labels` 是被遮挡单元格的特征值。原有 `train_mdi.py uci` 路径保持不变。上述命令是单元格补全训练的冒烟示例；跨受试者实验还需在训练折拟合缩放器、训练模型，并只对留出的受试者评估。特征单元格随机缺失也不同于原始时间点或整路传感器缺失，两种缺失协议应分别报告。

### UCI-HAR 数据集

当前保留两条独立流程，都直接读取官方 561 维特征表：

- **离线 cross comparison**：随机排列全量 UCI-HAR，以 7209 行训练 M3，其余
  3090 行构成 S₀；比较 S₀、均值占位 S_drop 和 M3 补全 S_m3 三种 XHAR 输入。
  使用 S_train 的缩放和均值统计，行集合互斥但 subject 可以重叠。入口及命令见
  [CROSS_COMPARISON_GUIDE.md](uci/raw_data/har_cross_comparison/CROSS_COMPARISON_GUIDE.md)。
- **全量联合训练**：全部 10,299 行进入 M3，每轮自动挖空；M3 重构特征直接进入
  Juicer，XHAR 损失反传到 M3，活动标签仅用于聚类指标。入口为 `train_joint.py`，
  详见[部署指南](joint/DEPLOYMENT.md)和[数据流图](joint/DATA_FLOW.md)。

`uci/raw_data/har_cross_comparison/` 的维护代码为 `preprocess_cross_comparison.py` 和其专用补全模块
`cross_comparison_inference.py`，配套说明为 `CROSS_COMPARISON_GUIDE.md`。
旧全量预处理、representation 导出和四路比较入口已退役。
`uci/raw_data/har/data/split_data_train_test.py` 按原仓库约定保留。
已有原始数据、生成数据和历史结果继续留在本地，不由此整理重新生成。

XHAR 的离线实验入口现在位于 `experiments/input_impute_cross_comparison/`，新结果
放在该目录的 `results/` 中。它与全量联合训练的划分和指标协议分别管理。

本地测试代码集中保存在被 Git 忽略的 `local_tests/`。`joint/configs/local_*.json`
及 `joint/LOCAL_*` 记录也仅留在本机；仓库保留运行代码、远端起始配置和通用部署说明。

## 3. 从表格到图的含义

对于 `N` 条样本和 `M` 个特征，节点顺序固定为：

```text
[样本节点 0 ... 样本节点 N-1, 特征节点 0 ... 特征节点 M-1]
```

每一个单元格 `X[i, j]` 生成两条边：样本 `i → 特征 j` 和特征 `j → 样本 i`。边属性就是归一化后的单元格值。`uci/uci_data.py` 先构造完整二部图，再通过 mask 将单元格划分为可见训练边和待补全测试边；这样不会改变节点编号。

训练过程中还会在已知训练边中随机隐藏一部分，形成当前 epoch 的输入图。模型只能看到剩余可见值和 `Known_Mask`，并用被隐藏边的原值计算自监督重构损失。`ImputeNet` 的两个可选上下文单元是：

- `--apply_attr`：FCU，利用同一行其它特征的关系；
- `--apply_peer`：SRU，采样其它样本在同一特征上的关系；
- 两者都不开启时，使用节点表示的直接预测头。

## 4. 最小训练命令

在仓库根目录执行。下面命令使用 `yacht` 数据集、70% 初始可见单元、少量 epoch 做 smoke test：

```bash
python train_mdi.py \
  --epochs 10 \
  --gpu 0 \
  --node_dim 64 \
  --edge_dim 64 \
  --impute_hiddens 64 \
  --known 0.7 \
  --log_dir smoke_test \
  uci --data yacht --train_edge 0.7
```

设备编号由 `torch.backends.mps.is_available()` 决定：MPS 可用时，`--gpu 0` 使用 MPS，`--gpu 1` 使用 `cuda:0`，后续依次递增；MPS 不可用时，`--gpu N` 直接使用 `cuda:N`。所选 CUDA 后端不可用时提示并回退到 CPU，可见 CUDA 设备编号越界则报错。训练入口默认编号在 MPS 可用时为 `0`，否则保留 `1`；多 GPU 机器请显式指定。CUDA 编号基于外部环境已配置的可见设备，程序不再重写 `CUDA_VISIBLE_DEVICES`。

用于正式回归实验的一个常见配置如下：

```bash
python train_mdi.py \
  --epochs 20000 \
  --repeat_exp_num 5 \
  --apply_attr --apply_peer \
  --sample_peer_size 5 \
  --sample_strategy cos-similarity \
  --node_dim 128 --edge_dim 128 \
  --impute_hiddens 128 \
  --known 0.5 \
  --save_model --save_prediction \
  --log_dir m3_default \
  uci --data yacht --train_edge 0.7
```

已有实验脚本可以直接运行：

```bash
bash run_exp1_impute.sh        # MCAR
bash run_exp1_impute_mar.sh    # MAR
bash run_exp1_impute_mnar.sh   # MNAR
bash run_exp2_robust.sh
bash run_exp3_ablation.sh
```

脚本中的参数数量较多，建议先用 `--epochs 10` 验证数据路径、设备和张量形状，再恢复论文实验的 epoch 数和重复次数。

训练启动时默认输出 `[Startup]` 阶段提示，无需 `--display_log`。每步开始前立即刷新输出，结束后显示秒数：依赖导入、设备和随机种子、TXT 读取、MinMax 归一化、双向图构建、缺失掩码和边划分、模型初始化、张量搬运、稠密值/可见性矩阵、优化器。CUDA/MPS 初始化步骤在结束时同步，以便耗时包含实际 GPU 工作；同步不加到 epoch 循环里。

UCI 的边列表和边值构建按行显示 tqdm（完成行数、速度、预计剩余时间）；启用冷启动划分或 MAR/MNAR 时，相应的边筛选或缺失机制拟合也显示进度。TXT 读取及单次张量操作没有可靠的内部百分比，因此显示开始/结束与耗时。看到 `[Startup] Training ready` 后进入 `[Train] Epochs` 进度条。HAR 对照包装器以无缓冲模式启动训练子进程，重定向日志时也能及时看到阶段输出。

## 5. 重要参数

### 数据划分与缺失机制

`uci --train_edge 0.7` 指定初始图中约 70% 的单元格作为训练可见边，剩余边作为测试补全目标。它不是训练时每一轮的遮挡率。

`--known 0.7` 指定每个训练 epoch 中保留多少训练单元可见；例如 `0.7` 表示每轮再隐藏约 30% 的训练单元。`--loss_mode 0` 对所有训练标签计算损失，`--loss_mode 1` 只对本轮隐藏的单元计算损失。

`--valid 0.1` 会从初始训练边中再划出 10% 做验证，并按验证 RMSE/L1 保存最佳模型。正式比较模型时建议设置验证集，例如 `--valid 0.1`；只想快速测试时可保持默认 `0`。

`--corrupt` 控制初始 train/test 边的缺失机制：

```text
mcar   独立随机缺失，默认值
mar    根据其它特征的 logistic 机制生成缺失
mnar   MAR mask 再叠加额外的可见性 mask
```

MAR/MNAR 的控制参数是 `--mar_rate_obs`、`--mar_rate_missing` 和 `--mnar_known_mask`。这些机制要求输入 `X` 本身没有原始缺失值。

### 模型与资源

`--model_types EGSAGE_EGSAGE_EGSAGE` 用下划线分隔 GNN 层，可以替换为仓库支持的层类型组合。`--node_dim` 和 `--edge_dim` 控制节点/边隐空间宽度。它们不能控制 FCU/SRU 沿完整输入特征列展开的张量大小。

`--impute_batch_size` 控制预测头每批的目标单元格数，默认 `1024`，训练、验证及单元格测试均使用它。例如 `python train_mdi.py --impute_batch_size 512 --apply_attr --apply_peer --save_model uci --data har`。每轮只编码一次训练图，逐批计算损失并立即反向，最后将累积的节点表示梯度回传给 GNN，再执行一次优化器更新。损失按该轮全部训练目标数归一化，最后不足一批的部分也有正确权重；`loss_mode=1` 仍保留原有的全部训练目标数分母。减少 batch 不会删除输入列或训练样本，但通常增加运行时间。

`--apply_attr` 和 `--apply_peer` 分别打开 FCU 和 SRU。SRU 的 `--sample_peer_size` 越大，采样上下文越多但计算和显存开销越高。`--sample_strategy random_sample` 不需要相似度缓存；`cos-similarity` 会按 `--update_cos_sample_prob_every` 个 epoch 更新采样概率。

当样本数很大且采用 `cos-similarity` 时，`--very_large_dataset` 会让 SRU 使用较小的候选空间，大小由 `--sample_space_size` 控制，以避免维护完整的样本两两相似度表。它不限制 FCU/SRU 的每批目标数，对默认随机 peer 策略没有这一节省作用。

`--init_epsilon` 是输入矩阵中隐藏单元的占位值。它和 `Known_Mask` 配合使用，用于区分“真实观测值为 0”和“当前不可见”。通常保留默认的 `1e-4`。

## 6. 在新的原始数据集上训练

下面流程不需要修改模型代码。

### 第一步：整理原始表格

假设原始数据有 1,000 行、20 个特征和 1 个目标列。先完成以下工作：

1. 删除表头，把类别字符串编码为数值；
2. 确定最后一列是目标 `y`，前 20 列是 `X`；
3. 处理原始数据中的缺失值、无穷值和非数值字段；
4. 保证每一行列数一致，并保存为空白分隔的 `data.txt`。

例如可以用 pandas 做一次离线检查（这一步只负责准备文件，不会改仓库代码）：

```python
import numpy as np
import pandas as pd

df = pd.read_csv('raw.csv')
df = df.apply(pd.to_numeric, errors='raise')
assert df.shape[1] >= 2
assert np.isfinite(df.to_numpy()).all()
df.to_csv('uci/raw_data/my_dataset/data/data.txt', sep=' ', header=False, index=False)
```

如果数据已经是 NumPy 数组，也可以使用 `np.savetxt`；不要使用 CSV 逗号分隔，因为当前加载器直接调用 `np.loadtxt` 的默认空白分隔模式。

### 第二步：放入约定目录

```bash
mkdir -p uci/raw_data/my_dataset/data
# 将生成的 data.txt 放入上面的目录
```

`--data my_dataset` 会让加载器读取 `uci/raw_data/my_dataset/data/data.txt`。数据集名字只要是合法目录名即可，不需要把它写入 Python 的枚举表。

### 第三步：先做数据和图的 smoke test

先用少量 epoch、较小隐藏维度运行：

```bash
python train_mdi.py \
  --epochs 3 --node_dim 16 --edge_dim 16 \
  --known 0.8 --log_dir my_dataset_smoke \
  uci --data my_dataset --train_edge 0.8
```

检查控制台是否打印正确的 known/missing ratio，且没有 `ValueError`、`NaN` 或 CUDA out-of-memory。这个步骤主要验证文件路径、列数、归一化和二部图索引。

### 第四步：选择训练设置

第一次正式实验可以从下面的设置开始：

```bash
python train_mdi.py \
  --epochs 20000 \
  --valid 0.1 \
  --apply_attr --apply_peer \
  --sample_strategy random_sample \
  --sample_peer_size 5 \
  --node_dim 64 --edge_dim 64 \
  --impute_hiddens 64 \
  --known 0.7 \
  --save_model --save_prediction \
  --log_dir my_dataset_m3 \
  uci --data my_dataset --train_edge 0.7
```

建议至少使用 3 个不同 `--seed` 或 `--repeat_exp_num 3`，报告均值和标准差。`train_mdi.py` 会为每次重复运行重新设置 `args.seed`，并把实际命令追加写入输出目录的 `cmd_input.txt`。

### 第五步：按数据规模调参

- 小数据集：可以使用 `node_dim/edge_dim=64` 或 `128`，`sample_peer_size=5`；
- 样本数较多：先关闭 `cos-similarity`，使用 `random_sample`，降低 `sample_peer_size`；
- 大样本数据集：打开 `--very_large_dataset`，设置较小的 `--sample_space_size`，必要时降低 `node_dim`；
- 显存不足：先降低 `node_dim`、`edge_dim` 和 `sample_peer_size`，再考虑减少 epoch；
- 训练曲线抖动：降低 `--lr`，或启用 `--opt_scheduler step` / `--opt_scheduler cos`。

## 7. 输出文件与评估

输出目录格式为：

```text
./uci/test/<data>/<log_dir>/
├── cmd_input.txt
├── result.pkl
├── run_0_curves.png
├── model.pt                 # 使用 --save_model 时生成
├── impute_model.pt          # 使用 --save_model 时生成
└── outputs_best_valid.png   # 使用 --save_prediction 且启用 --valid 时生成
```

`result.pkl` 是一个字典，主要包含：

```text
args       本次运行参数
curves     train_loss、valid_rmse/valid_l1、test_rmse/test_l1、lr
outputs    final_pred_train、label_train、final_pred_test、label_test
```

预测值和标签默认在 Min-Max 归一化空间中。如果要在原始单位评估，需要保存训练时的 scaler 或在数据准备阶段记录每列的最小值和最大值，然后对预测值做反归一化。当前仓库的 `load_data()` 没有把 scaler 单独写入 checkpoint。

基线和下游任务入口仍然可用：

```bash
python baseline_mdi.py --method mean uci --train_edge 0.7 --data my_dataset
python downstream_task.py --method mean
python downstream_task.py --method m3-impute
```

## 8. 常见问题

**找不到数据文件**：确认文件名精确为 `data.txt`，且路径是 `uci/raw_data/<data>/data/data.txt`；命令中的 `--data` 只填写目录名。

**列数或 shape 错误**：检查最后一列是否存在、每行是否同宽，以及 `data.txt` 是否误包含表头或逗号。

**出现 NaN**：先检查原始数据是否有 NaN/Inf，再检查学习率和隐藏维度。缺失机制是训练时模拟的，输入文件本身应先保持完整。

**CUDA 设备错误**：显式指定有效的 `--gpu`；MPS 可用时 CUDA 编号需加 1，例如 `--gpu 1` 对应 `cuda:0`。所选 CUDA 后端不可用时使用 CPU。

**运行很慢或显存不足**：FCU/SRU 报 OOM 时先降低 `--impute_batch_size`，例如 `1024 → 512 → 256`；GNN 编码报 OOM 时降低 `node_dim`、`edge_dim`。`sample_peer_size` 也影响 SRU 的每批开销；大数据集采用 cosine peer 时可使用 `--very_large_dataset`。完整图的边数约为 `2 × N × M`，预测头分块不能消除全图编码显存。

**实验脚本参数报错**：仓库脚本中的旧参数名 `--impute_hidden` 已由入口兼容；如果自定义脚本使用其它未声明参数，请以 `train_mdi.py` 和 `uci/uci_subparser.py` 中的 parser 定义为准。

## 9. 当前实现边界

训练分块保留每轮一次优化器更新和原损失归一化方式，并修复了以下实现问题：

1. `GNNStack.feature_nodes` 注册为 `nn.Parameter`，现在参与优化、`state_dict()` 和设备迁移。旧版本的完整模块 checkpoint 加载时会把普通 Tensor 注册为参数，并保留数值；新训练会更新这些向量，与旧实现的训练结果可能不同。
2. 图上下文保留双向边，但训练、评估和补全只预测每个目标单元格一次，避免曾经没有进入损失的反向重复预测。最后一层无人使用的边更新不再执行，模块仍保留以兼容 checkpoint。
3. cosine peer 概率缓存不保留计算图；更新时使用已定义的观测矩阵，并处理负权重或正候选不足的情况。无验证集时不会访问不存在的最佳验证轮次。

固定 peer、关闭 dropout 时，新旧预测、损失及编码器/预测头梯度的数值一致性已有回归测试。随机 peer 和 dropout 的随机数调用顺序会随分块大小改变，因此同 seed 的不同 batch 配置不保证逐位相同。完整模型训练测试覆盖 12/10、561 列、随机 peer、无验证集，以及另一组带验证和 cosine 缓存更新的配置；这些是小规模 CPU 测试，不是全量 HAR 的 CUDA 验收。

`concat_states` 保留原接口，当前 forward 未拼接中间状态；评估指标默认仍在归一化特征空间。图编码仍是全图方式，超大数据集需要继续控制编码器宽度及图规模。

## License

MIT
