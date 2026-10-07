# M³-Impute：缺失值补全训练指南

本仓库是 M³-Impute 的实现。模型把一个表格转换为“样本节点—特征节点”的二部图，用图神经网络学习节点表示，再由 `ImputeNet` 预测缺失的样本-特征边。原有训练入口仍然是 `train_mdi.py`，核心实现位于：

```text
train_mdi.py                 命令行入口、随机种子、设备和输出目录
uci/uci_data.py              读取 data.txt，归一化并构造二部图
training/gnn_mdi.py          训练、验证、测试和结果保存
models/gnn_model.py          GNNStack 图编码器
models/egsage.py             边感知 GraphSAGE 层
models/egcn.py               边感知 GCN 层
models/prediction_model.py   FCU、SRU 和 ImputeNet 预测头
```

本次整理只增加说明性注释，并保留了原有目录、类名和训练调用方式。为了兼容仓库中已有的实验脚本，`--impute_hidden` 现在是 `--impute_hiddens` 的别名；两个参数写入同一个配置项。

![M3-Impute 模型结构](assets/m3-model.png)

## 1. 安装环境

代码依赖 PyTorch、PyTorch Geometric、`torch-scatter`、NumPy、SciPy、scikit-learn、pandas、matplotlib、seaborn 和 `fancyimpute`。仓库提供了基础依赖列表：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`torch-scatter` 必须与 PyTorch 和 CUDA 版本匹配。先确认版本：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda :', torch.version.cuda)
print('available:', torch.cuda.is_available())
PY
```

例如，PyTorch 2.1.0 + CUDA 12.1 可以使用：

```bash
python -m pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

CPU 环境则使用对应的 `+cpu` 页面。不要只根据操作系统选择链接；`torch-scatter` 的 wheel 需要同时匹配 PyTorch 主版本和 CUDA 版本。安装后可以快速检查：

```bash
python - <<'PY'
import torch
import torch_geometric
from torch_scatter import scatter
print(torch.__version__, torch_geometric.__version__)
print('torch-scatter: ok')
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

### UCI-HAR 数据集

仓库中的 `uci/raw_data/har/original` 是 UCI-HAR 原始目录。可以使用预处理脚本合并官方 train/test 文件，并按指定比例生成缺失单元：

```bash
cd uci/raw_data/har
python preprocess.py --rate 0.1 --seed 0
python data/split_data_train_test.py
```

脚本会生成 561 个特征和 1 个活动标签列。完整的 `data/data.txt` 供当前 M3-Impute 加载器使用；模拟挖空结果保存在 `data/data_with_missing.txt`，行列位置保存在 `data/missing_indices.txt` 和 `data/missing_mask.txt`。由于当前 `uci_data.load_data()` 要求输入矩阵为有限值，训练命令应继续指向完整的 `data.txt`，缺失率由 `--train_edge`/`--known` 或后续读取 `missing_indices.txt` 的实验逻辑控制。

默认的 `split_data_train_test.py` 保留 UCI-HAR 原有的 7352/2947 行 subject 划分。如果需要和其它数据集一样生成 20 个随机 90/10 划分，可以运行：

```bash
python data/split_data_train_test.py --random-splits
```

### 用训练好的 M3 导出 HAR representation

训练并保存 HAR 的 M3 checkpoint 后，可以用独立脚本对缺失数据进行补全，再用同一个 `GNNStack` 编码完整数据和补全数据：

```bash
python uci/raw_data/har/generate_m3_representation.py \
  --model-path uci/test/har/<实验目录>/model.pt \
  --impute-model-path uci/test/har/<实验目录>/impute_model.pt \
  --output-dir uci/raw_data/har/representation \
  --chunk-size 512 \
  --device cuda:0
```

脚本会生成：

```text
har_complete_representation.txt
har_missing_imputed_representation.txt
har_missing_imputed_features.txt
har_missing_imputed_data.txt
har_labels.txt
har_manifest.json
```

其中两个 `representation.txt` 都是样本节点表示，行顺序与 HAR 原始数据一致；通常列数等于 M3 的 `node_dim`，但启用 `concat_states` 时以实际导出的 `representation_dim` 为准。完整数据和缺失数据使用同一个 scaler、同一个 checkpoint 和同一个编码流程；缺失输入会先由 `ImputeNet` 补全，再进行全可见编码。`--chunk-size` 只分块处理 ImputeNet 目标边，GNN 编码本身仍然是整图计算。

如果 XHAR-SDCN 使用 M3 representation 作为 `X_path`，其 `n_input` 必须设置为 `representation_dim`，并且需要重新训练一个输入维度相同的 XHAR-SDCN autoencoder checkpoint。例如 M3 `node_dim=64` 时：

```yaml
name: "har"
X_path: "data/har_missing_imputed_representation.txt"
y_path: "data/har_labels.txt"
graph_path: "graph/hhar5_graph.txt"
pretrain_path: "data/har_m3_pretrain_64.pkl"
n_input: 64
n_clusters: 6
```

现有 `hhar.pkl` 是 561 维输入的预训练模型，不能直接加载到 64 维 M3 representation 上。如果希望继续复用现有 561 维 XHAR-SDCN 配置，应使用脚本输出的 `har_missing_imputed_features.txt`，并将 `n_input` 保持为 561。

### UCI-HAR 的 XHAR-SDCN 四路比较

`UCI HAR Dataset/features_info.txt` 说明了 561 列的来源：原始 50 Hz 三轴传感器信号经过滤波、body/gravity 分解、jerk/magnitude 构造、FFT 和统计量计算后，每个窗口得到一个 561 维特征向量。`X_train` 和 `X_test` 拼接后正好是 10299×561，与 XHAR-SDCN 的 `data/hhar.txt` 对应。因此主实验应直接比较 561 维特征，避免把“缺失补全”和“representation 生成”两个因素混在一起。

当前 cell-level `rate=0.1` 会让几乎每一行都含缺失，`S_drop` 将没有可用样本。四路比较采用可复现的行级掩码时，可运行：

```bash
python uci/raw_data/har/prepare_xhar_comparison.py \
  --mask-mode row \
  --row-fraction 0.1 \
  --cells-per-row 102 \
  --rate 0.1 \
  --seed 0 \
  --output-tag har_missing_10rows_102cols \
  --batch-size 512 \
  --epochs 200
```

这里的实际全局缺失率是 `0.1×102/561≈1.82%`；`rate` 表示目标全局比例，但显式指定的 `cells-per-row` 优先。脚本会生成固定掩码的 `S′₀`、线性插值的 `S_lerp`、删除含缺失行并重映射图后的 `S_drop`，以及三份使用相同 XHAR-SDCN 超参数的 YAML。`S0`、`S_lerp` 与 `S_drop` 的文件位于 XHAR-SDCN 的 `data/har_missing_10rows_102cols/`，配置和运行脚本位于 `experiments/har_missing_10rows_102cols/`。

得到 M3 的 `har_missing_imputed_features.txt` 后，用同一掩码补入第四路：

```bash
python uci/raw_data/har/prepare_xhar_comparison.py \
  --mask-file /Users/joseph_phantom/CodingCenter/XHAR-SDCN/data/har_missing_10rows_102cols/hhar_missing_mask.txt \
  --m3-features uci/raw_data/har/representation/har_missing_imputed_features.txt \
  --output-tag har_missing_10rows_102cols \
  --batch-size 512 \
  --epochs 200
```

然后在 XHAR-SDCN 根目录运行 `experiments/har_missing_10rows_102cols/run_all.sh`。`S0`、`S_M3`、`S_lerp` 都保持 10299 行；`S_drop` 单独使用 9269 行及其重映射 graph。四路应固定模型、epoch、batch size、学习率、随机种子和 `data/hhar.pkl`；只有 `X_path`、标签文件以及 `S_drop` 的 graph 不同。统计结果由 XHAR-SDCN 的 `P` 分布行提供，`scripts/summarize_har_statistics.py` 可汇总最终和诊断用的最佳 NMI/ARI：

```bash
python scripts/summarize_har_statistics.py \
  statistics/har_missing_10rows_102cols_*.csv
```

正式结论应使用多个训练 seed 的均值±标准差；`S_drop` 还必须单独报告保留的样本数，不能只比较聚类指标。

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

没有 GPU 时删掉 `--gpu 0` 即可，程序会回退到 CPU。多 GPU 机器上请显式指定 `--gpu`；默认值是 `1`，不一定是当前可用的设备。

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

`--model_types EGSAGE_EGSAGE_EGSAGE` 用下划线分隔 GNN 层，可以替换为仓库支持的层类型组合。`--node_dim` 和 `--edge_dim` 控制节点/边隐空间宽度；数据较大或显存不足时优先降低它们。

`--apply_attr` 和 `--apply_peer` 分别打开 FCU 和 SRU。SRU 的 `--sample_peer_size` 越大，采样上下文越多但计算和显存开销越高。`--sample_strategy random_sample` 不需要相似度缓存；`cos-similarity` 会按 `--update_cos_sample_prob_every` 个 epoch 更新采样概率。

当样本数很大时，`--very_large_dataset` 会让 SRU 使用较小的候选空间，大小由 `--sample_space_size` 控制，以避免维护完整的样本两两相似度表。

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

**CUDA 设备错误**：显式指定有效的 `--gpu`，或在无 CUDA 环境中删除该参数。程序会在 CUDA 不可用时使用 CPU。

**运行很慢或显存不足**：降低 `node_dim`、`edge_dim`、`sample_peer_size`；大数据集使用 `--very_large_dataset`。完整图的边数约为 `2 × N × M`，样本数和特征数同时增大时内存会迅速上升。

**实验脚本参数报错**：仓库脚本中的旧参数名 `--impute_hidden` 已由入口兼容；如果自定义脚本使用其它未声明参数，请以 `train_mdi.py` 和 `uci/uci_subparser.py` 中的 parser 定义为准。

## 9. 当前实现边界

本次整理保持了原有计算语义。以下行为是当前实现的既有设计，后续若要改变需要单独验证：

1. `GNNStack.feature_nodes` 目前以普通 Tensor 保存，虽然设置了 `requires_grad=True`，但没有注册为 `nn.Parameter`；它不会像标准模块参数那样自动出现在 `model.parameters()` 中。
2. `concat_states` 分支保留了原有接口；当前 forward 会收集中间状态，但没有额外改变最终返回张量的拼接语义。
3. 评估指标默认在归一化特征值空间计算，不能直接当作原始单位的 RMSE。
4. 代码使用整张样本-特征图，超大数据集需要从节点维度、SRU 候选空间和数据切分三个方向共同控制资源。

## License

MIT
