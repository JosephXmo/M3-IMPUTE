# UCI-HAR：随机行划分的 M3 → XHAR 三组对照

这条独立流水线使用 UCI-HAR 官方 561 维特征，不再导出 M3 的节点 representation。
所有数据适配都位于 `preprocess_cross_comparison.py`，沿用现有 M3 训练入口、图编码器和 ImputeNet；训练器已修复预测头整批展开导致的显存问题。
M3-XHAR 联合训练不在此流程内。

## 1. 执行命令

以下命令均从 **M3-IMPUTE 仓库根目录**执行；`python` 应指向已经安装 M3 依赖的环境。
两仓库默认放在同一父目录。仅准备数据需要 NumPy；训练与补全需要 README 中的 PyTorch/PyG 等依赖。

### 准备 S_train、S₀、S_drop，更新三个 XHAR 配置的数据路径

```bash
python uci/raw_data/har/preprocess_cross_comparison.py prepare \
  --rate 0.10 --seed 42 --update-xhar-configs
```

默认读取 `uci/raw_data/har/original/UCI HAR Dataset`，将完整数据按原始 train 后 test 的顺序合并。
默认输出到 `../XHAR-SDCN/data/missing-cross-comparison`。
这里 `--rate 0.10` 表示**每个特征单元格独立以 10% 概率缺失**，不是旧实验的“10% 行 × 每行 102 列”。
实际缺失率会略有随机波动，写入 `manifest.json`。

### 64/64 训练并保存权重

```bash
python uci/raw_data/har/preprocess_cross_comparison.py train \
  --node-dim 64 --edge-dim 64 --epochs 200 --gpu 0 \
  --apply-attr --apply-peer --impute-batch-size 1024 \
  --log-dir cross_random_64_64_seed42
```

### 32/16 训练并保存权重，降低表示宽度以节约显存

```bash
python uci/raw_data/har/preprocess_cross_comparison.py train \
  --node-dim 32 --edge-dim 16 --epochs 200 --gpu 0 \
  --apply-attr --apply-peer --impute-batch-size 1024 \
  --log-dir cross_random_32_16_seed42
```

两条命令都会训练 **全部 S_train 样本和全部 561 个输入特征**。
减小的是 GNN 节点/边表示宽度及预测头隐藏宽度，不是删除特征列或减少样本。
`--apply-attr` 开启 FCU，`--apply-peer` 开启 SRU，默认采用随机 peer。
这两项与原训练入口一样均为可选；删除它们会训练没有相应上下文模块的消融版本，不能和完整模块版本混称。
`--impute-batch-size` 控制每批预测的目标单元格数，默认 `1024`；若 FCU/SRU 仍报 OOM，可降到 `512` 或 `256`。训练器每轮只编码一次图，预测头逐批算损失并立即反向，累积节点表示梯度后统一回传给图编码器，每轮仍只更新一次参数。验证和测试预测也分块；不会在最后拼接所有带梯度的预测而保留整轮中间张量。

例如，重跑 12/10 配置：

```bash
python uci/raw_data/har/preprocess_cross_comparison.py train \
  --node-dim 12 --edge-dim 10 --epochs 200 --gpu 0 \
  --apply-attr --apply-peer --impute-batch-size 1024 \
  --log-dir cross_random_12_10_seed42_batched
```

过去 6969 × 561 的训练图会一次预测约 547 万条双向目标边，单个 FCU `[目标数, 561]` 浮点张量就约 11.44 GiB，宽度降到 12/10 也不会缩小这两个维度。现在只预测约 274 万个唯一单元格并分块；batch 为 1024、peer 为 5 时，单个 SRU `[1024, 5, 1122]` 拼接张量约 21.9 MiB。这只是一个张量的大小，不是整个训练进程的显存估计。图编码仍为全图，降低表示宽度仍可减少它的显存需求。

`feature_nodes` 同时修复为可训练参数；旧完整模块权重加载时保留其向量数值。新训练结果可能与旧实现不同。不同 batch 大小会改变 peer/dropout 的随机数顺序，不能期望相同 seed 的训练逐位一致。

`--epochs 200` 是便于启动的显式预算，不表示已经证明收敛。可按事先确定的预算调节，但不应依靠 S₀ 或 XHAR 指标选择 M3 checkpoint。
训练包装器自动传入 `--save_model --valid 0 --seed 42`；其中 seed 从准备阶段的 manifest 读取。
每次训练使用新的 `--log-dir`，避免覆盖已有权重。

训练前默认显示 `[Startup]` 阶段及耗时，无需开启 `--display_log`。UCI 图的边列表、边值构建按行显示 tqdm；TXT 读取、归一化、掩码划分、设备搬运和模型初始化分别计时。`Training ready` 表示初始化结束，随后进入 epoch 进度条。包装器使用无缓冲子进程，重定向输出时阶段提示也会及时刷新。

32/16 权重及训练来源记录保存在：

```text
uci/test/har_cross_train/cross_random_32_16_seed42/
├── model.pt                   # 完整 GNNStack 模块
├── impute_model.pt            # 完整 ImputeNet 模块
└── cross_comparison.json      # 训练数据与权重的 SHA-256、命令及 seed
```

64/64 对应目录为 `cross_random_64_64_seed42`。
原训练器会在循环中把打印的 `args.seed` 改为重复实验编号（单次为 0），但最初 NumPy/PyTorch 初始化和数据掩码仍使用命令传入的 42。
此流水线不修改这一既有行为，包装器记录实际传入的种子。包装器默认 `--gpu 0`，将编号原样传给训练入口。训练入口用 `torch.backends.mps.is_available()` 检测：MPS 可用时，`0` 为 MPS、`1` 为 `cuda:0`，后续依次递增；否则编号直接对应 CUDA。所选 CUDA 后端不可用时提示并回退 CPU，可见设备编号越界时报错。补全阶段仍通过 `--device mps` 显式选择 MPS。

### 使用训练好的权重生成 S_m3

```bash
python uci/raw_data/har/preprocess_cross_comparison.py impute \
  --checkpoint-dir uci/test/har_cross_train/cross_random_32_16_seed42 \
  --device cuda:0 --chunk-size 128
```

使用 64/64 权重时更换 checkpoint 目录即可。输出为 `../XHAR-SDCN/data/missing-cross-comparison/hhar_m3.txt`。
这一步将 S_drop 在 mask 为 1 的位置转换成 NaN，用 S_train 的缩放器归一化，只对缺失位置预测，再还原为原特征单位；所有未缺失的值保持不变。
运行时会把 checkpoint 中的样本数量和 peer 缓存适配到新 S₀ 图，学习到的权重不变。
cosine peer 模式会从可见 S₀ 数据重新建缓存，把负采样权重截为 0；全零行回退到等权，若有效候选不足则报错。随机 peer 是默认且不依赖这个缓存。

`--chunk-size` 只控制每次 ImputeNet 调用的缺失目标单元格数；GNN 编码仍处理整个 S₀ 可见图。
减小它能降低预测头峰值显存，不能消除全图编码的显存消耗。
`--device cpu` 可用于没有 CUDA 的环境，但仍需足够内存。
当前本地没有完成这一步所需的真实训练权重；不能用 S_drop 冒充 S_m3。

已有全 HAR 权重见过 S₀，旧被试划分的权重也不对应新 S_train；两者都不能用于本次对照，必须重新训练。
补全阶段要求本包装器训练后生成的 `cross_comparison.json`，并核对训练数据及权重摘要，避免误用旧权重。

三个阶段都支持 `--output-dir`；使用自定义目录时，每一步都传同一个目录。
`--update-xhar-configs` 仅支持目标为 `<xhar-root>/data/missing-cross-comparison` 的情况；其他目录需手动改 X_path/y_path。

## 2. 数据来源、划分与实验含义

直接读取官方 `X_train.txt`（7352 × 561）和 `X_test.txt`（2947 × 561），按 train 后 test 拼接为 10299 × 561。
活动标签和被试编号按相同顺序拼接。流水线不重新提取特征；官方特征来自 2.56 秒、50% 重叠窗口的时域/频域统计，官方表已做归一化。

使用 `np.random.default_rng(seed).permutation(10299)` 无放回随机排列全部行。
前 `floor(10299 × 0.7) = 7209` 行作为 S_train，其余 3090 行作为 S₀；两部分均保留随机排列后的顺序，不排序回原始行号。
这是普通随机抽样，不按类别分层、不保持被试隔离；标签和被试编号只跟随行号重排，不参与抽样决策。

| 部分 | 大小 | 用途 |
|---|---:|---|
| S_train | 7209 × 561 | M3 训练及训练统计量拟合 |
| S₀ family | 3090 × 561 | S₀、S_drop、S_m3 三种 XHAR 输入条件 |

默认 seed 为 42；相同源数据、ratio、seed 对应相同划分，调整 missing rate 不改变划分。
抽样和缺失掩码各自使用以同一 seed 初始化的生成器，互不消耗对方的随机数状态。
保存两侧原始行号及 manifest 中的被试集合，检查行号不重叠且合起来覆盖全部 10299 行。

新协议比较混合被试分布下的补全效果。它不再评估跨被试泛化：同一被试和具有原始信号重叠的相邻窗口可能落到两侧。
随机抽样有助于减小人为分段带来的分布偏差，但不保证两个有限样本分布完全相同，也不能预先证明 XHAR 指标会改善。
随机行序没有时间邻接含义，因此本流程移除 S_lerp 生成、`lerp` 子命令和下游实验条件。

M3 仍采用现有的自监督训练：`--train-edge 0.7` 固定划分 S_train 中的特征单元格，`--known 0.7` 在每轮再随机隐藏一部分训练单元格。
`train_labels` 是原始归一化特征值，不是 HAR 活动类别。无验证集，保存最后一轮权重。
训练器内部的 cell test 与按行随机抽出的 S₀ 是不同层级；S₀ 不进入 M3 优化或缩放器拟合。
`--known` 和 `--train-edge` 不改变样本层面的划分。

三个变体分别重新建图、预热 AE 并训练 XHAR，比较 M3 未见行上的下游聚类输入质量。
这不等于固定一个已训练 XHAR 后在独立测试集上测泛化；活动标签单独保存，供下游指标计算。

## 3. 缺失值细节

| 变体 | mask 位置的值 | 是否访问被遮挡真值 |
|---|---|---|
| S₀ | 真实特征 | 作为完整对照 |
| S_drop | S_train 同列均值 | 否 |
| S_m3 | S_train 训练所得 M3 预测 | 否；预测结束后才读取真值算误差 |

S_drop 保留全部 3090 行，名字中的 drop 表示特征被遮挡。
均值填充使矩阵为有限数值且保留尺寸，但会降低方差、改变相关性；它不保持完整数学分布。
均值及 M3 的 MinMax 缩放参数均来自 S_train，避免从被隐藏的 S₀ 真值估计参数。
补全时依据显式 mask 将占位值转为 NaN，不把均值占位当成观测，也不把真实均值误当成缺失。

## 4. 产物和 XHAR 执行

主输入为 `hhar_s0.txt`、`hhar_drop.txt`、`hhar_m3.txt`，都是 3090 × 561 的无表头 NumPy 文本。
共同标签为 `hhar_labels.txt`，保留官方 1–6 类别编号。掩码为同形状的 0/1 文本，`missing_indices.txt` 为从 0 开始的行列坐标。
`s0_row_ids.txt` 与 `train_row_ids.txt` 是相对于完整 train+test 的原始行号；`s0_subjects.txt` 保存 S₀ 被试编号。
`hhar_missing.txt` 是显式 NaN 版本，供核验或其他补全方法使用，不用作 XHAR 输入。

训练输入另外保存在 M3 仓库的 `uci/raw_data/har_cross_train/data/data.txt`（7209 × 562）：
最后一列仅为兼容 UCI loader 的 HAR 标签列，M3 补全只使用前 561 列。
旧 `uci/raw_data/har/data` 和其他数据集保持原样。

旧 `hhar_lerp.txt` 和 `segments.txt` 会改名为带 `.retired-<摘要>` 的文件保留，避免误作当前输入。
重复 prepare 使用同一 seed/rate 会生成同一掩码；改变数据、mask 或统计量时，旧 `hhar_m3.txt` 会移为 `hhar_m3.stale-<摘要>.txt`，需重新补全。
只改变 missing rate 时 S_train 未变，可以继续用同一 checkpoint。改变划分后必须重新训练，并使用新的 `--log-dir`。
manifest schema 2 记录随机划分规则、共享被试、训练数据、掩码、S₀、S_drop、训练统计量、行号及标签摘要。

生成 hhar_m3 后，在 XHAR 仓库执行三组各五次的批量实验及绘图：

```bash
cd ../XHAR-SDCN
bash experiments/har_missing_10rows_102cols/run_all.sh
```

结果写入独立的 `statistics/cross-random-*/` 目录，包含本轮 Parquet、manifest 副本、三组对照图和 27 行指标范围 CSV，避免混入旧划分的结果。
现有五次重复的 seed 均为 42，不能把它们解释为独立随机种子误差范围。

也可以逐组执行：

```bash
source .venv/bin/activate
python train.py -C experiments/har_missing_10rows_102cols/hhar_s0.yaml
python train.py -C experiments/har_missing_10rows_102cols/hhar_drop.yaml
python train.py -C experiments/har_missing_10rows_102cols/hhar_m3.yaml
```

自动配置更新只替换三份 YAML 的 X_path/y_path，其余设备、图参数、AE 预热、模型、训练和 seed 设置不变。
目录保留旧实验名，但 active 数据路径指向新产物。手动执行仍使用 YAML 的 `statistics_path: auto`，分析时须自行区分新旧结果。

补全后 manifest 记录隐藏单元格的 MAE/RMSE（原始特征单位），另行比较 XHAR 的 ACC/NMI/ARI 等下游指标。
局部测试：`PYTHONPATH=uci/raw_data/har:. python -m unittest uci/raw_data/har/test_cross_comparison.py -v`。
测试覆盖随机划分的互斥/完整/可复现、特征与标签对齐、训练统计量、掩码、旧产物失效及真实 ImputeNet 的新样本数量适配；不替代全量 HAR 训练与目标 GPU 验收。
