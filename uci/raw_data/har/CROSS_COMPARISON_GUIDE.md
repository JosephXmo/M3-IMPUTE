# UCI-HAR：被试分离的 M3 → XHAR 四组对照

这条独立流水线使用 UCI-HAR 官方 561 维特征，不再导出 M3 的节点 representation。
所有数据适配都位于 `preprocess_cross_comparison.py`，沿用现有 M3 训练器、图编码器和 ImputeNet。
M3-XHAR 联合训练不在此流程内。

## 1. 执行命令

以下命令均从 **M3-IMPUTE 仓库根目录**执行；`python` 应指向已经安装 M3 依赖的环境。
两仓库默认放在同一父目录。仅准备数据和插值需要 NumPy；训练与补全需要 README 中的 PyTorch/PyG 等依赖。

### 准备 S_train、S₀、S_drop、S_lerp，更新四个 XHAR 配置的数据路径

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
  --apply-attr --apply-peer --log-dir cross_64_64_seed42
```

### 32/16 训练并保存权重，降低表示宽度以节约显存

```bash
python uci/raw_data/har/preprocess_cross_comparison.py train \
  --node-dim 32 --edge-dim 16 --epochs 200 --gpu 0 \
  --apply-attr --apply-peer --log-dir cross_32_16_seed42
```

两条命令都会训练 **全部 S_train 样本和全部 561 个输入特征**。
减小的是 GNN 节点/边表示宽度及预测头隐藏宽度，不是删除特征列或减少样本。
`--apply-attr` 开启 FCU，`--apply-peer` 开启 SRU，默认采用随机 peer。
这两项与原训练入口一样均为可选；删除它们会训练没有相应上下文模块的消融版本，不能和完整模块版本混称。
降低表示宽度不保证解决所有 OOM，尤其 FCU/SRU 的中间张量仍与目标单元格数和 561 维输入相关；原训练器仍是全图训练，没有新增 batch-size 支持。

`--epochs 200` 是便于启动的显式预算，不表示已经证明收敛。可按事先确定的预算调节，但不应依靠 S₀ 或 XHAR 指标选择 M3 checkpoint。
训练包装器自动传入 `--save_model --valid 0 --seed 42`；其中 seed 从准备阶段的 manifest 读取。
每次训练使用新的 `--log-dir`，避免覆盖已有权重。

32/16 权重及训练来源记录保存在：

```text
uci/test/har_cross_train/cross_32_16_seed42/
├── model.pt                   # 完整 GNNStack 模块
├── impute_model.pt            # 完整 ImputeNet 模块
└── cross_comparison.json      # 训练数据与权重的 SHA-256、命令及 seed
```

64/64 对应目录为 `cross_64_64_seed42`。
原训练器会在循环中把打印的 `args.seed` 改为重复实验编号（单次为 0），但最初 NumPy/PyTorch 初始化和数据掩码仍使用命令传入的 42。
此流水线不修改这一既有行为，包装器记录实际传入的种子。GPU 使用现有训练器的设备选择逻辑，单卡建议 `--gpu 0`。

### 使用训练好的权重生成 S_m3

```bash
python uci/raw_data/har/preprocess_cross_comparison.py impute \
  --checkpoint-dir uci/test/har_cross_train/cross_32_16_seed42 \
  --device cuda:0 --chunk-size 128
```

使用 64/64 权重时更换 checkpoint 目录即可。输出为 `../XHAR-SDCN/data/missing-cross-comparison/hhar_m3.txt`。
这一步将 S_drop 在 mask 为 1 的位置转换成 NaN，用 S_train 的缩放器归一化，只对缺失位置预测，再还原为原特征单位；所有未缺失的值保持不变。
运行时会把 checkpoint 中的样本数量和 peer 缓存适配到新 S₀ 图，学习到的权重不变。
cosine peer 模式会从可见 S₀ 数据重新建缓存，把负采样权重截为 0；全零行回退到等权，若有效候选不足则报错。随机 peer 是默认且不依赖这个缓存。

`--chunk-size` 只控制每次 ImputeNet 调用的缺失目标单元格数；GNN 编码仍处理整个 S₀ 可见图。
减小它能降低预测头峰值显存，不能消除全图编码的显存消耗。
`--device cpu` 可用于没有 CUDA 的环境，但仍需足够内存。
当前本地没有完成这一步所需的真实训练权重；不能用 S_drop 或 S_lerp 冒充 S_m3。

已有全 HAR 权重不适用于严格的被试分离实验，因为它见过 S₀。
补全阶段要求本包装器训练后生成的 `cross_comparison.json`，并核对训练数据及权重摘要，避免误用旧权重。

### 单独重算插值，无需原始 HAR 或完整 S₀

```bash
python uci/raw_data/har/preprocess_cross_comparison.py lerp \
  --radius 5 --side-margin 0.10 --side-ratio 1.5 --min-overlap 8
```

`lerp` 只读取 `hhar_drop.txt`、`missing_mask.txt`、`segments.txt`、`train_statistics.json` 和 `manifest.json`。
不读取 `hhar_s0.txt`、标签或 `original` 中的 raw data。上述文件及目录可以单独复制到其他机器重算。
四个阶段都支持 `--output-dir`；使用自定义目录时，每一步都传同一个目录。
`--update-xhar-configs` 仅支持目标为 `<xhar-root>/data/missing-cross-comparison` 的情况；其他目录需手动改 X_path/y_path。

## 2. 划分与实验含义

首先在完整 10299 行的 70% 位置，即第 7209 行前后设置边界，再把跨边界被试的全部样本移到 S₀。
原始行序下被试 30 跨边界，最终得到：

| 部分 | 大小 | 用途 |
|---|---:|---|
| S_train | 6969 × 561 | M3 训练及训练统计量拟合 |
| S₀ family | 3330 × 561 | 四种 XHAR 输入条件 |

S₀ 被试为 2、4、9、10、12、13、18、20、24、30，与 S_train 没有重叠。
这是原行序下的被试优先约 7:3 划分，不是随机分层样本划分；会与官方 7352/2947 划分有所不同。
窗口在同一被试的**连续原始行段**内搜索，避免从同被试不连续片段跨界取邻居。
官方特征表的邻行不保证来自严格连续的时间点，因此这是一种基于行邻域的离线插值启发式，不等价于原始传感器时间序列插值。

M3 仍采用现有的自监督训练：`--train-edge 0.7` 固定划分 S_train 中的特征单元格，`--known 0.7` 在每轮再随机隐藏一部分训练单元格。
`train_labels` 是原始归一化特征值，不是 HAR 活动类别。无验证集，保存最后一轮权重。
这里训练器内部的 cell test 与整个被试分离出来的 S₀ 是不同层级；S₀ 不进入 M3 优化或缩放器拟合。
如需改变单元格训练策略，可在 `train` 阶段传入 `--known` 和 `--train-edge`；它们不改变样本层面的划分。

四个变体用于分别重新建图、预热 AE 并训练 XHAR，这是“在 M3 未见被试上比较下游聚类输入质量”的实验。
它不等于“固定一个已训练 XHAR，在独立测试集上测泛化”。当前方案允许在每个 S₀ 变体重新拟合 XHAR；活动标签单独保存，仅供下游指标计算。

## 3. 缺失值和插值细节

| 变体 | mask 位置的值 | 是否访问被遮挡真值 |
|---|---|---|
| S₀ | 真实特征 | 作为完整对照 |
| S_drop | S_train 同列均值 | 否 |
| S_lerp | 相似度选方向的局部线性估计 | 否 |
| S_m3 | S_train 训练所得 M3 预测 | 否；预测结束后才读取真值算误差 |

S_drop 保留全部 3330 行，名字中的 drop 表示特征被遮挡。
均值填充使矩阵为有限数值且保留尺寸，但会降低方差、改变相关性；它**不保持完整数学分布**。
均值来自 S_train，避免从被隐藏的 S₀ 真值估计参数。

对每个缺失单元格，插值策略如下：

1. 在同一连续被试段中检查前后最多各 5 行。
2. 用双方都可见的其他特征计算标准化均方距离 `d²`，相似度为 `exp(-d²/2)`。
   缺失占位值、目标缺失列和活动标签不参与距离；共同可见特征少于 8 列的邻居不参与。
3. 对当前目标列，仅采用该列未缺失的邻居。比较左右侧平均相似度：
   同时满足差值至少 0.10、比例至少 1.5 时，选更相似的一侧；否则两侧一起使用。
4. 按相似度加权拟合局部直线，在缺失行位置求值。单侧时是外推，两侧时是局部插值估计；
   一个可用邻居时复制其值，没有可用邻居时使用 S_train 列均值。
5. 将预测值限制在 S_train 同列最小/最大值之间。已补出的值不再充当后续单元格的已知观测。

这个策略可复现且不读取隐藏真值，但相似度阈值、局部线性假设和范围限制是否有效仍需实验验证。
`manifest.json` 记录左右/双侧/单邻居/均值回退及截断次数。

## 4. 产物和 XHAR 执行

主输入为 `hhar_s0.txt`、`hhar_drop.txt`、`hhar_lerp.txt`、`hhar_m3.txt`，都是 3330 × 561 的无表头 NumPy 文本。
共同标签为 `hhar_labels.txt`，保留官方 1–6 类别编号。掩码为同形状的 0/1 文本，`missing_indices.txt` 为从 0 开始的行列坐标。
`s0_row_ids.txt` 与 `train_row_ids.txt` 是相对于完整 train+test 的原始行号；`s0_subjects.txt` 和 `segments.txt` 保存被试及连续段信息。
`hhar_missing.txt` 是显式 NaN 版本，供核验或其他补全方法使用，不用作 XHAR 输入。

训练输入另外保存在 M3 仓库的 `uci/raw_data/har_cross_train/data/data.txt`（6969 × 562）：
最后一列仅为兼容未修改的 UCI loader 的 HAR 标签列，M3 补全只使用前 561 列。
旧 `uci/raw_data/har/data` 和其他数据集保持原样。

重复 prepare 使用同一 seed/rate 会生成同一掩码；改变数据、mask 或统计量时，旧 `hhar_m3.txt` 会移为 `hhar_m3.stale-<摘要>.txt`，需重新补全。
只改变 missing rate 时 S_train 未变，可以继续用同一 checkpoint。改变划分后必须重新训练。

确认 hhar_m3 已生成，再到 XHAR 仓库执行：

```bash
cd ../XHAR-SDCN
source .venv/bin/activate
python train.py -C experiments/har_missing_10rows_102cols/hhar_s0.yaml
python train.py -C experiments/har_missing_10rows_102cols/hhar_drop.yaml
python train.py -C experiments/har_missing_10rows_102cols/hhar_lerp.yaml
python train.py -C experiments/har_missing_10rows_102cols/hhar_m3.yaml
```

自动配置更新只替换现有四份 YAML 的 X_path/y_path，其余设备、图参数、AE 预热、模型、训练和 seed 设置不变。
该目录保留旧实验名，但 active 数据路径已指向新产物。现有 `run_all.sh` 每种条件重复 5 次且种子均为 42，不能把这些重复解释为独立随机种子误差范围。
脚本最后引用的 `plot_har_statistics.py` 当前不在该目录，批量运行最后绘图会报错；上述四条训练命令不依赖它。

补全后 manifest 会记录隐藏单元格的 MAE/RMSE（原始特征单位），另行比较 XHAR 的 ACC/NMI/ARI 等下游指标。
局部测试可以执行 `PYTHONPATH=uci/raw_data/har:. python -m unittest uci/raw_data/har/test_cross_comparison.py -v`；
这些测试覆盖划分、训练统计量、遮挡真值隔离、插值方向及真实 ImputeNet 的新样本数量适配，不能替代完整 GNN 训练验收。
