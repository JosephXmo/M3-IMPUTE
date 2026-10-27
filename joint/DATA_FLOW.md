# 全量 UCI-HAR 联合架构数据流

`N=10,299` 个样本，`F=561` 个特征，`K=6` 个簇。
官方 train/test 只决定输入拼接顺序；全体样本参与本次传导式训练。

```mermaid
flowchart TD
    X["X_train + X_test<br/>完整特征 X：N×F"] --> Scale["全量输入拟合 Min-Max<br/>输出 X01：N×F，offset/scale：F"]
    Scale --> Mask["每轮自动挖空<br/>输出可见 mask M：N×F"]
    Mask --> Visible["可见矩阵 V：N×F<br/>M=True 保留 X01；其余为 epsilon"]
    Mask --> BG["只用可见单元建双向二部图<br/>edge_index：2×2E；edge_attr：2E×1"]
    Visible --> Encoder["M3 GNNStack<br/>特征节点参数 + V 初始化<br/>可见图消息传递"]
    BG --> Encoder
    Encoder --> H["节点 embedding H：(N+F)×D<br/>前 N 行样本；后 F 行特征"]
    H --> Decoder["M3 ImputeNet<br/>可选 FCU / SRU<br/>全 N×F 个单元分块解码"]
    Mask --> Decoder
    Decoder --> R["M3 完整重构 R：N×F<br/>每个位置均为预测值"]
    R --> Inverse["用固定 offset/scale 逆变换<br/>XHAR 输入 XR：N×F"]
    Inverse --> Topology["离散选边：余弦 KNN<br/>+ 同片段时间边 + 自环"]
    Subject["subject_train/test + 行序<br/>样本归属与片段边界"] --> Topology
    Subject --> RoPE["全局行号差的 RoPE<br/>按当前边生成位置编码"]
    Topology --> RoPE
    Topology --> Edge["可微边投影与端点聚合<br/>输出 edge_attr：Ex×De"]
    Inverse --> Edge
    RoPE --> Edge
    Inverse --> Juicer["JuicerESageRoPE<br/>AE + 五层图网络 + 聚类中心"]
    Topology --> Juicer
    Edge --> Juicer
    Juicer --> Out["AE重构：N×F；latent：N×Zdim<br/>Q：N×K；GNN概率 Z：N×K"]
    Out --> P["P = target_distribution(Q)<br/>全图统计，停止梯度"]
    Out --> XLoss["XHAR 损失<br/>AE重构 + KL(P∥Q) + KL(P∥Z)<br/>+ 同片段时间平滑"]
    P --> XLoss
    Inverse -. "停止目标侧梯度" .-> XLoss
    R --> MLoss["M3 重构损失<br/>仅当轮隐藏位置的 MSE"]
    Scale -. "原值只作为监督目标" .-> MLoss
    Mask --> MLoss
    XLoss --> Total["加权总损失<br/>一次联合参数更新"]
    MLoss --> Total
    Total -. "反向：XR → R → decoder → H → encoder" .-> Decoder
    Total -. "更新 Juicer 和边投影" .-> Juicer
    Out --> Argmax["Q / Z / P 各自 argmax<br/>输出每行的聚类 ID：N"]
    P --> Argmax
    Y["y_train + y_test<br/>活动标签 y：N"] --> Metrics["XHAR 聚类评估<br/>Hungarian匹配后的 Accuracy/F1/P/R<br/>+ NMI / ARI"]
    Argmax --> Metrics
```

## 每种输入的去向

| 输入 | 使用模块 | 输出与用途 |
|---|---|---|
| 原始特征 X | Min-Max | X01、固定 offset/scale；允许使用全体输入的传导式预处理 |
| X01 与当轮随机数 | 挖空模块 | bool mask M；每轮新采样，无永久 7–3 划分 |
| X01、M | 可见矩阵与图构造 | V、仅可见单元的二部图和数值边属性 |
| V、可见图 | M3 encoder | `[样本节点; 特征节点]` embedding H |
| H、M、目标单元坐标 | ImputeNet / FCU / SRU | 完整预测 R；隐藏位置和可见位置都用预测输出 |
| R、offset/scale | 可微逆变换 | 与原始 UCI 数值尺度一致的 XR |
| XR、subject/片段信息 | XHAR 图构造 | 稀疏 edge_index；KNN 只读取 M3 重构，时间边不跨片段 |
| XR、edge_index、行号位置编码 | EdgeFeatures | 每条边的可微属性，梯度能回到 XR |
| XR、XHAR 图、边属性 | Juicer | AE 重构、AE latent、Q、GNN 概率 Z |
| Q | DEC target distribution | 停止梯度的 P，不读取 y |
| 原始 X01、M、R | M3 loss | 原值只在隐藏单元 MSE 中作为自监督目标 |
| Q、Z、P、AE 重构、XR | XHAR loss | 图聚类与重构目标；XR 的 target 副本停止梯度 |
| 活动标签 y、预测聚类 ID | metrics | 仅生成评估数值，不影响 loss、图、KMeans 或参数更新 |

## 评估时的变化

评估使用固定 `eval_seed` 生成的 mask，完整走同一条 M3 → XHAR 路径，关闭 dropout，
固定 SRU 的随机采样序列。使用最近一次保存的 XHAR 拓扑；边属性仍随当前重构计算。
所有 N 行参与评估。`M3 hidden RMSE` 额外使用隐藏原值作诊断，活动标签仍只流向 metrics。

## 实际反向实现

图中的反向箭头由 `joint/bridge.py` 显式实现：decoder 首次分块 forward 保存预测与
RNG；XHAR 得到 `dL/dR` 后逐块重算 decoder，累计参数梯度与 `dL/dH`，最后回传
encoder。这与对应分块顺序的普通 autograd 已做对照，包含 FCU/SRU、dropout 和随机 peer。
它节省 decoder 激活，不改变全部样本参与的范围，也不消除全图 encoder 的内存成本。
