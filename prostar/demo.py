"""
ProSTAR 论文算法复现 Demo — 带完整逐行注释版本
================================================
原论文: "ProSTAR: Prototype-Based Satellite-Terrestrial Adaptation
         for Resource Constrained Co-Inference"
作者:   Shenhu Zhang, Shi Yan, Fengxian Guo, Mianji Li, Nan Li, Mugen Peng
单位:   北京邮电大学 + 中国移动研究院

本 Demo 精确复现论文中的4张核心图表:
  Fig.2 — 代理模型对 A(l,d) 关系的拟合效果 (4个Stage的sigmoid退化曲线)
  Fig.3 — 穷举搜索 vs 提出方案的准确率与联合代价热力图对比
  Fig.4 — 最优配置下的混淆矩阵 (所有类视为新类, d=256, l=3)
  Fig.5 — 3D原始中间特征 vs 1D压缩特征的准确率/延迟/能耗柱状图对比

运行方式: python -m prostar.demo
"""

# ---------- 基础依赖 ----------
import sys, os, time
# 将项目根目录加入搜索路径, 保证 prostar 包可以被正确导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                    # 矩阵运算: 原型构建、梯度下降、批量匹配
import matplotlib                      # 科学绘图
matplotlib.use('Agg')                  # 使用非交互式后端, 直接渲染到PNG文件
import matplotlib.pyplot as plt        # 绘图API
from matplotlib.colors import LinearSegmentedColormap  # 自定义渐变色 (混淆矩阵用)
from matplotlib.patches import Patch   # 图例补丁 (备用)
from collections import defaultdict    # 自动创建嵌套字典 (备用)

# 从核心模块导入所需组件
# prototype.py — 原型构建、特征压缩、余弦匹配、EMA更新
from prostar.prototype import (
    ProSTARPipeline, ProSTARConfig,      # 完整推理流水线 + 配置对象
    initialize_prototypes,               # Eq.7: 嵌套子空间原型初始化
    cosine_similarity_match,             # Eq.9: 余弦相似度匹配
    compress_feature,                    # Eq.8: 特征压缩投影
    PrototypeBank,                       # 原型仓库数据结构
)
# surrogate.py — 代理模型拟合 + 联合优化 + 在线决策
from prostar.surrogate import (
    LayerSurrogateParams,                # 单层代理模型参数字典 (A_max, A_min, α, d0)
    fit_surrogate_params,                # Eq.18: 非线性最小二乘参数拟合
    SystemParams,                        # 星地协同系统参数 (计算/通信/能耗)
    OptimizationWeights,                 # 联合代价函数权重 (λ1, λ2, λ3)
    online_decision,                     # Eq.19: 在线自适应决策
    joint_cost,                          # Eq.13: 联合代价 J(l,d)
    generate_d_candidates,               # 维度候选集生成 (论文设定24个均匀级别)
    _stable_sigmoid,                     # 数值稳定的sigmoid函数
)


# ============================================================
# 第一部分: 全局绘图样式 — 统一论文风格
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',              # 字体族: 无衬线 (学术论文标配)
    'font.sans-serif': ['Arial', 'DejaVu Sans'], # 优先Arial, 回退DejaVu Sans
    'font.size': 10,                          # 全局基础字号
    'axes.titlesize': 11,                     # 子图标题字号
    'axes.labelsize': 10,                     # 坐标轴标签字号
    'xtick.labelsize': 9,                     # x轴刻度字号
    'ytick.labelsize': 9,                     # y轴刻度字号
    'legend.fontsize': 8,                     # 图例字号
    'figure.dpi': 150,                        # 屏幕显示分辨率
    'savefig.dpi': 150,                       # 文件保存分辨率 (150 dpi适合印刷)
    'savefig.bbox': 'tight',                  # 裁剪空白边距
    'savefig.pad_inches': 0.05,               # 图片四周留白
})


# ============================================================
# 第二部分: 模拟 Swin-Tiny 骨干网络 — 论文用 ImageNet-1K 预训练权重
# ============================================================
class SimulatedFeatureExtractor:
    """
    模拟 Swin-Tiny 的特征提取器
    ============================
    论文 Section V 明确使用 Swin-Tiny 作为骨干, 有4个 Stage (L=4),
    各层通道数 C_l ∈ {96, 192, 384, 768}。

    由于无法在此环境加载真实的 Swin-Tiny + ImageNet 权重 (~28MB),
    我们用带信息衰减先验的随机投影来模拟 DNN 特征的两个关键性质:
    (1) 类间可分性 — 不同类别在特征空间中可被余弦相似度区分
    (2) 信息沿维度非均匀分布 — 前言几个维度承载主要判别信息,
        后段维度为冗余, 这使得维度压缩呈现论文所述的 sigmoid 退化曲线

    这个模拟器生成的特征 f_l ∈ R^{D_l} 能定性复现论文中
    A(l,d) 随 d 减少而呈现的 sigmoid 退化行为。
    """

    def __init__(self, num_classes=21, seed=42):
        """
        初始化虚拟特征提取器
        ————————————————
        num_classes: 类别总数 (UCMerced LandUse 共21类)
        seed:         随机种子 (保证每次运行结果可复现)
        """
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)   # 独立的随机数流

        # Swin-Tiny 各 Stage 输出通道数 (论文配置)
        # Stage1→96, Stage2→192, Stage3→384, Stage4→768
        self.layer_dims = {0: 96, 1: 192, 2: 384, 3: 768}

        # 每个类别有一个 768 维的"语义中心"
        # 这模拟了 ImageNet 预训练后类别的嵌入 (embedding) 分布
        # 不同的语义中心具有随机但相互正交的方向, 确保类间可分性
        self.class_centers = {}
        for c in range(num_classes):
            v = self.rng.randn(768)             # 标准高斯随机生成
            self.class_centers[c] = v / np.linalg.norm(v)  # L2归一化到单位球面

        # 逐层投影矩阵 (模拟 DNN 的线性变换层)
        # 使用 Kaiming/He 初始化: std = sqrt(2 / fan_in)
        # 这能保证信息在逐层传播中不会放大或衰减
        self.layer_projections = {}
        prev_dim = 768                           # 初始隐藏维度
        for l in range(4):
            D_l = self.layer_dims[l]             # 当前层输出维度
            # He-normal 初始化: 均值为0, 标准差为 sqrt(2 / prev_dim)
            self.layer_projections[l] = (
                self.rng.randn(prev_dim, D_l) * np.sqrt(2.0 / prev_dim)
            ).astype(np.float32)
            prev_dim = D_l                       # 更新下一层的输入维度

    def extract_feature(self, class_id: int, layer_idx: int,
                        noise_level: float = 0.05) -> np.ndarray:
        """
        提取单张"图像"在指定分区分割层的 1D 特征向量 f_l ∈ R^{D_l}
        ——————————————————————————————————————————————————————————
        class_id:   类别索引 (0 ~ 20)
        layer_idx:  分区层 (0~3, 对应 Swin-Tiny 的4个 Stage)
        noise_level: 加性高斯白噪声的标准差 (模拟传感器噪声和域偏移)

        处理流程:
          隐藏语义向量 → 逐层线性投影 + tanh 非线性 → 信息衰减加权 → L2归一化
        """
        D_l = self.layer_dims[layer_idx]

        # Step 1: 从类别的语义中心出发, 叠加随机噪声
        # 噪声模拟了: 同类别内部不同图片之间的自然差异 (intra-class variance)
        hidden = self.class_centers[class_id] + noise_level * self.rng.randn(768)

        # Step 2: 逐层传播 (模拟 DNN 的前向计算)
        # 每层: h = tanh(h @ W_l) * 2.0
        # tanh 把输出限制在 (-1, 1), 乘以 2.0 保证激活范围足够
        h = hidden
        for l in range(layer_idx + 1):
            h = h @ self.layer_projections[l]      # 线性投影: (prev_dim,) → (D_l,)
            h = np.tanh(h) * 2.0                   # 非线性激活 (类 Swish)

        # Step 3: 施加信息衰减先验 — 这是本模拟器的核心假设
        # 物理直觉: 深层网络中, 前几个通道 (低频分量) 编码全局/粗粒度特征,
        #   后几个通道 (高频分量) 编码细节/噪声, 对分类的贡献递减
        # 数学形式: w_i = exp(-i / (D_l * 0.15)),  i ∈ [0, D_l-1]
        #   这确保前 ~15% 的维度承载大部分能量
        decay = np.exp(-np.arange(D_l) / (D_l * 0.15))
        f_l = (h * decay).astype(np.float32)

        # Step 4: L2 归一化
        #   将所有特征向量映射到单位超球面上
        #   这样后续余弦相似度就直接等价于点积, 省去反复归一化的开销
        return f_l / (np.linalg.norm(f_l) + 1e-8)

    def extract_batch(self, class_ids: np.ndarray, layer_idx: int,
                      noise_level: float = 0.05) -> np.ndarray:
        """
        批量提取特征
        ———————————
        class_ids:  (N,) 每个样本的类索引
        返回:        (N, D_l) 特征矩阵
        """
        feats = np.zeros((len(class_ids), self.layer_dims[layer_idx]), dtype=np.float32)
        for i, cid in enumerate(class_ids):
            feats[i] = self.extract_feature(cid, layer_idx, noise_level)
        return feats


# ============================================================
# 第三部分: 数据集生成 — 模拟 UCMerced LandUse 的小样本新类场景
# ============================================================
def generate_ucmerced_like_dataset(
    num_classes: int = 21,        # M: 类别总数 (UCMerced 原始21类)
    k_shot: int = 5,              # K: 每类支持集样本数
    num_query: int = 30,          # 每类查询集样本数
    seed: int = 42,               # 随机种子
):
    """
    生成一个 M-way K-shot 的小样本分类数据集
    ========================================
    本函数模拟论文 Section V 的实验场景:
    - 21 个类全部作为"新类" (novel classes), 即 DNN 从未见过
    - 每个新类只提供 K=5 个标注样本 (支持集, support set)
    - 查询集 (query set) 用于评估匹配准确率

    返回字典结构:
      {
        'support': (class_ids, labels),   # 支持集: (M*K,) 的类索引和标签
        'query':   (class_ids, labels),   # 查询集: (M*Q,) 的类索引和标签
        'class_names': [...],             # M 个类名
        'num_classes': M,                 # 类别总数
      }
    """
    rng = np.random.RandomState(seed)
    all_classes = list(range(num_classes))
    rng.shuffle(all_classes)               # 打乱类别顺序 (消除数据集构造偏差)

    # 构建支持集: 每类 K 个样本
    # 标签从 0 开始重新编号 (小样本学习标准做法)
    support_class_ids = []
    support_labels = []
    for m, cls_id in enumerate(all_classes):     # m: 重标号后的类索引 (0~M-1)
        for _ in range(k_shot):                  # 每类生成 K 个
            support_class_ids.append(cls_id)
            support_labels.append(m)

    # 构建查询集: 每类 Q 个样本
    query_class_ids = []
    query_labels = []
    for m, cls_id in enumerate(all_classes):
        for _ in range(num_query):
            query_class_ids.append(cls_id)
            query_labels.append(m)

    class_names = [f"Class_{i:02d}" for i in range(num_classes)]

    return {
        'support': (np.array(support_class_ids), np.array(support_labels)),
        'query': (np.array(query_class_ids), np.array(query_labels)),
        'class_names': class_names,
        'num_classes': num_classes,
    }


# ============================================================
# 第四部分: Fig.2 — 代理模型拟合性能
# ============================================================
def figure_2_surrogate_fitting(extractor, dataset):
    """
    复现论文 Fig.2: Surrogate model fitting performance.
    ====================================================
    对应论文 Section IV-B, Eq.16-18。

    实验流程:
      对4个分区层 (Stage 1~4),
      在24个均匀间隔的压缩维度上测量实际匹配准确率 A(l, d_j),
      然后用改进的 sigmoid 代理模型 Â_l(d) 拟合这些散点。

    每个子图展示:
      - 蓝色散点: 实测 A(l, d_j)
      - 蓝色实线: 拟合的 sigmoid 代理模型
      - 红色虚线: 临界维度 d0 (准确率开始急剧下降的拐点)

    输出文件: prostar_fig2_surrogate_fitting.png (1行×4列, 16''×3.8'')
    """
    print("[Fig.2] 代理模型拟合性能...")

    # 解包数据集
    supp_ids, supp_labels = dataset['support']   # 支持集 (用于构建原型)
    query_ids, query_labels = dataset['query']   # 查询集 (用于评估准确率)
    class_names = dataset['class_names']         # 类名列表

    surrogate_models = []                        # 存储4层的代理模型参数
    empirical_data_all = {}                      # 存储各层的实测数据

    # ---------- 对每一层独立建模 ----------
    for layer_idx in range(4):
        D_l = extractor.layer_dims[layer_idx]    # 该层的最大特征维度

        # Step 1: 用支持集初始化原型 (Eq.7)
        #    噪声水平 0.03 — 模拟支持集的高质量标注
        supp_feats = extractor.extract_batch(supp_ids, layer_idx, noise_level=0.03)
        config = ProSTARConfig(layer_idx=layer_idx, seed=42)
        pipeline = ProSTARPipeline(config)
        pipeline.initialize(supp_feats, supp_labels, class_names)

        # Step 2: 生成24个均匀间隔的压缩维度候选值
        #    论文规定24个压缩级别 (Section V)
        d_candidates = np.linspace(2, D_l, 24).astype(int)

        # Step 3: 在每个压缩维度上测量匹配准确率
        #    这对应论文中的 D^l_profile = {(d_j, A(l, d_j))}  (Eq.17)
        empirical_accs = []
        for d in d_candidates:
            correct = 0
            # 遍历所有查询样本
            for i in range(len(query_ids)):
                # 提取特征 (噪声 0.05 — 模拟查询集的不确定性)
                q_feat = extractor.extract_feature(query_ids[i], layer_idx, noise_level=0.05)
                # 执行 ProSTAR 推理: 压缩 → 匹配 → 预测
                result = pipeline.infer(q_feat, d=int(d))
                if result.pred_class == query_labels[i]:
                    correct += 1
            empirical_accs.append(correct / len(query_ids) * 100)

        d_vals = d_candidates.astype(np.float64)
        acc_vals = np.array(empirical_accs, dtype=np.float64)
        empirical_data_all[layer_idx] = (d_vals, acc_vals)

        # Step 4: 用非线性最小二乘法拟合代理模型参数 θ*_l (Eq.18)
        surrogate = fit_surrogate_params(d_vals, acc_vals, layer_idx, D_l)
        surrogate_models.append(surrogate)

    # ---------- 绘制 1×4 子图 ----------
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    for layer_idx in range(4):
        ax = axes[layer_idx]
        d_vals, acc_vals = empirical_data_all[layer_idx]  # 实测数据
        surrogate = surrogate_models[layer_idx]             # 拟合后的代理模型

        # (a) 蓝色散点 — 各压缩维度下的实测准确率
        ax.scatter(d_vals, acc_vals, s=18, c='#2166AC', alpha=0.7,
                   edgecolors='white', linewidth=0.5, zorder=5)

        # (b) 蓝色曲线 — sigmoid 代理模型 (Eq.16)
        d_smooth = np.linspace(d_vals.min(), d_vals.max(), 300)
        acc_smooth = surrogate.predict_batch(d_smooth)
        ax.plot(d_smooth, acc_smooth, '-', color='#2166AC', linewidth=2.0,
                alpha=0.9, zorder=4, label='Surrogate model')

        # (c) 红色虚线 — 临界维度 d0 (拐点位置)
        #    论文定义: d0 是准确率开始急剧下降的临界值
        ax.axvline(x=surrogate.d0, color='#D73027', linestyle='--',
                   linewidth=1.2, alpha=0.7)
        ax.annotate(f"$d_0^{{{layer_idx}}}$={surrogate.d0:.0f}",
                    xy=(surrogate.d0, acc_vals.mean()),
                    xytext=(surrogate.d0 + 30, acc_vals.mean() - 8),
                    fontsize=8, color='#D73027',
                    arrowprops=dict(arrowstyle='->', color='#D73027', lw=1.0))

        # 子图标题和标签
        ax.set_title(f"Stage {layer_idx + 1}  ($D_l$={surrogate.D_l})",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Target Dimension $d$', fontsize=9)
        if layer_idx == 0:
            ax.set_ylabel('Matching Accuracy (%)', fontsize=9)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(left=0)

    fig.suptitle('Fig. 2: Surrogate model fitting performance',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('prostar_fig2_surrogate_fitting.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  -> prostar_fig2_surrogate_fitting.png")

    return surrogate_models, empirical_data_all


# ============================================================
# 第五部分: Fig.3 — 穷举搜索 vs 提出方案的对比
# ============================================================
def figure_3_exhaustive_vs_proposed(extractor, dataset, surrogate_models):
    """
    复现论文 Fig.3: Comparison between Exhaustive Search and the Proposed Scheme.
    ============================================================================
    这是论文的核心实验 — 验证代理模型能否以极低的计算开销替代穷举搜索。

    穷举搜索 (Exhaustive Search):
      对每对 (l,d) 遍历所有查询样本, 实测准确率, 计算 J(l,d)
      代价: O(L × |D_l| × N_q × 前向传播) — 卫星上根本无法承受

    ProSTAR 提出方案 (Surrogate Model):
      用拟合好的 Â_l(d) 代入 J(l,d), 仅需 O(L × |D_l|) 次代数运算
      代价: < 5 ms — 卫星上实时决策完全可行

    布局: 2×2 网格
      左上: 穷举 — 准确率热力图
      右上: 代理 — 准确率热力图
      左下: 穷举 — 联合代价热力图
      右下: 代理 — 联合代价热力图
      蓝/红星: 各自的最优决策 (l*, d*)
    """
    print("[Fig.3] 穷举搜索 vs 提出方案对比...")

    supp_ids, supp_labels = dataset['support']
    query_ids, query_labels = dataset['query']
    class_names = dataset['class_names']

    # ---------- 系统参数 (论文 Section V) ----------
    # gamma_l: Swin-Tiny 各 Stage 的累积 FLOPs 比例
    #   Stage1: 15%, Stage2: 35%, Stage3: 65%, Stage4: 100%
    gamma_l = np.array([0.15, 0.35, 0.65, 1.0])
    sys_params = SystemParams(
        f_sat=5.0e9,        # 卫星计算频率 5 GHz (论文原值)
        kappa=1e-28,         # CMOS 电容系数 (论文原值)
        gamma_l=gamma_l,     # 各层 FLOPs 比例
        P_tx=15.0,           # 发射功率 15W
        B=1.0e6,             # 可用带宽 1MHz
        N0=1e-19,            # 噪声功率谱密度
        F_total=4.5e9,       # Swin-Tiny 总 FLOPs ≈ 4.5G
    )
    # 联合代价权重 (论文 Section V)
    #   λ1=200: 准确率项的权重最大, 因为1%的准确率损失≈200的代价单位
    #   λ2=30:  每 ms 延迟 ≈ 30 代价单位
    #   λ3=2:   每 J 能耗 ≈ 2 代价单位
    weights = OptimizationWeights(lambda_acc=200.0, lambda_lat=30.0, lambda_energy=2.0)

    # 为每层生成24个均匀压缩级别
    layer_dims_all = [extractor.layer_dims[l] for l in range(4)]
    d_candidates_list = [np.linspace(2, layer_dims_all[l], 24).astype(int) for l in range(4)]

    # ---------- 初始化结果矩阵 ----------
    N_layers = 4   # L=4 候选分区层
    J = 24         # 每层24个压缩级别

    exhaust_acc = np.zeros((N_layers, J))    # 穷举-实测准确率矩阵
    surr_acc = np.zeros((N_layers, J))       # 代理-预测准确率矩阵
    exhaust_cost = np.zeros((N_layers, J))   # 穷举-联合代价矩阵
    surr_cost = np.zeros((N_layers, J))      # 代理-联合代价矩阵

    # ---------- 逐层遍历所有 (l, d) 组合 ----------
    for l in range(N_layers):
        D_l = layer_dims_all[l]

        # 为当前层准备原型
        supp_feats = extractor.extract_batch(supp_ids, l, noise_level=0.03)
        config = ProSTARConfig(layer_idx=l, seed=42)
        pipeline = ProSTARPipeline(config)
        pipeline.initialize(supp_feats, supp_labels, class_names)

        d_vals = d_candidates_list[l]
        surrogate = surrogate_models[l]          # 该层的代理模型

        for j, d in enumerate(d_vals):
            d_i = int(d)

            # ---- 穷举路径: 遍历所有查询样本实测准确率 ----
            correct = 0
            for i in range(len(query_ids)):
                q_feat = extractor.extract_feature(query_ids[i], l, noise_level=0.05)
                result = pipeline.infer(q_feat, d=d_i)
                if result.pred_class == query_labels[i]:
                    correct += 1
            exhaust_acc[l, j] = correct / len(query_ids) * 100

            # ---- 代理路径: 直接调用拟合好的 sigmoid 函数 (O(1)) ----
            surr_acc[l, j] = surrogate.predict(float(d_i))

            # ---- 系统开销 (通信 + 计算) ----
            lat = sys_params.total_latency(l, d_i)      # Eq.11: T_total
            energy = sys_params.total_energy(l, d_i)    # Eq.12: E_total

            # ---- 联合代价评估 (Eq.13) ----
            exhaust_cost[l, j] = joint_cost(
                l, d_i, 100.0, exhaust_acc[l, j], lat, energy, weights)
            surr_cost[l, j] = joint_cost(
                l, d_i, 100.0, surr_acc[l, j], lat, energy, weights)

    # 找出联合代价最小的 (l, d) 组合 (即最优决策)
    exhaust_best = np.unravel_index(np.argmin(exhaust_cost), exhaust_cost.shape)
    surr_best = np.unravel_index(np.argmin(surr_cost), surr_cost.shape)

    # ---------- 绘制 2×2 热力图 ----------
    acc_cmap = plt.cm.RdYlGn          # 准确率用红-黄-绿渐变 (绿色好)
    cost_cmap = plt.cm.RdYlGn_r       # 代价用绿-黄-红渐变 (红色差, 反序)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    layer_labels = [f"Layer {i}" for i in range(4)]

    # --- (a) 穷举搜索 — 匹配准确率热力图 ---
    im00 = axes[0, 0].imshow(exhaust_acc.T, aspect='auto', origin='lower',
                              cmap=acc_cmap, vmin=0, vmax=100)
    axes[0, 0].set_title('Exhaustive Search — Matching Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Partition Layer $l$')
    axes[0, 0].set_ylabel('Transmission Dimension $d$')
    axes[0, 0].set_xticks(range(N_layers))
    axes[0, 0].set_xticklabels(layer_labels)
    axes[0, 0].set_yticks([0, 5, 10, 15, 20, 23])
    axes[0, 0].set_yticklabels([d_candidates_list[0][k] for k in [0, 5, 10, 15, 20, 23]])
    axes[0, 0].scatter([exhaust_best[1]], [exhaust_best[0]],
                       marker='*', s=200, c='blue', edgecolors='white',
                       linewidth=1.5, zorder=10)
    plt.colorbar(im00, ax=axes[0, 0], label='Accuracy (%)', fraction=0.046)

    # --- (b) 代理模型 — 匹配准确率热力图 ---
    im01 = axes[0, 1].imshow(surr_acc.T, aspect='auto', origin='lower',
                              cmap=acc_cmap, vmin=0, vmax=100)
    axes[0, 1].set_title('Surrogate Model — Matching Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Partition Layer $l$')
    axes[0, 1].set_ylabel('Transmission Dimension $d$')
    axes[0, 1].set_xticks(range(N_layers))
    axes[0, 1].set_xticklabels(layer_labels)
    axes[0, 1].set_yticks([0, 5, 10, 15, 20, 23])
    axes[0, 1].set_yticklabels([d_candidates_list[0][k] for k in [0, 5, 10, 15, 20, 23]])
    axes[0, 1].scatter([surr_best[1]], [surr_best[0]],
                       marker='*', s=200, c='blue', edgecolors='white',
                       linewidth=1.5, zorder=10)
    plt.colorbar(im01, ax=axes[0, 1], label='Accuracy (%)', fraction=0.046)

    # --- (c) 穷举搜索 — 联合代价热力图 ---
    im10 = axes[1, 0].imshow(exhaust_cost.T, aspect='auto', origin='lower',
                              cmap=cost_cmap)
    axes[1, 0].set_title('Exhaustive Search — Joint Cost', fontweight='bold')
    axes[1, 0].set_xlabel('Partition Layer $l$')
    axes[1, 0].set_ylabel('Transmission Dimension $d$')
    axes[1, 0].set_xticks(range(N_layers))
    axes[1, 0].set_xticklabels(layer_labels)
    axes[1, 0].set_yticks([0, 5, 10, 15, 20, 23])
    axes[1, 0].set_yticklabels([d_candidates_list[0][k] for k in [0, 5, 10, 15, 20, 23]])
    axes[1, 0].scatter([exhaust_best[1]], [exhaust_best[0]],
                       marker='*', s=200, c='red', edgecolors='white',
                       linewidth=1.5, zorder=10)
    plt.colorbar(im10, ax=axes[1, 0], label='Joint Cost', fraction=0.046)

    # --- (d) 代理模型 — 联合代价热力图 ---
    im11 = axes[1, 1].imshow(surr_cost.T, aspect='auto', origin='lower',
                              cmap=cost_cmap)
    axes[1, 1].set_title('Surrogate Model — Joint Cost', fontweight='bold')
    axes[1, 1].set_xlabel('Partition Layer $l$')
    axes[1, 1].set_ylabel('Transmission Dimension $d$')
    axes[1, 1].set_xticks(range(N_layers))
    axes[1, 1].set_xticklabels(layer_labels)
    axes[1, 1].set_yticks([0, 5, 10, 15, 20, 23])
    axes[1, 1].set_yticklabels([d_candidates_list[0][k] for k in [0, 5, 10, 15, 20, 23]])
    axes[1, 1].scatter([surr_best[1]], [surr_best[0]],
                       marker='*', s=200, c='red', edgecolors='white',
                       linewidth=1.5, zorder=10)
    plt.colorbar(im11, ax=axes[1, 1], label='Joint Cost', fraction=0.046)

    # 提取最优决策的具体数值
    l_ex, j_ex = exhaust_best
    l_su, j_su = surr_best
    d_ex = d_candidates_list[l_ex][j_ex]    # 穷举最优 d*
    d_su = d_candidates_list[l_su][j_su]    # 代理最优 d*
    cost_mse = np.mean((exhaust_cost - surr_cost) ** 2)  # 联合代价均方误差

    fig.suptitle(
        f'Fig. 3: Exhaustive Search vs Proposed Scheme\n'
        f'Exhaustive: ($l^*$={l_ex}, $d^*$={d_ex})  |  '
        f'Surrogate: ($l^*$={l_su}, $d^*$={d_su})  |  '
        f'Joint Cost MSE = {cost_mse:.4f}',
        fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig('prostar_fig3_exhaustive_vs_proposed.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  -> prostar_fig3_exhaustive_vs_proposed.png")

    return {
        'exhaust_best': (l_ex, d_ex, exhaust_acc[exhaust_best]),
        'surr_best': (l_su, d_su, surr_acc[surr_best]),
        'cost_mse': cost_mse,
    }


# ============================================================
# 第六部分: Fig.4 — 混淆矩阵
# ============================================================
def figure_4_confusion_matrix(extractor, dataset, surrogate_models):
    """
    复现论文 Fig.4: Confusion Matrix of Proposed Scheme.
    ====================================================
    论文最优配置: l*=3, d*=256 (Layer 3, 压缩至256维)

    这个实验的核心意义在于验证:
      "即使所有类别从未在 DNN 训练中出现过,
       仅凭 5-shot 原型 + 256 维压缩特征,
       余弦相似度匹配依然能达到 >90% 的总体准确率。"

    这证明了 ProSTAR 在新类扩展场景下的有效性:
      - 不需要在卫星上做任何反向传播或模型更新
      - 仅传输 256 维浮点向量 (~1KB) 而非完整中间特征 (~150KB)
      - 地面站以 O(M·d) 的极低成本完成匹配

    输出文件: prostar_fig4_confusion_matrix.png
    """
    print("[Fig.4] 混淆矩阵...")

    # 取前16类以保证混淆矩阵的可读性
    num_classes = min(dataset['num_classes'], 16)
    class_names = dataset['class_names'][:num_classes]

    l_opt, d_opt = 3, 256         # 论文最优配置
    D_l = extractor.layer_dims[l_opt]

    # 为前 num_classes 个类重新生成支持集和查询集
    rng = np.random.RandomState(42)
    all_classes = list(range(num_classes))

    # 支持集: 每类 K=5 个样本
    supp_ids = []
    supp_labels = []
    for m, cls_id in enumerate(all_classes):
        for _ in range(5):
            supp_ids.append(cls_id)
            supp_labels.append(m)

    # 查询集: 每类 20 个样本
    query_ids = []
    query_labels = []
    for m, cls_id in enumerate(all_classes):
        for _ in range(20):
            query_ids.append(cls_id)
            query_labels.append(m)

    # 初始化原型库 (Eq.7)
    supp_feats = extractor.extract_batch(np.array(supp_ids), l_opt, noise_level=0.03)
    config = ProSTARConfig(layer_idx=l_opt, seed=42)
    pipeline = ProSTARPipeline(config)
    pipeline.initialize(supp_feats, np.array(supp_labels), class_names)

    # 初始化混淆矩阵: 行=真实类, 列=预测类
    cf = np.zeros((num_classes, num_classes), dtype=np.int32)

    # 逐样本推理
    for i in range(len(query_ids)):
        q_feat = extractor.extract_feature(query_ids[i], l_opt, noise_level=0.05)
        result = pipeline.infer(q_feat, d=d_opt)            # 256维压缩 → 余弦匹配
        cf[query_labels[i], result.pred_class] += 1         # 累加计数

    # ---------- 绘制混淆矩阵 ----------
    fig, ax = plt.subplots(figsize=(8, 7))

    # 按行归一化为百分比 (每行总和=100%)
    cf_normalized = cf.astype(float) / (cf.sum(axis=1, keepdims=True) + 1e-8) * 100

    # 自定义蓝白渐变 (仿论文配色)
    blues = LinearSegmentedColormap.from_list('paper_blues',
                                              ['#F7FBFF', '#4292C6', '#08519C'])

    im = ax.imshow(cf_normalized, cmap=blues, aspect='equal', vmin=0, vmax=100)

    # 在每个单元格中标注百分比数值 (>5% 才显示, 避免视觉杂乱)
    for i in range(num_classes):
        for j in range(num_classes):
            val = cf_normalized[i, j]
            text_color = 'white' if val > 50 else 'black'     # 深色背景用白字
            ax.text(j, i, f'{val:.0f}%' if val > 5 else '',
                    ha='center', va='center', fontsize=7, color=text_color,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('True Class', fontsize=10)
    ax.set_title(f'Confusion Matrix ($l$={l_opt}, $d$={d_opt})', fontsize=12,
                 fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Percentage (%)', fontsize=8)

    # 计算总体准确率 (对角线之和 / 总样本数)
    overall_acc = np.trace(cf) / cf.sum() * 100
    ax.text(0.5, -0.22, f'Overall Accuracy: {overall_acc:.1f}%  |  '
            f'All {num_classes} classes treated as novel (no pre-training)',
            transform=ax.transAxes, ha='center', fontsize=9,
            fontstyle='italic', color='gray')

    fig.suptitle('Fig. 4: Confusion Matrix of Proposed Scheme',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig('prostar_fig4_confusion_matrix.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  -> prostar_fig4_confusion_matrix.png  (OA={overall_acc:.1f}%)")

    return {'overall_accuracy': overall_acc, 'confusion_matrix': cf}


# ============================================================
# 第七部分: Fig.5 — 3D原始特征 vs 1D压缩特征性能对比
# ============================================================
def figure_5_performance_comparison(extractor, dataset, surrogate_models):
    """
    复现论文 Fig.5: Performance comparison.
    ======================================
    这是论文的"性能收益总结"图 — 直观展示 ProSTAR 在三个指标上的增益。

    对比维度:
      左柱 (深红): 保留的3D中间特征 (H×W×C)
        传统 Split Inference: 卫星提取 → 传输完整3D特征 → 地面分类
        这相当于论文 Table I 中的 Traditional Split 方案

      右柱 (深蓝): ProSTAR 压缩的1D特征 (d-vector)
        卫星提取 → 嵌套子空间压缩 → 仅传输 d 个浮点数 → 地面原型匹配

    三组子图 (1行×3列):
      子图1: 匹配准确率 (%) — 压缩后保持多少判别力
      子图2: 总延迟 (ms)    — 传输+计算总时间
      子图3: 总能耗 (J)     — 星上计算+通信总能耗

    论文原值: 总延迟减少 38.4%, 总能耗减少 41.7%

    输出文件: prostar_fig5_performance_comparison.png
    """
    print("[Fig.5] 3D特征 vs 1D压缩特征性能对比...")

    # 最优配置 (论文 Fig.4 和 Fig.5 均使用同一组参数)
    l_opt, d_opt = 3, 256
    D_l = extractor.layer_dims[l_opt]              # 768

    # ---------- 实测准确率 ----------
    supp_ids, supp_labels = dataset['support']
    query_ids, query_labels = dataset['query']
    class_names = dataset['class_names']

    # 初始化原型 (公用)
    supp_feats = extractor.extract_batch(supp_ids, l_opt, noise_level=0.03)
    config = ProSTARConfig(layer_idx=l_opt, seed=42)
    pipeline = ProSTARPipeline(config)
    pipeline.initialize(supp_feats, supp_labels, class_names)

    # 3D 特征准确率: d = D_l (完整维度, 无压缩)
    correct_3d = 0
    for i in range(len(query_ids)):
        q_feat = extractor.extract_feature(query_ids[i], l_opt, noise_level=0.05)
        result = pipeline.infer(q_feat, d=D_l)          # 使用全部768维
        if result.pred_class == query_labels[i]:
            correct_3d += 1
    acc_3d = correct_3d / len(query_ids) * 100

    # 1D 特征准确率: d = 256 (ProSTAR 压缩)
    correct_1d = 0
    for i in range(len(query_ids)):
        q_feat = extractor.extract_feature(query_ids[i], l_opt, noise_level=0.05)
        result = pipeline.infer(q_feat, d=d_opt)        # 压缩至256维
        if result.pred_class == query_labels[i]:
            correct_1d += 1
    acc_1d = correct_1d / len(query_ids) * 100

    # ---------- 系统开销 (Eq.11 & Eq.12) ----------
    gamma_l = np.array([0.15, 0.35, 0.65, 1.0])
    sys_params = SystemParams(
        f_sat=5.0e9, kappa=1e-28, gamma_l=gamma_l,
        P_tx=15.0, B=1.0e6, N0=1e-19, F_total=4.5e9,
    )

    # 3D 特征的等效传输维度
    # Swin-Tiny Stage4 的输出空间尺寸是 7×7, 通道数768
    # 不做 GAP + 展开的话, 传输量 ≈ 7*7*768 = 37632 个 float32 = ~150KB
    hw = {0: 56*56, 1: 28*28, 2: 14*14, 3: 7*7}
    d_3d_equivalent = hw[l_opt] * D_l              # 49 × 768 = 37632

    lat_3d = sys_params.total_latency(l_opt, d_3d_equivalent) * 1000    # 转换为 ms
    energy_3d = sys_params.total_energy(l_opt, d_3d_equivalent)

    lat_1d = sys_params.total_latency(l_opt, d_opt) * 1000
    energy_1d = sys_params.total_energy(l_opt, d_opt)

    # 计算降幅百分比
    lat_reduction = (1 - lat_1d / lat_3d) * 100
    energy_reduction = (1 - energy_1d / energy_3d) * 100

    # ---------- 绘制 1×3 红蓝分组柱状图 ----------
    categories = ['Matching\nAccuracy (%)',
                  'Total Latency\n(ms)',
                  'Total Energy\nConsumption (J)']
    values_3d = [acc_3d, lat_3d, energy_3d]         # 深红柱数据
    values_1d = [acc_1d, lat_1d, energy_1d]         # 深蓝柱数据

    colors_3d = '#B2182B'    # 深红 — 传统3D方案
    colors_1d = '#2166AC'    # 深蓝 — ProSTAR方案

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x_pos = np.array([0, 1])                        # 两柱的x坐标

    for idx, (ax, cat, v3d, v1d) in enumerate(zip(axes, categories, values_3d, values_1d)):
        bar_width = 0.35
        # 左柱: 3D特征 (红色)
        bars_3d = ax.bar(x_pos[0], v3d, bar_width, color=colors_3d, alpha=0.85,
                         edgecolor='white', linewidth=1.2, label='Retained 3D Features')
        # 右柱: 1D特征 (蓝色)
        bars_1d = ax.bar(x_pos[1], v1d, bar_width, color=colors_1d, alpha=0.85,
                         edgecolor='white', linewidth=1.2, label='Proposed 1D Features (ProSTAR)')

        # 在每个柱顶标注数值
        for bar in [bars_3d, bars_1d]:
            for rect in bar:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Retained\n3D Features', 'Proposed\n1D Features\n(ProSTAR)'],
                           fontsize=8)
        ax.set_title(cat.replace('\n', ' '), fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # 在延迟和能耗子图上标注降幅百分比
    axes[1].annotate(f'↓ {lat_reduction:.1f}%',
                     xy=(0.5, 0.92), xycoords='axes fraction',
                     fontsize=12, fontweight='bold', color='#D73027',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#D73027', alpha=0.9))
    axes[2].annotate(f'↓ {energy_reduction:.1f}%',
                     xy=(0.5, 0.92), xycoords='axes fraction',
                     fontsize=12, fontweight='bold', color='#D73027',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='#D73027', alpha=0.9))

    # 底部共享图例
    fig.legend([bars_3d[0], bars_1d[0]],
               ['Retained 3D Features', 'Proposed 1D Features (ProSTAR)'],
               loc='lower center', ncol=2, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(
        f'Fig. 5: Performance Comparison (Layer {l_opt}, '
        f'3D ${hw[l_opt]}^{{1/2}}\\times{hw[l_opt]}^{{1/2}}\\times{D_l}$ vs '
        f'Compressed $d={d_opt}$)',
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('prostar_fig5_performance_comparison.png', dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  -> prostar_fig5_performance_comparison.png  "
          f"(Lat↓{lat_reduction:.1f}%, Energy↓{energy_reduction:.1f}%)")

    return {
        'acc_3d': acc_3d, 'acc_1d': acc_1d,
        'lat_3d_ms': lat_3d, 'lat_1d_ms': lat_1d,
        'energy_3d_j': energy_3d, 'energy_1d_j': energy_1d,
        'lat_reduction': lat_reduction, 'energy_reduction': energy_reduction,
    }


# ============================================================
# 第八部分: 程序入口 — 串联所有实验
# ============================================================
def main():
    """运行全部4个实验并生成对应图表"""

    print("╔" + "═" * 62 + "╗")
    print("║  ProSTAR Paper Figure Reproduction                    ║")
    print("║  Fig.2: Surrogate Model Fitting                       ║")
    print("║  Fig.3: Exhaustive Search vs Proposed Scheme          ║")
    print("║  Fig.4: Confusion Matrix                              ║")
    print("║  Fig.5: 3D Features vs 1D Compressed Features         ║")
    print("╚" + "═" * 62 + "╝")

    # 固定全局随机种子 — 保证每次运行结果完全可复现
    np.random.seed(42)

    # 初始化虚拟特征提取器 (模拟 Swin-Tiny + ImageNet 预训练)
    extractor = SimulatedFeatureExtractor(num_classes=21, seed=42)

    # 生成 21-way 5-shot 数据集 (模拟 UCMerced)
    dataset = generate_ucmerced_like_dataset(
        num_classes=21, k_shot=5, num_query=30, seed=42)

    print(f"  数据集: {dataset['num_classes']}-way 5-shot, "
          f"{len(dataset['query'][0])} query samples")

    # ---- 依次运行4个实验 ----
    # Fig.2: 代理模型拟合 (产出 surrogate_models 供后续实验使用)
    surrogate_models, empirical_data = figure_2_surrogate_fitting(extractor, dataset)

    # Fig.3: 穷举 vs 代理对比 (验证代理模型的替代有效性)
    fig3_result = figure_3_exhaustive_vs_proposed(extractor, dataset, surrogate_models)

    # Fig.4: 混淆矩阵 (验证压缩1D特征的判别力)
    fig4_result = figure_4_confusion_matrix(extractor, dataset, surrogate_models)

    # Fig.5: 性能对比柱状图 (量化延迟和能耗收益)
    fig5_result = figure_5_performance_comparison(extractor, dataset, surrogate_models)

    # ---- 结果汇总 ---- (Note: Chinese characters appear garbled in some terminals due to GBK)
    print("\n" + "=" * 62)
    print("  All figures generated successfully!")
    print("=" * 62)
    print(f"  Fig.2: prostar_fig2_surrogate_fitting.png")
    print(f"  Fig.3: prostar_fig3_exhaustive_vs_proposed.png")
    print(f"         Exhaustive best: L={fig3_result['exhaust_best'][0]}, "
          f"d={fig3_result['exhaust_best'][1]}")
    print(f"         Surrogate best:  L={fig3_result['surr_best'][0]}, "
          f"d={fig3_result['surr_best'][1]}")
    print(f"         Joint Cost MSE:  {fig3_result['cost_mse']:.4f}")
    print(f"  Fig.4: prostar_fig4_confusion_matrix.png")
    print(f"         Overall Accuracy: {fig4_result['overall_accuracy']:.1f}%")
    print(f"  Fig.5: prostar_fig5_performance_comparison.png")
    print(f"         Latency reduction:  {fig5_result['lat_reduction']:.1f}%")
    print(f"         Energy reduction:   {fig5_result['energy_reduction']:.1f}%")
    print(f"         3D Accuracy: {fig5_result['acc_3d']:.1f}%  "
          f"-> 1D Accuracy: {fig5_result['acc_1d']:.1f}%")
    print("=" * 62)


# 当直接运行 python -m prostar.demo 或 python demo.py 时, 调用 main()
if __name__ == '__main__':
    main()
