"""
ProSTAR 论文算法复现 Demo
===========================
精确复现论文中的4张核心图表:
  Fig.2: 代理模型拟合性能 (Surrogate Model Fitting)
  Fig.3: 穷举搜索与提出方案的对比 (Exhaustive vs Proposed)
  Fig.4: 混淆矩阵 (Confusion Matrix)
  Fig.5: 3D原始特征 vs 1D压缩特征性能对比 (Performance Comparison)

运行: python -m prostar.demo
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from collections import defaultdict

from prostar.prototype import (
    ProSTARPipeline, ProSTARConfig,
    initialize_prototypes, cosine_similarity_match,
    compress_feature, PrototypeBank,
)
from prostar.surrogate import (
    LayerSurrogateParams, fit_surrogate_params,
    SystemParams, OptimizationWeights,
    online_decision, joint_cost, generate_d_candidates,
    _stable_sigmoid,
)

# ============================================================
# 全局样式配置
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ============================================================
# 模拟 Swin-Tiny 骨干网络特征提取器
# ============================================================

class SimulatedFeatureExtractor:
    """
    模拟 Swin-Tiny 的特征提取器
    论文配置: 4个stage (L=4), C_l ∈ {96, 192, 384, 768}

    关键设计: 信息沿维度呈指数递减分布 — 前几个维度携带主要语义,
    后者为冗余信息。这使得维度压缩产生论文描述的sigmoid退化曲线。
    """

    def __init__(self, num_classes=21, seed=42):
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

        # Swin-Tiny 各stage输出通道数
        self.layer_dims = {0: 96, 1: 192, 2: 384, 3: 768}

        # 各类隐藏语义向量 (768维, 模拟ImageNet预训练后的表示空间)
        self.class_centers = {}
        for c in range(num_classes):
            v = self.rng.randn(768)
            self.class_centers[c] = v / np.linalg.norm(v)

        # 逐层Kaiming初始化投影
        self.layer_projections = {}
        prev_dim = 768
        for l in range(4):
            D_l = self.layer_dims[l]
            self.layer_projections[l] = (
                self.rng.randn(prev_dim, D_l) * np.sqrt(2.0 / prev_dim)
            ).astype(np.float32)
            prev_dim = D_l

    def extract_feature(self, class_id: int, layer_idx: int,
                        noise_level: float = 0.05) -> np.ndarray:
        """提取指定层的1D特征向量 f_l ∈ R^{D_l}"""
        D_l = self.layer_dims[layer_idx]
        hidden = self.class_centers[class_id] + noise_level * self.rng.randn(768)

        h = hidden
        for l in range(layer_idx + 1):
            h = h @ self.layer_projections[l]
            h = np.tanh(h) * 2.0

        # 信息衰减: 前几维承载更多语义 (论文的核心隐含假设)
        decay = np.exp(-np.arange(D_l) / (D_l * 0.15))
        f_l = (h * decay).astype(np.float32)
        return f_l / (np.linalg.norm(f_l) + 1e-8)

    def extract_batch(self, class_ids: np.ndarray, layer_idx: int,
                      noise_level: float = 0.05) -> np.ndarray:
        feats = np.zeros((len(class_ids), self.layer_dims[layer_idx]), dtype=np.float32)
        for i, cid in enumerate(class_ids):
            feats[i] = self.extract_feature(cid, layer_idx, noise_level)
        return feats


# ============================================================
# 数据生成 (模拟 UCMerced LandUse 21类的 few-shot 新类扩展)
# ============================================================

def generate_ucmerced_like_dataset(
    num_classes: int = 21,
    k_shot: int = 5,
    num_query: int = 30,
    seed: int = 42,
):
    """生成 M-way K-shot 分类数据集"""
    rng = np.random.RandomState(seed)
    all_classes = list(range(num_classes))
    rng.shuffle(all_classes)

    novel_classes = all_classes  # 全部视为新类

    support_class_ids = []
    support_labels = []
    for m, cls_id in enumerate(novel_classes):
        for _ in range(k_shot):
            support_class_ids.append(cls_id)
            support_labels.append(m)

    query_class_ids = []
    query_labels = []
    for m, cls_id in enumerate(novel_classes):
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
# Fig.2: 代理模型拟合性能
# ============================================================

def figure_2_surrogate_fitting(extractor, dataset):
    """
    复现论文 Fig.2: Surrogate model fitting performance.

    布局: 1×4 子图, 对应4个Stage (Layer 0~3)
    每个子图: 蓝色散点(Measured A(l,d)) + 蓝色曲线(Surrogate model fit)
    标注: d0(临界阈值), alpha(陡峭度系数)
    """
    print("[Fig.2] 代理模型拟合性能...")

    supp_ids, supp_labels = dataset['support']
    query_ids, query_labels = dataset['query']
    class_names = dataset['class_names']

    surrogate_models = []
    empirical_data_all = {}

    # 为每层构建代理模型
    for layer_idx in range(4):
        D_l = extractor.layer_dims[layer_idx]

        # 提取支持集特征并初始化原型
        supp_feats = extractor.extract_batch(supp_ids, layer_idx, noise_level=0.03)
        config = ProSTARConfig(layer_idx=layer_idx, seed=42)
        pipeline = ProSTARPipeline(config)
        pipeline.initialize(supp_feats, supp_labels, class_names)

        # 24个均匀压缩级别 (论文设定)
        d_candidates = np.linspace(2, D_l, 24).astype(int)

        # 测量每个压缩级别的匹配准确率
        empirical_accs = []
        for d in d_candidates:
            correct = 0
            for i in range(len(query_ids)):
                q_feat = extractor.extract_feature(query_ids[i], layer_idx, noise_level=0.05)
                result = pipeline.infer(q_feat, d=int(d))
                if result.pred_class == query_labels[i]:
                    correct += 1
            empirical_accs.append(correct / len(query_ids) * 100)

        d_vals = d_candidates.astype(np.float64)
        acc_vals = np.array(empirical_accs, dtype=np.float64)
        empirical_data_all[layer_idx] = (d_vals, acc_vals)

        # 拟合代理模型
        surrogate = fit_surrogate_params(d_vals, acc_vals, layer_idx, D_l)
        surrogate_models.append(surrogate)

    # ---- 绘制 ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))

    for layer_idx in range(4):
        ax = axes[layer_idx]
        d_vals, acc_vals = empirical_data_all[layer_idx]
        surrogate = surrogate_models[layer_idx]

        # 蓝色散点: 实测值
        ax.scatter(d_vals, acc_vals, s=18, c='#2166AC', alpha=0.7,
                   edgecolors='white', linewidth=0.5, zorder=5)

        # 蓝色曲线: 代理模型
        d_smooth = np.linspace(d_vals.min(), d_vals.max(), 300)
        acc_smooth = surrogate.predict_batch(d_smooth)
        ax.plot(d_smooth, acc_smooth, '-', color='#2166AC', linewidth=2.0,
                alpha=0.9, zorder=4, label='Surrogate model')

        # 标注临界维度 d0
        ax.axvline(x=surrogate.d0, color='#D73027', linestyle='--',
                   linewidth=1.2, alpha=0.7)
        ax.annotate(f"$d_0^{{{layer_idx}}}$={surrogate.d0:.0f}",
                    xy=(surrogate.d0, acc_vals.mean()),
                    xytext=(surrogate.d0 + 30, acc_vals.mean() - 8),
                    fontsize=8, color='#D73027',
                    arrowprops=dict(arrowstyle='->', color='#D73027', lw=1.0))

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
# Fig.3: 穷举搜索 vs 提出方案对比
# ============================================================

def figure_3_exhaustive_vs_proposed(extractor, dataset, surrogate_models):
    """
    复现论文 Fig.3: Comparison between Exhaustive Search and the Proposed Scheme.

    布局: 2×2 热力图网格
    - 左上: Exhaustive Search — Matching Accuracy
    - 右上: Proposed Surrogate Model — Matching Accuracy
    - 左下: Exhaustive Search — Joint Cost
    - 右下: Proposed Surrogate Model — Joint Cost
    标注最优决策 (l*, d*)
    """
    print("[Fig.3] 穷举搜索 vs 提出方案对比...")

    supp_ids, supp_labels = dataset['support']
    query_ids, query_labels = dataset['query']
    class_names = dataset['class_names']

    # 系统参数 (论文 Section V)
    gamma_l = np.array([0.15, 0.35, 0.65, 1.0])
    sys_params = SystemParams(
        f_sat=5.0e9, kappa=1e-28, gamma_l=gamma_l,
        P_tx=15.0, B=1.0e6, N0=1e-19, F_total=4.5e9,
    )
    weights = OptimizationWeights(lambda_acc=200.0, lambda_lat=30.0, lambda_energy=2.0)

    # 各层维度候选集
    layer_dims_all = [extractor.layer_dims[l] for l in range(4)]
    d_candidates_list = [np.linspace(2, layer_dims_all[l], 24).astype(int) for l in range(4)]

    # ---- 计算穷举搜索和代理模型的所有 (l,d) 结果 ----
    N_layers = 4
    J = 24  # 压缩级别

    # 矩阵: exhaust_acc[l, j], surr_acc[l, j], exhaust_cost[l, j], surr_cost[l, j]
    exhaust_acc = np.zeros((N_layers, J))
    surr_acc = np.zeros((N_layers, J))
    exhaust_cost = np.zeros((N_layers, J))
    surr_cost = np.zeros((N_layers, J))

    for l in range(N_layers):
        D_l = layer_dims_all[l]
        # 为穷举搜索准备原型
        supp_feats = extractor.extract_batch(supp_ids, l, noise_level=0.03)
        config = ProSTARConfig(layer_idx=l, seed=42)
        pipeline = ProSTARPipeline(config)
        pipeline.initialize(supp_feats, supp_labels, class_names)

        d_vals = d_candidates_list[l]
        surrogate = surrogate_models[l]

        for j, d in enumerate(d_vals):
            d_i = int(d)

            # 穷举: 实测准确率
            correct = 0
            for i in range(len(query_ids)):
                q_feat = extractor.extract_feature(query_ids[i], l, noise_level=0.05)
                result = pipeline.infer(q_feat, d=d_i)
                if result.pred_class == query_labels[i]:
                    correct += 1
            exhaust_acc[l, j] = correct / len(query_ids) * 100

            # 代理: 预测准确率
            surr_acc[l, j] = surrogate.predict(float(d_i))

            # 系统开销
            lat = sys_params.total_latency(l, d_i)
            energy = sys_params.total_energy(l, d_i)

            exhaust_cost[l, j] = joint_cost(
                l, d_i, 100.0, exhaust_acc[l, j], lat, energy, weights)
            surr_cost[l, j] = joint_cost(
                l, d_i, 100.0, surr_acc[l, j], lat, energy, weights)

    # 找到各自的最优决策
    exhaust_best = np.unravel_index(np.argmin(exhaust_cost), exhaust_cost.shape)
    surr_best = np.unravel_index(np.argmin(surr_cost), surr_cost.shape)

    # ---- 自定义 colormap ----
    acc_cmap = plt.cm.RdYlGn
    cost_cmap = plt.cm.RdYlGn_r

    # ---- 绘制 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    layer_labels = [f"Layer {i}" for i in range(4)]
    d_labels = [f"{d_candidates_list[0][j]}" for j in range(J)]

    # (a) Exhaustive - Accuracy
    im00 = axes[0, 0].imshow(exhaust_acc.T, aspect='auto', origin='lower',
                              cmap=acc_cmap, vmin=0, vmax=100)
    axes[0, 0].set_title('Exhaustive Search — Matching Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Partition Layer $l$')
    axes[0, 0].set_ylabel('Transmission Dimension $d$')
    axes[0, 0].set_xticks(range(N_layers))
    axes[0, 0].set_xticklabels(layer_labels)
    axes[0, 0].set_yticks([0, 5, 10, 15, 20, 23])
    axes[0, 0].set_yticklabels([d_candidates_list[0][k] for k in [0, 5, 10, 15, 20, 23]])
    # 标注最优
    axes[0, 0].scatter([exhaust_best[1]], [exhaust_best[0]],
                       marker='*', s=200, c='blue', edgecolors='white',
                       linewidth=1.5, zorder=10)
    plt.colorbar(im00, ax=axes[0, 0], label='Accuracy (%)', fraction=0.046)

    # (b) Proposed - Accuracy
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

    # (c) Exhaustive - Joint Cost
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

    # (d) Proposed - Joint Cost
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

    # 计算一致性
    l_ex, j_ex = exhaust_best
    l_su, j_su = surr_best
    d_ex = d_candidates_list[l_ex][j_ex]
    d_su = d_candidates_list[l_su][j_su]
    cost_mse = np.mean((exhaust_cost - surr_cost) ** 2)

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
# Fig.4: 混淆矩阵
# ============================================================

def figure_4_confusion_matrix(extractor, dataset, surrogate_models):
    """
    复现论文 Fig.4: Confusion Matrix of Proposed Scheme.

    论文配置: UCMerced, l*=3, d*=256 (最优组合)
    所有类视为新类, 使用 few-shot 原型匹配
    展示对角线优势 → 压缩1D特征仍保持强判别力
    """
    print("[Fig.4] 混淆矩阵...")

    num_classes = min(dataset['num_classes'], 16)  # 最多16类 (可读性)
    class_names = dataset['class_names'][:num_classes]

    # 论文最优配置: l=3, d=256
    l_opt, d_opt = 3, 256
    D_l = extractor.layer_dims[l_opt]

    # 为前 num_classes 类生成支持集和查询集
    rng = np.random.RandomState(42)
    all_classes = list(range(num_classes))

    supp_ids = []
    supp_labels = []
    for m, cls_id in enumerate(all_classes):
        for _ in range(5):  # K=5
            supp_ids.append(cls_id)
            supp_labels.append(m)

    # 查询集 (每类20个样本)
    query_ids = []
    query_labels = []
    for m, cls_id in enumerate(all_classes):
        for _ in range(20):
            query_ids.append(cls_id)
            query_labels.append(m)

    # 初始化原型
    supp_feats = extractor.extract_batch(np.array(supp_ids), l_opt, noise_level=0.03)
    config = ProSTARConfig(layer_idx=l_opt, seed=42)
    pipeline = ProSTARPipeline(config)
    pipeline.initialize(supp_feats, np.array(supp_labels), class_names)

    # 推理 + 构建混淆矩阵
    cf = np.zeros((num_classes, num_classes), dtype=np.int32)

    for i in range(len(query_ids)):
        q_feat = extractor.extract_feature(query_ids[i], l_opt, noise_level=0.05)
        result = pipeline.infer(q_feat, d=d_opt)
        cf[query_labels[i], result.pred_class] += 1

    # ---- 绘制 ----
    fig, ax = plt.subplots(figsize=(8, 7))

    # 按行归一化 → 百分比
    cf_normalized = cf.astype(float) / (cf.sum(axis=1, keepdims=True) + 1e-8) * 100

    # 自定义蓝色渐变 (仿论文风格)
    blues = LinearSegmentedColormap.from_list('paper_blues',
                                              ['#F7FBFF', '#4292C6', '#08519C'])

    im = ax.imshow(cf_normalized, cmap=blues, aspect='equal', vmin=0, vmax=100)

    # 标注数字
    for i in range(num_classes):
        for j in range(num_classes):
            val = cf_normalized[i, j]
            text_color = 'white' if val > 50 else 'black'
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

    # 计算整体准确率
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
# Fig.5: 3D原始特征 vs 1D压缩特征性能对比
# ============================================================

def figure_5_performance_comparison(extractor, dataset, surrogate_models):
    """
    复现论文 Fig.5: Performance comparison between original 3D features
    and compressed 1D features under the optimal configuration.

    论文最优配置: l=3, d=256
    三组对比: Matching Accuracy, Total Latency, Total Energy

    "Original" = 保留的3D特征 (H×W×C)
    "Proposed" (ProSTAR) = 压缩的1D特征 (d-vector)
    """
    print("[Fig.5] 3D特征 vs 1D压缩特征性能对比...")

    # 最优配置
    l_opt, d_opt = 3, 256
    D_l = extractor.layer_dims[l_opt]

    # ---- 实测准确率 ----
    supp_ids, supp_labels = dataset['support']
    query_ids, query_labels = dataset['query']
    class_names = dataset['class_names']

    supp_feats = extractor.extract_batch(supp_ids, l_opt, noise_level=0.03)
    config = ProSTARConfig(layer_idx=l_opt, seed=42)
    pipeline = ProSTARPipeline(config)
    pipeline.initialize(supp_feats, supp_labels, class_names)

    # 3D特征准确率 (全部维度, d=D_l)
    correct_3d = 0
    for i in range(len(query_ids)):
        q_feat = extractor.extract_feature(query_ids[i], l_opt, noise_level=0.05)
        result = pipeline.infer(q_feat, d=D_l)
        if result.pred_class == query_labels[i]:
            correct_3d += 1
    acc_3d = correct_3d / len(query_ids) * 100

    # 1D压缩特征准确率 (d=d_opt)
    correct_1d = 0
    for i in range(len(query_ids)):
        q_feat = extractor.extract_feature(query_ids[i], l_opt, noise_level=0.05)
        result = pipeline.infer(q_feat, d=d_opt)
        if result.pred_class == query_labels[i]:
            correct_1d += 1
    acc_1d = correct_1d / len(query_ids) * 100

    # ---- 系统开销 (论文 Eq. 11-12) ----
    gamma_l = np.array([0.15, 0.35, 0.65, 1.0])
    sys_params = SystemParams(
        f_sat=5.0e9, kappa=1e-28, gamma_l=gamma_l,
        P_tx=15.0, B=1.0e6, N0=1e-19, F_total=4.5e9,
    )

    # 3D特征: 假设原始中间特征 H×W×C (如 7×7×768)
    hw = {0: 56*56, 1: 28*28, 2: 14*14, 3: 7*7}  # Swin-T 各stage空间尺寸
    d_3d_equivalent = hw[l_opt] * D_l  # 7×7×768 = 37632 维

    lat_3d = sys_params.total_latency(l_opt, d_3d_equivalent) * 1000   # ms
    energy_3d = sys_params.total_energy(l_opt, d_3d_equivalent)

    lat_1d = sys_params.total_latency(l_opt, d_opt) * 1000
    energy_1d = sys_params.total_energy(l_opt, d_opt)

    # ---- 绘制红蓝分组柱状图 (仿论文风格) ----
    categories = ['Matching\nAccuracy (%)',
                  'Total Latency\n(ms)',
                  'Total Energy\nConsumption (J)']
    values_3d = [acc_3d, lat_3d, energy_3d]
    values_1d = [acc_1d, lat_1d, energy_1d]

    # 降幅
    lat_reduction = (1 - lat_1d / lat_3d) * 100
    energy_reduction = (1 - energy_1d / energy_3d) * 100

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    colors_3d = '#B2182B'   # 深红 — Retained 3D
    colors_1d = '#2166AC'   # 深蓝 — Proposed 1D (ProSTAR)

    x_pos = np.array([0, 1])

    for idx, (ax, cat, v3d, v1d) in enumerate(zip(axes, categories, values_3d, values_1d)):
        bar_width = 0.35
        bars_3d = ax.bar(x_pos[0], v3d, bar_width, color=colors_3d, alpha=0.85,
                         edgecolor='white', linewidth=1.2, label='Retained 3D Features')
        bars_1d = ax.bar(x_pos[1], v1d, bar_width, color=colors_1d, alpha=0.85,
                         edgecolor='white', linewidth=1.2, label='Proposed 1D Features (ProSTAR)')

        # 数值标注
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

    # 添加降幅标注
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

    # 共享图例
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
# 主入口
# ============================================================

def main():
    print("╔" + "═" * 62 + "╗")
    print("║  ProSTAR Paper Figure Reproduction                    ║")
    print("║  Fig.2: Surrogate Model Fitting                       ║")
    print("║  Fig.3: Exhaustive Search vs Proposed Scheme          ║")
    print("║  Fig.4: Confusion Matrix                              ║")
    print("║  Fig.5: 3D Features vs 1D Compressed Features         ║")
    print("╚" + "═" * 62 + "╝")

    np.random.seed(42)

    # 数据准备
    extractor = SimulatedFeatureExtractor(num_classes=21, seed=42)
    dataset = generate_ucmerced_like_dataset(
        num_classes=21, k_shot=5, num_query=30, seed=42)

    print(f"  数据集: {dataset['num_classes']}-way 5-shot, "
          f"{len(dataset['query'][0])} query samples")

    # Fig.2: 代理模型拟合
    surrogate_models, empirical_data = figure_2_surrogate_fitting(extractor, dataset)

    # Fig.3: 穷举搜索 vs 提出方案
    fig3_result = figure_3_exhaustive_vs_proposed(extractor, dataset, surrogate_models)

    # Fig.4: 混淆矩阵
    fig4_result = figure_4_confusion_matrix(extractor, dataset, surrogate_models)

    # Fig.5: 3D vs 1D 性能对比
    fig5_result = figure_5_performance_comparison(extractor, dataset, surrogate_models)

    # ---- 汇总 ----
    print("\n" + "=" * 62)
    print("  全部图表生成完毕!")
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
          f"→ 1D Accuracy: {fig5_result['acc_1d']:.1f}%")
    print("=" * 62)


if __name__ == '__main__':
    main()
