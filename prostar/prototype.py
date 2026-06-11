"""
ProSTAR 核心算法: 原型构建、特征压缩、原型匹配与自适应更新
基于论文: "ProSTAR: Prototype-Based Satellite-Terrestrial Adaptation
          for Resource Constrained Co-Inference"

核心模块包含:
1. 嵌套子空间原型初始化 (Eq. 7)
2. 特征提取与嵌套子空间压缩 (Eq. 8)
3. 基于距离的原型匹配 (Eq. 9) — 余弦相似度
4. 自适应原型更新 (Eq. 10) — EMA
"""

import math
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ============================================================
# 数据结构
# ============================================================

@dataclass
class PrototypeBank:
    """原型仓库: 存储各类的 base prototype (D_l 维)
    论文核心思想: 构建嵌套子空间，按需截取前 d 维
    """
    base_prototypes: np.ndarray           # (M, D_l) — M个类别的D_l维base prototype
    class_names: List[str] = field(default_factory=list)  # M个类名
    layer_idx: int = 0                    # 分区层索引 l
    base_dim: int = 0                     # D_l
    class_counts: np.ndarray = None       # (M,) 各类支持样本数

    def get_prototype(self, d: int) -> np.ndarray:
        """O(1)复杂度: 直接从base prototype截取前d维
        返回: (M, d) — M类 d维原型矩阵
        """
        d = min(d, self.base_dim)
        return self.base_prototypes[:, :d]

    def update_base_dim(self, first_d: int, new_values: np.ndarray):
        """更新 base prototype 的前 d 维 (EMA)
        Args:
            first_d: 要更新的维度数
            new_values: (M, d) 新值
        """
        self.base_prototypes[:, :first_d] = new_values


# ============================================================
# 1. 嵌套子空间原型初始化 (论文 Section III, Stage 1, Eq. 7)
# ============================================================

def _generate_random_orthogonal_matrix(dim: int, seed: int = 0) -> np.ndarray:
    """生成随机正交投影矩阵 (QR分解)
    P_base ∈ R^{D_l × D_l}, P^T P = I
    论文使用 random projection，此处用 QR 分解保证正交性
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(dim, dim)
    Q, R = np.linalg.qr(A)
    return Q.astype(np.float32)


def initialize_prototypes(
    features: np.ndarray,          # (N, D_l) GAP后的1D特征
    labels: np.ndarray,            # (N,) 类别标签 (0 ~ M-1)
    class_names: List[str],
    layer_idx: int = 0,
    seed: int = 42,
) -> Tuple[PrototypeBank, np.ndarray]:
    """
    原型初始化 + 嵌套子空间构建 (Eq. 7)

    Algorithm (论文 Stage 1):
    1. 计算各类均值 c_base_m = mean(GAP(F_ext(x_i))) for x_i in class m
    2. 生成 D_l × D_l 随机正交投影矩阵 P_base
    3. 在基投影矩阵下计算原型: c_base_m = c_base_m · P_base
    4. 存储 base prototypes，后续按需截取前 d 维

    Args:
        features: (N, D_l) 支持集样本特征
        labels: (N,) 标签
        class_names: 类名列表
        layer_idx: 分区层索引
        seed: 随机种子

    Returns:
        bank: PrototypeBank (base prototypes)
        P_base: (D_l, D_l) 基投影矩阵
    """
    N, D_l = features.shape
    M = len(class_names)

    # 生成基投影矩阵 (D_l × D_l)
    P_base = _generate_random_orthogonal_matrix(D_l, seed=seed)

    # 对特征施加基投影
    features_proj = features @ P_base  # (N, D_l)

    # 计算各类原型: c_base_m = mean(features_proj[class_m])  (Eq. 7)
    base_prototypes = np.zeros((M, D_l), dtype=np.float32)
    class_counts = np.zeros(M, dtype=np.int32)

    for m in range(M):
        mask = (labels == m)
        class_counts[m] = mask.sum()
        if class_counts[m] > 0:
            base_prototypes[m] = features_proj[mask].mean(axis=0)

    # L2 归一化 (便于后续余弦相似度计算)
    norms = np.linalg.norm(base_prototypes, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    base_prototypes /= norms

    bank = PrototypeBank(
        base_prototypes=base_prototypes,
        class_names=class_names,
        layer_idx=layer_idx,
        base_dim=D_l,
        class_counts=class_counts,
    )

    return bank, P_base


# ============================================================
# 2. 特征压缩 (论文 Section III, Stage 2, Eq. 8)
# ============================================================

def compress_feature(
    feature: np.ndarray,   # (D_l,) 1D特征向量
    P_base: np.ndarray,    # (D_l, D_l) 基投影矩阵
    d: int,                # 目标维度
) -> np.ndarray:
    """
    嵌套子空间特征压缩 (Eq. 8)

    数学表达: z_new = f_l · P^l_d ∈ R^d
    其中 P^l_d 是 P^l_base 的前 d 列

    计算复杂度: O(D_l × d)
    相比 DNN 前向传播的计算量可忽略

    Args:
        feature: (D_l,) 原始1D特征
        P_base: (D_l, D_l) 基投影矩阵
        d: 目标传输维度

    Returns:
        z_new: (d,) 压缩后的特征向量
    """
    d = min(d, feature.shape[0])
    P_d = P_base[:, :d]       # (D_l, d)
    z_new = feature @ P_d     # (d,)  (Eq. 8)
    return z_new


def compress_features_batch(
    features: np.ndarray,   # (N, D_l)
    P_base: np.ndarray,     # (D_l, D_l)
    d: int,                 # 目标维度
) -> np.ndarray:
    """批量压缩"""
    d = min(d, features.shape[1])
    P_d = P_base[:, :d]       # (D_l, d)
    return features @ P_d     # (N, d)


# ============================================================
# 3. 基于距离的原型匹配 (论文 Section III, Stage 3, Eq. 9)
# ============================================================

def cosine_similarity_match(
    z_new: np.ndarray,        # (d,) 压缩特征
    prototypes: np.ndarray,    # (M, d) d维原型矩阵
) -> Tuple[int, np.ndarray]:
    """
    余弦相似度原型匹配 (Eq. 9)

    数学表达:
    ŷ = argmax_m (z_new · c_m(d)^⊤) / (||z_new||₂ · ||c_m(d)||₂)

    Args:
        z_new: (d,) 查询特征
        prototypes: (M, d) 原型矩阵

    Returns:
        pred_idx: 预测类别索引
        similarities: (M,) 各类余弦相似度
    """
    z_norm = z_new / (np.linalg.norm(z_new) + 1e-8)
    proto_norms = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

    # 余弦相似度: z·c^⊤ / (||z||·||c||)  (Eq. 9)
    similarities = z_norm @ proto_norms.T  # (M,)
    pred_idx = int(np.argmax(similarities))

    return pred_idx, similarities


def batch_cosine_match(
    query_features: np.ndarray,   # (Nq, d)
    prototypes: np.ndarray,       # (M, d)
) -> Tuple[np.ndarray, np.ndarray]:
    """批量原型匹配"""
    q_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    p_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

    similarities = q_norm @ p_norm.T          # (Nq, M)
    predictions = np.argmax(similarities, axis=1)

    return predictions, similarities


# ============================================================
# 4. 自适应原型更新 (论文 Section III, Stage 4, Eq. 10)
# ============================================================

def update_prototype_ema(
    bank: PrototypeBank,
    z_new: np.ndarray,           # (d,) 新查询特征
    pred_class: int,             # 预测类别
    confidence_threshold: float = 0.7,
    beta: float = 0.1,           # EMA 更新率
    similarity: float = None,     # 余弦相似度 (用于置信度判断)
) -> bool:
    """
    自适应EMA原型更新 (Eq. 10)

    论文原文: "For query samples with high matching confidence,
    the GS updates the prototype of the predicted class ŷ
    using an exponential moving average (EMA) scheme."

    数学表达:
    c_low_ŷ(d, t+1) = (1-β) · c_low_ŷ(d, t) + β · z_new  (Eq. 10)

    注意: 只更新 base prototype 的前 d 维

    Args:
        bank: 原型仓库
        z_new: (d,) 新特征
        pred_class: 预测的类别
        confidence_threshold: 置信度阈值
        beta: EMA 更新率 β (论文未指定具体值)
        similarity: 余弦相似度

    Returns:
        updated: 是否执行了更新
    """
    if similarity is not None and similarity < confidence_threshold:
        return False

    d = z_new.shape[0]
    # L2归一化新特征
    z_norm = z_new / (np.linalg.norm(z_new) + 1e-8)

    # Eq. 10: c(t+1) = (1-β)·c(t) + β·z_new
    old_proto = bank.base_prototypes[pred_class, :d]
    new_proto = (1.0 - beta) * old_proto + beta * z_norm

    # 更新后重新归一化
    new_proto = new_proto / (np.linalg.norm(new_proto) + 1e-8)

    bank.base_prototypes[pred_class, :d] = new_proto
    return True


# ============================================================
# 端到端推理流程 (ProSTAR Inference Pipeline)
# ============================================================

@dataclass
class ProSTARInferenceResult:
    pred_class: int
    pred_name: str
    similarities: np.ndarray
    compressed_dim: int
    layer_idx: int

@dataclass
class ProSTARConfig:
    layer_idx: int = 0
    target_dim: int = 256
    confidence_threshold: float = 0.7
    ema_beta: float = 0.1
    seed: int = 42


class ProSTARPipeline:
    """ProSTAR 完整推理流水线"""

    def __init__(self, config: ProSTARConfig = None):
        self.config = config or ProSTARConfig()
        self.prototype_bank: Optional[PrototypeBank] = None
        self.P_base: Optional[np.ndarray] = None

    def initialize(self, features: np.ndarray, labels: np.ndarray,
                   class_names: List[str]):
        """Stage 1: 原型初始化 (Eq. 7)"""
        self.prototype_bank, self.P_base = initialize_prototypes(
            features, labels, class_names,
            layer_idx=self.config.layer_idx,
            seed=self.config.seed,
        )

    def extract_and_compress(
        self, feature: np.ndarray, d: int = None
    ) -> np.ndarray:
        """Stage 2: 特征压缩 (Eq. 8)"""
        d = d or self.config.target_dim
        return compress_feature(feature, self.P_base, d)

    def infer(
        self, feature: np.ndarray, d: int = None
    ) -> ProSTARInferenceResult:
        """Stage 3: 原型匹配 (Eq. 9)"""
        d = d or self.config.target_dim
        z_new = self.extract_and_compress(feature, d)

        prototypes = self.prototype_bank.get_prototype(d)
        pred_idx, similarities = cosine_similarity_match(z_new, prototypes)

        return ProSTARInferenceResult(
            pred_class=pred_idx,
            pred_name=self.prototype_bank.class_names[pred_idx],
            similarities=similarities,
            compressed_dim=d,
            layer_idx=self.config.layer_idx,
        )

    def infer_and_update(
        self, feature: np.ndarray, d: int = None
    ) -> ProSTARInferenceResult:
        """Stage 3+4: 匹配 + 自适应更新"""
        result = self.infer(feature, d)
        d = result.compressed_dim

        z_new = self.extract_and_compress(feature, d)
        max_sim = float(result.similarities[result.pred_class])

        update_prototype_ema(
            self.prototype_bank, z_new, result.pred_class,
            confidence_threshold=self.config.confidence_threshold,
            beta=self.config.ema_beta,
            similarity=max_sim,
        )

        return result
