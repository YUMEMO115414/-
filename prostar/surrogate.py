"""
ProSTAR 代理模型: 联合优化方案 (论文 Section IV)

核心:
1. 改进的Sigmoid代理模型 (Eq. 16)
2. 非线性最小二乘参数拟合 (Eq. 18)
3. 在线自适应决策 (Eq. 19)

问题表述:
min_{l,d} J(l,d) = λ₁(A_max - A(l,d)) + λ₂ T_total(l,d) + λ₃ E_total(l,d)  (Eq. 13)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 代理模型 (Surrogate Model) — Eq. 16
# ============================================================

@dataclass
class LayerSurrogateParams:
    """单层代理模型参数 θ*_l = {A_max, A_min, α, d_0}"""
    layer_idx: int
    A_max: float       # 无降维时的baseline准确率上界
    A_min: float       # 极端压缩时的准确率下界 (≈随机猜测)
    alpha: float       # 层敏感系数 (控制退化曲线陡峭度)
    d0: float          # 临界阈值 (准确率开始急剧下降的点)
    D_l: int = 0       # 该层最大特征维度

    def predict(self, d: float) -> float:
        """代理模型预测 (Eq. 16) — 数值稳定版, capping at 100%"""
        x = np.clip(-self.alpha * (d - self.d0), -50.0, 50.0)
        return min(100.0, self.A_max + (self.A_max - self.A_min) / (1.0 + np.exp(x)))

    def predict_batch(self, d_array: np.ndarray) -> np.ndarray:
        """批量预测 — 数值稳定版, capping at 100%"""
        result = _stable_sigmoid(d_array.astype(np.float64),
                                 self.A_max, self.A_min, self.alpha, self.d0)
        return np.clip(result, 0.0, 100.0)


def sigmoid_surrogate(d: np.ndarray, A_max: float, A_min: float,
                      alpha: float, d0: float) -> np.ndarray:
    """原始 Eq. 16: modified sigmoid with baseline offset

    论文解释: 该函数捕获了"特征维度约减 → 准确率变化"的非线性饱和特征:
    - d 较大时: 冗余信息被移除,准确率基本稳定
    - d 低于临界阈值时: 关键信息丢失,准确率急剧下降
    """
    return A_max + (A_max - A_min) / (1.0 + np.exp(-alpha * (d - d0)))


# ============================================================
# 非线性最小二乘拟合 — Eq. 17, Eq. 18
# ============================================================

def generate_empirical_data(
    d_values: np.ndarray,   # (J,) 各压缩维度
    accuracies: np.ndarray, # (J,) 对应准确率
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建经验数据集 D^l_profile = {(d_j, A(l, d_j))}  (Eq. 17)
    论文: 在验证集上对每个 partition layer l 进行启发式搜索
    """
    return d_values.astype(np.float64), accuracies.astype(np.float64)


def _stable_sigmoid(d: np.ndarray, A_max: float, A_min: float,
                    alpha: float, d0: float) -> np.ndarray:
    """数值稳定的 sigmoid 计算

    使用两种形式避免 exp 溢出:
    - 当 alpha*(d-d0) 较大时: 使用 exp(-x) 形式
    - 当 alpha*(d-d0) 较小时: 使用 exp(x) 形式 (在 A_max >> A_min 时退化)
    """
    x = -alpha * (d - d0)

    # 裁剪指数参数避免溢出 (exp(80) ≈ 5.5e34, 单精度可处理)
    x_clipped = np.clip(x, -80.0, 80.0)

    # 数值稳定: 1/(1+exp(-x))
    # 对 x > 0: sigmoid → 1; 对 x < 0: sigmoid → 0
    denom = 1.0 + np.exp(x_clipped)
    sigmoid_val = 1.0 / denom

    return A_max + (A_max - A_min) * sigmoid_val


def _safe_exp(x):
    """安全的指数函数"""
    return np.exp(np.clip(x, -50.0, 50.0))


def fit_surrogate_params(
    d_values: np.ndarray,   # (J,) 维度值
    accuracies: np.ndarray, # (J,) 实际准确率
    layer_idx: int = 0,
    D_l: int = 768,
) -> LayerSurrogateParams:
    """
    非线性最小二乘拟合 (Eq. 18)
    θ*_l = argmin_θ Σ (A(l,d_j) - Â_l(d_j;θ))²

    使用带有数值稳定性保护的梯度下降
    """
    d = d_values.astype(np.float64)
    A = accuracies.astype(np.float64)

    # --- 智能初始值估计 ---
    A_max_init = float(np.max(A))

    # A_min: 假设随机猜测 ≈ 1/M 作为先验，但不应低于实际最低值
    num_classes_guess = max(2, int(100.0 / (100.0 - A_max_init + 1.0)))
    A_min_prior = 100.0 / num_classes_guess  # 随机猜测基线
    A_min_init = max(A_min_prior, float(np.min(A)))
    A_min_init = min(A_min_init, A_max_init * 0.5)

    # d0: 在 [d_min, d_max] 中间位置初始化
    d0_init = float(d[len(d) // 2])

    # alpha: 使用 log-space 参数化; alpha = exp(alpha_log)
    d_range = d.max() - d.min()
    alpha_log_init = np.log(2.0 / (d_range + 1e-8))  # 温和初始值

    # --- 参数: [A_max, A_min, alpha_log, d0] ---
    params = np.array([A_max_init, A_min_init, alpha_log_init, d0_init], dtype=np.float64)

    # 使用随机小批量 + 小学习率的稳定梯度下降
    d_min, d_max = d.min(), d.max()
    A_range = A_max_init - A_min_init

    best_loss = float('inf')
    best_params = params.copy()

    for iteration in range(2000):
        A_max, A_min, alpha_log, d0_c = params
        alpha_c = _safe_exp(alpha_log)  # alpha = exp(log_alpha) → always positive

        # 前向计算 (数值稳定)
        A_hat = _stable_sigmoid(d, A_max, A_min, alpha_c, d0_c)
        residuals = A - A_hat

        # 检查数值问题
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            # 回退到安全值
            params = best_params.copy()
            break

        loss = np.mean(residuals ** 2)
        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()

        # 梯度计算 (使用 _stable_sigmoid 的内部结构)
        x = np.clip(-alpha_c * (d - d0_c), -80.0, 80.0)
        sig = 1.0 / (1.0 + _safe_exp(x))
        dsig = sig * (1.0 - sig)  # sigmoid 的标准导数

        common = (A_max - A_min) * dsig

        dA_dAmax = 1.0 + sig           # ∂Â/∂A_max
        dA_dAmin = -sig                 # ∂Â/∂A_min
        dA_dAlpha = -common * (d - d0_c) * alpha_c  # ∂Â/∂log_alpha = common * (-(d-d0)) * alpha
        dA_dd0 = common * alpha_c       # ∂Â/∂d0

        # Jacobian-vector product
        dJ_dAmax = -2.0 * np.mean(dA_dAmax * residuals)
        dJ_dAmin = -2.0 * np.mean(dA_dAmin * residuals)
        dJ_dAlpha = -2.0 * np.mean(dA_dAlpha * residuals)
        dJ_dd0 = -2.0 * np.mean(dA_dd0 * residuals)

        grad = np.array([dJ_dAmax, dJ_dAmin, dJ_dAlpha, dJ_dd0])

        # 自适应学习率
        lr = 0.1 / (1.0 + 0.001 * iteration)

        params = params - lr * grad

        # 约束
        params[0] = np.clip(params[0], max(A_max_init * 0.9, 10.0), 100.0)  # A_max
        params[1] = np.clip(params[1], 0.0, params[0] * 0.8)                # A_min
        params[2] = np.clip(params[2], np.log(0.001), np.log(100.0))        # alpha_log
        params[3] = np.clip(params[3], d_min, d_max)                        # d0

        if iteration % 100 == 0 and np.max(np.abs(grad)) < 1e-6:
            break

    A_max, A_min, alpha_log, d0_c = best_params
    alpha_c = float(_safe_exp(alpha_log))
    A_max = np.clip(A_max, max(float(A.max()) * 0.9, 10.0), 100.0)
    A_min = np.clip(A_min, 0.0, A_max * 0.8)
    d0_c = np.clip(d0_c, d_min, d_max)

    return LayerSurrogateParams(
        layer_idx=layer_idx,
        A_max=float(A_max),
        A_min=float(A_min),
        alpha=float(alpha_c),
        d0=float(d0_c),
        D_l=D_l,
    )


# ============================================================
# 延迟与能耗模型 — Eq. 1~6
# ============================================================

@dataclass
class SystemParams:
    """卫星-地面协同推理系统参数"""
    # 计算参数
    f_sat: float = 5.0e9         # 卫星计算频率 (Hz) — 论文: 5GHz
    kappa: float = 1e-28          # CMOS电容系数
    gamma_l: np.ndarray = None    # 各层累积FLOPs比例

    # 通信参数
    P_tx: float = 15.0            # 发射功率 (W) — 论文: 15W
    G_tx: float = 10.0            # 发射天线增益 (线性)
    G_rx: float = 10.0            # 接收天线增益
    B: float = 1.0e6              # 带宽 (Hz) — 论文: 1MHz
    N0: float = 1e-19             # 噪声功率谱密度 (W/Hz)
    f_c: float = 2.4e9            # 载波频率 (Hz)
    x_sg: float = 800e3           # 星地距离 (m)
    b: int = 32                   # 每维量化比特 (Float32)
    c: float = 3e8                # 光速 (m/s)
    F_total: float = 4.5e9        # DNN总FLOPs (Swin-Tiny约4.5G)

    def channel_rate(self) -> float:
        """Shannon公式计算可达到的通信速率 (Eq. 4)"""
        # 自由空间路径损耗
        L = (4 * np.pi * self.x_sg * self.f_c / self.c) ** 2
        h = 1.0 / L
        snr = (self.P_tx * self.G_tx * self.G_rx * h) / (self.N0 * self.B)
        return self.B * np.log2(1.0 + snr)

    def comp_latency(self, l: int) -> float:
        """卫星计算延迟 T_comp(l) = C(l)/f_sat (Eq. 1)"""
        C_l = self.gamma_l[l] * self.F_total
        return C_l / self.f_sat

    def comp_energy(self, l: int) -> float:
        """卫星计算能耗 E_comp(l) = κ·f²_sat·C(l) (Eq. 2)"""
        C_l = self.gamma_l[l] * self.F_total
        return self.kappa * (self.f_sat ** 2) * C_l

    def comm_latency(self, d: int) -> float:
        """通信延迟 T_comm(d) = S(d)/R + d_sg/c (Eq. 5)"""
        S = d * self.b
        R = self.channel_rate()
        return S / R + self.x_sg / self.c

    def comm_energy(self, d: int) -> float:
        """通信能耗 E_comm(d) = P_tx·S(d)/R (Eq. 6)"""
        S = d * self.b
        R = self.channel_rate()
        return self.P_tx * S / R

    def total_latency(self, l: int, d: int) -> float:
        """总延迟 T_total(l,d) (Eq. 11)"""
        return self.comp_latency(l) + self.comm_latency(d)

    def total_energy(self, l: int, d: int) -> float:
        """总能耗 E_total(l,d) (Eq. 12)"""
        return self.comp_energy(l) + self.comm_energy(d)


# ============================================================
# 联合代价函数 — Eq. 13
# ============================================================

@dataclass
class OptimizationWeights:
    """优化权重 λ₁, λ₂, λ₃ (Eq. 13)"""
    lambda_acc: float = 200.0     # λ₁: 准确率权重 (论文: 200)
    lambda_lat: float = 30.0      # λ₂: 延迟权重 (论文: 30)
    lambda_energy: float = 2.0    # λ₃: 能耗权重 (论文: 2)


def joint_cost(
    l: int, d: int,
    A_max_ideal: float,
    accuracy: float,
    lat: float, energy: float,
    weights: OptimizationWeights,
) -> float:
    """
    联合代价函数 (Eq. 13)
    J(l,d) = λ₁ max(0, A_max - A(l,d)) + λ₂ T_total + λ₃ E_total

    使用 max(0, ...) 防止 accuracy > A_max 时出现负代价
    """
    acc_penalty = max(0.0, A_max_ideal - accuracy)
    return (weights.lambda_acc * acc_penalty +
            weights.lambda_lat * lat +
            weights.lambda_energy * energy)


# ============================================================
# 在线自适应决策 — Eq. 19
# ============================================================

@dataclass
class OptimalDecision:
    layer_idx: int
    target_dim: int
    predicted_accuracy: float
    total_latency_ms: float
    total_energy_j: float
    joint_cost: float


def online_decision(
    surrogate_models: List[LayerSurrogateParams],  # 各层代理模型参数
    sys_params: SystemParams,
    weights: OptimizationWeights,
    A_max_ideal: float,
    d_candidates: List[np.ndarray],  # 每层的维度候选集
) -> OptimalDecision:
    """
    在线自适应决策 (Eq. 19)

    论文: 卫星只需插入实时系统参数, 进行轻量级数值搜索
    复杂度: O(Σ|D_l|) — 纯代数运算, 无需高维张量计算

    (l*, d*) = argmin_{l,d} [λ₁(A^l_max - Â_l(d)) + λ₂ T_total(l,d) + λ₃ E_total(l,d)]
    """
    best_cost = float('inf')
    best_decision = None

    for l, surrogate in enumerate(surrogate_models):
        for d in d_candidates[l]:
            d = int(d)
            if d < 1:
                continue

            # 用代理模型预测准确率 (Eq. 16)
            acc_pred = surrogate.predict(float(d))

            # 计算系统开销 (Eq. 11, 12)
            lat = sys_params.total_latency(l, d)
            energy = sys_params.total_energy(l, d)

            # 联合代价 (Eq. 19)
            cost = joint_cost(l, d, A_max_ideal, acc_pred, lat, energy, weights)

            if cost < best_cost:
                best_cost = cost
                best_decision = OptimalDecision(
                    layer_idx=l,
                    target_dim=d,
                    predicted_accuracy=acc_pred,
                    total_latency_ms=lat * 1000,
                    total_energy_j=energy,
                    joint_cost=cost,
                )

    return best_decision


def generate_d_candidates(D_l: int, num_levels: int = 24) -> np.ndarray:
    """生成均匀间隔的维度候选集 (论文: 24个压缩级别)"""
    return np.linspace(1, D_l, num_levels).astype(int)


# ============================================================
# 完整离线-在线流程
# ============================================================

def build_surrogate_models(
    empirical_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    layer_dims: Dict[int, int],
) -> List[LayerSurrogateParams]:
    """
    构建所有层的代理模型 (Offline at GS)
    返回: 轻量参数表 θ*_1, ..., θ*_L (传输到卫星)
    """
    models = []
    for l in sorted(empirical_data.keys()):
        d_vals, acc_vals = empirical_data[l]
        model = fit_surrogate_params(d_vals, acc_vals, layer_idx=l, D_l=layer_dims.get(l, d_vals.max()))
        models.append(model)
    return models
