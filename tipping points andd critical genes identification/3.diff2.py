"""
Master Regulator Detection via Dynamic Perturbation Analysis
============================================================
用于识别发育阶段转变过程中的关键转录因子(TF)和靶基因

主要功能：
1. 动态速度分析与网络平滑
2. Tipping Point 检测与Bootstrap验证
3. TF虚拟扰动分析
4. 靶基因执行层识别
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import zscore
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# =============================
# 配置类
# =============================
@dataclass
class Config:
    """分析参数配置"""
    # 输入文件
    csv_file: str = "merged_embedding_common_genes_diff.csv"
    network_file: str = "network_union_all_stages_diff.csv"

    # 分析参数
    bootstrap_n: int = 1000
    top_percent: float = 0.05
    smoothing_alpha: float = 0.2
    num_timepoints: int = 4
    embedding_dim: int = 512

    # 输出文件
    output_dir: str = "results"
    tf_driver_output: str = "TF_driver_scores.csv"
    target_gene_output: str = "Target_gene_execution_scores.csv"

    # 随机种子
    random_seed: int = 42


# =============================
# 数据加载模块
# =============================
class DataLoader:
    """数据加载与预处理"""

    def __init__(self, config: Config):
        self.config = config
        self.genes: List[str] = []
        self.gene_to_idx: Dict[str, int] = {}
        self.num_genes: int = 0
        self.X: Optional[np.ndarray] = None
        self.valid_edges: Optional[pd.DataFrame] = None

    def load(self) -> 'DataLoader':
        """加载所有数据"""
        self._load_embeddings()
        self._load_network()
        return self

    def _load_embeddings(self):
        """加载基因embedding数据"""
        df = pd.read_csv(self.config.csv_file, index_col=0)
        self.genes = df.index.tolist()
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.genes)}
        self.num_genes = len(self.genes)
        self.X = df.values.reshape(
            self.config.num_timepoints,
            self.num_genes,
            self.config.embedding_dim
        )
        print(f"✓ 加载 {self.num_genes} 个基因, {self.config.num_timepoints} 个时间点")

    def _load_network(self):
        """加载基因调控网络"""
        edges_df = pd.read_csv(self.config.network_file)
        self.valid_edges = edges_df[
            edges_df['TF'].isin(self.gene_to_idx) &
            edges_df['Gene'].isin(self.gene_to_idx)
        ].copy()
        print(f"✓ 加载 {len(self.valid_edges)} 条有效调控边")


# =============================
# 动态分析模块
# =============================
class DynamicAnalyzer:
    """动态速度分析与网络平滑"""

    def __init__(self, data: DataLoader, config: Config):
        self.data = data
        self.config = config
        self.velocities: Optional[np.ndarray] = None
        self.adj_matrices: List[sparse.csr_matrix] = []

    def analyze(self) -> 'DynamicAnalyzer':
        """执行动态分析"""
        X = self.data.X
        num_t = X.shape[0]
        num_genes = self.data.num_genes
        alpha = self.config.smoothing_alpha

        self.velocities = np.zeros((num_t - 1, num_genes, self.config.embedding_dim))
        self.adj_matrices = []

        for t in range(num_t - 1):
            # 构建时间点特异的邻接矩阵
            norm_adj_t = self._build_adjacency_matrix(t)
            self.adj_matrices.append(norm_adj_t)

            # 计算平滑后的速度
            v_raw = X[t + 1] - X[t]
            self.velocities[t] = (1 - alpha) * v_raw + alpha * (norm_adj_t @ v_raw)

        print(f"✓ 完成 {num_t - 1} 个时间区间的动态分析")
        return self

    def _build_adjacency_matrix(self, t: int) -> sparse.csr_matrix:
        """构建归一化邻接矩阵"""
        X = self.data.X
        rows, cols, weights = [], [], []

        for _, row in self.data.valid_edges.iterrows():
            u = self.data.gene_to_idx[row['TF']]
            v = self.data.gene_to_idx[row['Gene']]

            # 余弦相似度归一化到[0,1]
            sim = self._cosine_similarity(X[t, u], X[t, v])
            w = max(0, (sim + 1) / 2)

            rows.extend([u, v])
            cols.extend([v, u])
            weights.extend([w, w])

        adj_t = sparse.csr_matrix(
            (weights, (rows, cols)),
            shape=(self.data.num_genes, self.data.num_genes)
        )
        return self._normalize_adjacency(adj_t)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    @staticmethod
    def _normalize_adjacency(adj: sparse.csr_matrix) -> sparse.csr_matrix:
        """行归一化邻接矩阵"""
        d = np.array(adj.sum(axis=1)).flatten()
        d_inv = np.where(d > 0, 1.0 / d, 0.0)
        return sparse.diags(d_inv) @ adj


# =============================
# Tipping Point 检测模块
# =============================
@dataclass
class TippingPointResult:
    """Tipping Point检测结果"""
    best_t: int
    reliability: float
    mean_speed: np.ndarray
    network_entropy: np.ndarray
    tipping_score: np.ndarray
    bootstrap_distribution: np.ndarray


class TippingPointDetector:
    """Tipping Point检测与Bootstrap验证"""

    def __init__(self, dynamic: DynamicAnalyzer, config: Config):
        self.dynamic = dynamic
        self.config = config

    def detect(self) -> TippingPointResult:
        """检测tipping point并进行bootstrap验证"""
        velocities = self.dynamic.velocities

        # 计算网络熵和平均速度
        network_entropy = self._compute_network_entropy(velocities)
        mean_speed = np.linalg.norm(velocities, axis=2).mean(axis=1)

        # 综合tipping score
        tipping_score = zscore(mean_speed) + zscore(network_entropy)
        best_t = int(np.argmax(tipping_score))
        # 打印每个阶段转换的综合 tipping score
        print("\n各阶段的综合 Tipping Score：")
        for t, score in enumerate(tipping_score):
            print(f"  阶段 {t + 1} → {t + 2}: {score:.4f}")
        # Bootstrap验证
        bootstrap_dist = self._bootstrap_validation(velocities)
        reliability = np.mean(bootstrap_dist == best_t)

        result = TippingPointResult(
            best_t=best_t,
            reliability=reliability,
            mean_speed=mean_speed,
            network_entropy=network_entropy,
            tipping_score=tipping_score,
            bootstrap_distribution=bootstrap_dist
        )

        print(f"✓ Tipping Point: 阶段 {best_t + 1} → {best_t + 2}")
        print(f"  Bootstrap可靠性: {reliability * 100:.1f}%")

        return result

    def _compute_network_entropy(self, velocities: np.ndarray) -> np.ndarray:
        """计算每个时间点的网络熵"""
        entropy_list = []
        for t in range(velocities.shape[0]):
            v_t = velocities[t]
            p = np.linalg.norm(v_t, axis=1)
            p = p / (p.sum() + 1e-8)
            entropy = -np.sum(p * np.log(p + 1e-8))
            entropy_list.append(entropy)
        return np.array(entropy_list)

    def _bootstrap_validation(self, velocities: np.ndarray) -> np.ndarray:
        """Bootstrap验证tipping point的稳定性"""
        np.random.seed(self.config.random_seed)
        num_genes = velocities.shape[1]
        bootstrap_best_t = []

        for _ in range(self.config.bootstrap_n):
            sample_idx = np.random.choice(num_genes, num_genes, replace=True)
            v_sample = velocities[:, sample_idx, :]

            mean_speed_b = np.linalg.norm(v_sample, axis=2).mean(axis=1)
            entropy_b = self._compute_network_entropy(v_sample)

            score_b = zscore(mean_speed_b) + zscore(entropy_b)
            bootstrap_best_t.append(np.argmax(score_b))

        return np.array(bootstrap_best_t)


# =============================
# 扰动分析模块
# =============================
class PerturbationAnalyzer:
    """TF虚拟扰动分析"""

    def __init__(self, data: DataLoader, config: Config):
        self.data = data
        self.config = config
        self.A_norm: Optional[sparse.csr_matrix] = None

    def build_stage_network(self, stage_idx: int) -> 'PerturbationAnalyzer':
        """构建阶段特异的CMI加权网络"""
        rows, cols, weights = [], [], []
        cmi_col = f'CMI_{stage_idx}'

        for _, row in self.data.valid_edges.iterrows():
            u = self.data.gene_to_idx[row['TF']]
            v = self.data.gene_to_idx[row['Gene']]
            w = row.get(cmi_col, 0)

            if w > 0:
                rows.append(u)
                cols.append(v)
                weights.append(w)

        A = sparse.csr_matrix(
            (weights, (rows, cols)),
            shape=(self.data.num_genes, self.data.num_genes)
        )

        # 行归一化
        row_sum = np.array(A.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1.0
        self.A_norm = sparse.diags(1.0 / row_sum) @ A

        print(f"✓ 构建阶段 {stage_idx} 的CMI加权网络")
        return self

    def virtual_perturbation(self, tf_idx: int, X_ref: np.ndarray,
                             alpha: float = 1.0) -> np.ndarray:
        """对单个TF进行虚拟扰动并传播（方向一致的一阶线性响应）"""
        u = np.zeros_like(X_ref)

        tf_vec = X_ref[tf_idx]
        scale = np.linalg.norm(tf_vec)

        if scale < 1e-6:
            return np.zeros_like(X_ref)

        # 方向扰动（不引入幅值偏置）
        u[tf_idx] = alpha * tf_vec / (scale + 1e-8)

        return self.A_norm @ u

    def analyze_tf_drivers(self, t0: int) -> pd.DataFrame:
        """分析TF驱动力得分"""
        X3 = self.data.X[t0]
        delta_X = self.data.X[t0 + 1] - X3

        # 获取TF列表
        TFs = sorted(self.data.valid_edges['TF'].unique())
        TFs = [tf for tf in TFs if tf in self.data.gene_to_idx]

        tf_scores = []
        for tf in TFs:
            tf_idx = self.data.gene_to_idx[tf]
            delta_tf = self.virtual_perturbation(tf_idx, X3)

            # 计算与真实转变的对齐度
            num = np.sum(delta_tf * delta_X)
            denom = np.linalg.norm(delta_tf) * np.linalg.norm(delta_X) + 1e-8

            tf_scores.append({
                "TF": tf,
                "AlignScore": num / denom,
                "EffectNorm": np.linalg.norm(delta_tf)
            })

        df = pd.DataFrame(tf_scores).sort_values("AlignScore", ascending=False)
        print(f"✓ 分析了 {len(TFs)} 个TF的驱动力")
        return df

    def analyze_target_genes(self, t0: int, top_tfs: List[str]) -> pd.DataFrame:
        """分析靶基因执行得分"""
        X3 = self.data.X[t0]
        delta_X = self.data.X[t0 + 1] - X3
        top_tf_indices = [self.data.gene_to_idx[tf] for tf in top_tfs]

        gene_scores = []
        for g, gene in enumerate(self.data.genes):
            align_scores = []

            for tf_idx in top_tf_indices:
                delta_tf = self.virtual_perturbation(tf_idx, X3)
                dx_tf = delta_tf[g]
                dx_real = delta_X[g]

                if np.linalg.norm(dx_tf) < 1e-8 or np.linalg.norm(dx_real) < 1e-8:
                    continue

                align = np.dot(dx_tf, dx_real) / (
                    np.linalg.norm(dx_tf) * np.linalg.norm(dx_real) + 1e-8
                )
                align_scores.append(align)

            if align_scores:
                gene_scores.append({
                    "Gene": gene,
                    "ExecScore": np.mean(align_scores),
                    "AlignStd": np.std(align_scores),
                    "SupportTFs": len(align_scores)
                })

        df = pd.DataFrame(gene_scores).sort_values("ExecScore", ascending=False)
        # 排除TF本身
        df = df[~df["Gene"].isin(top_tfs)]

        print(f"✓ 识别了 {len(df)} 个靶基因")
        return df


# =============================
# 可视化模块
# =============================
class Visualizer:
    """结果可视化"""

    @staticmethod
    def plot_tipping_point(result: TippingPointResult, save_path: Optional[str] = None):
        """绘制tipping point分析图"""
        num_t = len(result.mean_speed)
        x = range(1, num_t + 1)
        plt.rcParams.update({
            "font.size": 18,  # 默认字体
            "axes.titlesize": 18,  # 标题
            "axes.labelsize": 18,  # 坐标轴标签
            "xtick.labelsize": 18,  # x刻度
            "ytick.labelsize": 18,  # y刻度
            "legend.fontsize": 16  # 图例
        })
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x, zscore(result.mean_speed), 'o-', label='System Velocity')
        ax.plot(x, zscore(result.network_entropy), 's-', color='#259DBB', label='Network Entropy')
        ax.axvline(result.best_t + 1, color='red', linestyle='--',
                   label='Predicted Tipping Point')
        ax.fill_between(x, -2, zscore(result.tipping_score),
                        alpha=0.6, color='#D7EEF0', label='Transition Intensity')

        ax.set_xlabel('Time Point')
        ax.set_ylabel('Z-Score')
        ax.set_title(
            f"Comprehensive identification of tipping points\n"

        )
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图片已保存: {save_path}")
        plt.show()

    @staticmethod
    def plot_subgraph(subgraph_df: pd.DataFrame, merged_df: pd.DataFrame,
                      save_path: Optional[str] = None, top_n: int = 100):
        """
        绘制调控子图网络

        Parameters:
        -----------
        subgraph_df : 子图边数据
        merged_df : 合并的基因得分数据
        save_path : 保存路径
        top_n : 显示的top节点数量
        """
        try:
            import networkx as nx
        except ImportError:
            print("⚠ 需要安装networkx: pip install networkx")
            return

        # 创建有向图
        G = nx.DiGraph()

        # 获取top基因用于可视化
        top_genes = set(merged_df.head(top_n)['Gene'].tolist())

        # 筛选与top基因相关的边
        viz_edges = subgraph_df[
            subgraph_df['TF'].isin(top_genes) |
            subgraph_df['Gene'].isin(top_genes)
        ]

        # 添加边
        for _, row in viz_edges.iterrows():
            tf, gene = row['TF'], row['Gene']
            # 使用平均CMI作为边权重
            cmi_cols = [c for c in row.index if c.startswith('CMI_')]
            avg_cmi = row[cmi_cols].mean() if cmi_cols else 1.0
            G.add_edge(tf, gene, weight=avg_cmi)

        if len(G.nodes()) == 0:
            print("⚠ 子图中没有边可供可视化")
            return

        # 创建基因类型和得分映射
        gene_info = merged_df.set_index('Gene')

        # 节点颜色：TF为红色，Target为蓝色
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node in gene_info.index:
                info = gene_info.loc[node]
                if info['Type'] == 'TF_Driver':
                    node_colors.append('#E74C3C')  # 红色
                    node_sizes.append(800 + info['Score'] * 500)
                else:
                    node_colors.append('#3498DB')  # 蓝色
                    node_sizes.append(400 + info['Score'] * 300)
            else:
                node_colors.append('#95A5A6')  # 灰色
                node_sizes.append(200)

        # 边权重用于线条粗细
        edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]

        # 布局
        fig, ax = plt.subplots(figsize=(14, 10))

        # 使用spring布局
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # 绘制边
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#BDC3C7',
            width=edge_weights,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1"
        )

        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9
        )

        # 绘制标签
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=8,
            font_weight='bold'
        )

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
                   markersize=15, label='TF Driver'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
                   markersize=12, label='Target Gene'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        ax.set_title(f"Regulatory Subgraph (Top {top_n} Genes)\n"
                     f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}",
                     fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 网络图已保存: {save_path}")
        plt.show()

    @staticmethod
    def plot_score_distribution(merged_df: pd.DataFrame, save_path: Optional[str] = None):
        """绘制得分分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. TF vs Target 得分分布
        ax1 = axes[0]
        tf_scores = merged_df[merged_df['Type'] == 'TF_Driver']['Score']
        target_scores = merged_df[merged_df['Type'] == 'Target_Gene']['Score']

        ax1.hist(tf_scores, bins=30, alpha=0.7, label='TF Drivers', color='#E74C3C')
        ax1.hist(target_scores, bins=30, alpha=0.7, label='Target Genes', color='#3498DB')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Score Distribution by Type')
        ax1.legend()

        # 2. Top基因得分条形图
        ax2 = axes[1]
        top_20 = merged_df.head(20)
        colors = ['#E74C3C' if t == 'TF_Driver' else '#3498DB'
                  for t in top_20['Type']]
        bars = ax2.barh(range(len(top_20)), top_20['Score'], color=colors)
        ax2.set_yticks(range(len(top_20)))
        ax2.set_yticklabels(top_20['Gene'], fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Score')
        ax2.set_title('Top 20 Genes by Score')

        # 3. 类型占比饼图
        ax3 = axes[2]
        type_counts = merged_df['Type'].value_counts()
        ax3.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
                colors=['#E74C3C', '#3498DB'])
        ax3.set_title('Gene Type Distribution')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 分布图已保存: {save_path}")
        plt.show()


# =============================
# 结果整合与子图提取模块
# =============================
class ResultIntegrator:
    """整合分析结果、提取子图"""

    def __init__(self, data: DataLoader, config: Config):
        self.data = data
        self.config = config
        self.merged_df: Optional[pd.DataFrame] = None
        self.subgraph_df: Optional[pd.DataFrame] = None

    def merge_results(self, tf_scores: pd.DataFrame,
                      target_scores: pd.DataFrame) -> pd.DataFrame:
        """
        合并TF driver和Target gene结果

        Parameters:
        -----------
        tf_scores : TF驱动力得分表
        target_scores : 靶基因执行得分表

        Returns:
        --------
        合并后的DataFrame，包含统一的得分和类型标签
        """
        # 准备TF数据
        tf_df = tf_scores.copy()
        tf_df['Type'] = 'TF_Driver'
        tf_df['Score'] = tf_df['AlignScore']
        tf_df = tf_df.rename(columns={'TF': 'Gene'})
        tf_df['Rank_in_Type'] = range(1, len(tf_df) + 1)

        # 准备Target数据
        target_df = target_scores.copy()
        target_df['Type'] = 'Target_Gene'
        target_df['Score'] = target_df['ExecScore']
        target_df['Rank_in_Type'] = range(1, len(target_df) + 1)

        # 选择需要的列
        tf_cols = ['Gene', 'Type', 'Score', 'AlignScore', 'EffectNorm', 'Rank_in_Type']
        target_cols = ['Gene', 'Type', 'Score', 'ExecScore', 'AlignStd', 'SupportTFs', 'Rank_in_Type']

        # 合并
        tf_subset = tf_df[tf_cols].copy()
        tf_subset['ExecScore'] = np.nan
        tf_subset['AlignStd'] = np.nan
        tf_subset['SupportTFs'] = np.nan

        target_subset = target_df[target_cols].copy()
        target_subset['AlignScore'] = target_subset['ExecScore']  # 使用ExecScore作为AlignScore
        target_subset['EffectNorm'] = np.nan

        # 统一列顺序
        final_cols = ['Gene', 'Type', 'Score', 'Rank_in_Type',
                      'AlignScore', 'EffectNorm', 'ExecScore', 'AlignStd', 'SupportTFs']

        tf_final = tf_subset.reindex(columns=final_cols)
        target_final = target_subset.reindex(columns=final_cols)

        # 合并并按Score排序
        self.merged_df = pd.concat([tf_final, target_final], ignore_index=True)
        self.merged_df = self.merged_df.sort_values('Score', ascending=False)
        self.merged_df['Overall_Rank'] = range(1, len(self.merged_df) + 1)

        # 调整列顺序
        col_order = ['Overall_Rank', 'Gene', 'Type', 'Score', 'Rank_in_Type',
                     'AlignScore', 'EffectNorm', 'ExecScore', 'AlignStd', 'SupportTFs']
        self.merged_df = self.merged_df[col_order]

        print(f"✓ 合并了 {len(tf_scores)} 个TF和 {len(target_scores)} 个靶基因")
        return self.merged_df

    def extract_subgraph(self, merged_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        从原始网络中提取关键基因的子图

        Parameters:
        -----------
        merged_df : 合并后的基因列表，如果为None则使用self.merged_df

        Returns:
        --------
        子图的边列表DataFrame
        """
        if merged_df is None:
            merged_df = self.merged_df

        if merged_df is None:
            raise ValueError("请先运行merge_results或提供merged_df")

        # 获取所有关键基因
        key_genes = set(merged_df['Gene'].tolist())

        # 从原始网络提取子图（边的两端都在关键基因中）
        network_df = self.data.valid_edges
        # 获取只用的 TF
        tf_nodes = set(merged_df.loc[merged_df['Type'] == 'TF_Driver', 'Gene'].tolist())

        # 从原始网络提取子图（边的TF必须是top TF，gene在关键基因里）
        subgraph = network_df[
            network_df['TF'].isin(tf_nodes) &
            network_df['Gene'].isin(key_genes)
            ].copy()

        # 添加基因类型和得分信息
        gene_info = merged_df.set_index('Gene')[['Type', 'Score', 'Overall_Rank']]

        # TF信息
        subgraph = subgraph.merge(
            gene_info.add_prefix('TF_'),
            left_on='TF', right_index=True, how='left'
        )

        # Target信息
        subgraph = subgraph.merge(
            gene_info.add_prefix('Target_'),
            left_on='Gene', right_index=True, how='left'
        )

        # 计算边的重要性得分（TF和Target得分的几何平均）
        subgraph['Edge_Importance'] = np.sqrt(
            subgraph['TF_Score'].fillna(0) * subgraph['Target_Score'].fillna(0)
        )

        # 按边重要性排序
        subgraph = subgraph.sort_values('Edge_Importance', ascending=False)

        self.subgraph_df = subgraph
        print(f"✓ 提取子图: {len(key_genes)} 个节点, {len(subgraph)} 条边")

        return self.subgraph_df

    def get_subgraph_stats(self) -> Dict:
        """获取子图统计信息"""
        if self.subgraph_df is None:
            return {}

        subgraph = self.subgraph_df

        # 节点统计
        all_nodes = set(subgraph['TF'].tolist()) | set(subgraph['Gene'].tolist())
        tf_nodes = set(subgraph[subgraph['TF_Type'] == 'TF_Driver']['TF'].tolist())
        target_nodes = all_nodes - tf_nodes

        # 边统计
        tf_to_tf = len(subgraph[
            (subgraph['TF_Type'] == 'TF_Driver') &
            (subgraph['Target_Type'] == 'TF_Driver')
        ])
        tf_to_target = len(subgraph[
            (subgraph['TF_Type'] == 'TF_Driver') &
            (subgraph['Target_Type'] == 'Target_Gene')
        ])

        # 度分布
        out_degree = subgraph.groupby('TF').size()
        in_degree = subgraph.groupby('Gene').size()

        stats = {
            'num_nodes': len(all_nodes),
            'num_tf_nodes': len(tf_nodes),
            'num_target_nodes': len(target_nodes),
            'num_edges': len(subgraph),
            'tf_to_tf_edges': tf_to_tf,
            'tf_to_target_edges': tf_to_target,
            'avg_out_degree': out_degree.mean(),
            'max_out_degree': out_degree.max(),
            'avg_in_degree': in_degree.mean(),
            'max_in_degree': in_degree.max(),
            'top_hubs': out_degree.nlargest(5).to_dict()
        }

        return stats


# =============================
# 主流程协调器
# =============================
class MasterRegulatorPipeline:
    """主分析流程"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.data: Optional[DataLoader] = None
        self.dynamic: Optional[DynamicAnalyzer] = None
        self.tipping_result: Optional[TippingPointResult] = None
        self.tf_scores: Optional[pd.DataFrame] = None
        self.target_scores: Optional[pd.DataFrame] = None
        self.integrator: Optional[ResultIntegrator] = None
        self.merged_df: Optional[pd.DataFrame] = None
        self.subgraph_df: Optional[pd.DataFrame] = None

    def run(self) -> 'MasterRegulatorPipeline':
        """执行完整分析流程"""
        print("=" * 60)
        print("Master Regulator Detection Pipeline")
        print("=" * 60)

        # 1. 加载数据
        print("\n[1/6] 加载数据...")
        self.data = DataLoader(self.config).load()

        # 2. 动态分析
        print("\n[2/6] 动态速度分析...")
        self.dynamic = DynamicAnalyzer(self.data, self.config).analyze()

        # 3. Tipping Point检测
        print("\n[3/6] Tipping Point检测...")
        detector = TippingPointDetector(self.dynamic, self.config)
        self.tipping_result = detector.detect()

        # 4. 扰动分析
        print("\n[4/6] TF扰动分析...")
        t0 = self.tipping_result.best_t
        perturb = PerturbationAnalyzer(self.data, self.config)
        perturb.build_stage_network(t0 + 1)

        self.tf_scores = perturb.analyze_tf_drivers(t0)

        # 获取Top TFs
        top_k = int(len(self.tf_scores) * self.config.top_percent)
        top_tfs = self.tf_scores.head(top_k)["TF"].tolist()

        self.target_scores = perturb.analyze_target_genes(t0, top_tfs)

        # 5. 结果整合与子图提取
        print("\n[5/6] 结果整合与子图提取...")
        self.integrator = ResultIntegrator(self.data, self.config)

        # 合并TF和Target结果
        self.merged_df = self.integrator.merge_results(
            self.tf_scores.head(top_k),  # 只合并top TFs
            self.target_scores
        )

        # 提取子图
        self.subgraph_df = self.integrator.extract_subgraph()

        # 6. 保存结果
        print("\n[6/6] 保存结果...")
        self._save_results()

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)

        return self

    def _save_results(self):
        """保存分析结果"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        t0 = self.tipping_result.best_t
        stage_str = f"stage{t0+1}_to_{t0+2}"

        # 保存TF得分
        tf_path = output_dir / f"TF_driver_scores_{stage_str}.csv"
        self.tf_scores.to_csv(tf_path, index=False)
        print(f"  → {tf_path}")

        # 保存靶基因得分
        target_path = output_dir / f"Target_gene_scores_{stage_str}.csv"
        self.target_scores.to_csv(target_path, index=False)
        print(f"  → {target_path}")

        # 保存合并结果
        merged_path = output_dir / f"Merged_key_genes_{stage_str}.csv"
        self.merged_df.to_csv(merged_path, index=False)
        print(f"  → {merged_path}")

        # 保存子图
        subgraph_path = output_dir / f"Regulatory_subgraph_{stage_str}.csv"
        self.subgraph_df.to_csv(subgraph_path, index=False)
        print(f"  → {subgraph_path}")

        # 保存子图统计信息
        stats = self.integrator.get_subgraph_stats()
        stats_path = output_dir / f"Subgraph_statistics_{stage_str}.txt"
        with open(stats_path, 'w') as f:
            f.write("Subgraph Statistics\n")
            f.write("=" * 40 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        print(f"  → {stats_path}")

    def visualize(self, save_dir: Optional[str] = None):
        """生成所有可视化"""
        if save_dir is None:
            save_dir = self.config.output_dir

        output_dir = Path(save_dir)
        output_dir.mkdir(exist_ok=True)

        t0 = self.tipping_result.best_t
        stage_str = f"stage{t0+1}_to_{t0+2}"

        print("\n生成可视化...")

        # 1. Tipping Point分析图
        if self.tipping_result:
            Visualizer.plot_tipping_point(
                self.tipping_result,
                save_path=str(output_dir / f"tipping_point_analysis.svg")
            )

        # 2. 得分分布图
        if self.merged_df is not None:
            Visualizer.plot_score_distribution(
                self.merged_df,
                save_path=str(output_dir / f"score_distribution_{stage_str}.png")
            )

        # 3. 调控网络子图
        if self.subgraph_df is not None and self.merged_df is not None:
            Visualizer.plot_subgraph(
                self.subgraph_df,
                self.merged_df,
                save_path=str(output_dir / f"regulatory_network_{stage_str}.png"),
                top_n=100
            )

    def summary(self) -> str:
        """生成分析摘要"""
        if not self.tipping_result:
            return "尚未运行分析"

        t0 = self.tipping_result.best_t
        top_k = int(len(self.tf_scores) * self.config.top_percent)

        # 子图统计
        stats = self.integrator.get_subgraph_stats() if self.integrator else {}

        lines = [
            "\n" + "=" * 50,
            "分析摘要",
            "=" * 50,
            f"Tipping Point: 阶段 {t0+1} → {t0+2}",
            f"Bootstrap可靠性: {self.tipping_result.reliability*100:.1f}%",
            "",
            f"关键基因统计:",
            f"  - Top TF Drivers: {top_k}",
            f"  - Target Genes: {len(self.target_scores)}",
            f"  - 合并后总数: {len(self.merged_df)}",
            "",
            f"调控子图统计:",
            f"  - 节点数: {stats.get('num_nodes', 'N/A')}",
            f"  - 边数: {stats.get('num_edges', 'N/A')}",
            f"  - TF→TF边: {stats.get('tf_to_tf_edges', 'N/A')}",
            f"  - TF→Target边: {stats.get('tf_to_target_edges', 'N/A')}",
            "",
            f"Top 10 TF Drivers:",
            self.tf_scores.head(10)[['TF', 'AlignScore']].to_string(index=False),
            "",
            f"Top 10 Target Genes:",
            self.target_scores.head(10)[['Gene', 'ExecScore']].to_string(index=False),
            "",
            f"Top 5 Hub TFs (by out-degree):",
        ]

        if 'top_hubs' in stats:
            for gene, degree in stats['top_hubs'].items():
                lines.append(f"  - {gene}: {degree}")

        lines.append("=" * 50)
        return "\n".join(lines)


# =============================
# 入口点
# =============================
def main():
    """主入口函数"""
    config = Config(
        csv_file="merged_embedding_common_genes_diff2.csv",
        network_file="network_union_all_stages_diff2.csv",
        bootstrap_n=500,
        top_percent=0.08,
        smoothing_alpha=0.7,
        output_dir="results_diff2"
    )

    # 运行分析流程
    pipeline = MasterRegulatorPipeline(config)
    pipeline.run()

    # 输出摘要
    print(pipeline.summary())

    # 生成所有可视化（包括tipping point、得分分布、网络图）
    pipeline.visualize()


if __name__ == "__main__":
    main()