import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler


plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ===============================
# 参数与数据加载
# ===============================
CSV_FILE = "merged_embedding_common_genes_diff2.csv"
TOP_GENES_FILE = "results_diff2/Merged_key_genes_stage2_to_3.csv"
NUM_TIMEPOINTS = 4
EMBEDDING_DIM = 512

# 暖色光晕样式：集中放在这里，方便后续微调
STAR_COLOR = "#C62828"      # 深红色核心星标
STAR_SIZE = 650              # 中心红星（较原设置放大约 24%）
STAR_EDGE_WIDTH = 1.8

df = pd.read_csv(CSV_FILE, index_col=0)
top_genes_df = pd.read_csv(TOP_GENES_FILE)
top_genes = top_genes_df["Gene"].tolist()
num_top = len(top_genes)

X_top = df.loc[top_genes].values.reshape(
    NUM_TIMEPOINTS, num_top, EMBEDDING_DIM
)
scaler = StandardScaler()
X_top_std = scaler.fit_transform(
    X_top.reshape(-1, EMBEDDING_DIM)
).reshape(NUM_TIMEPOINTS, num_top, EMBEDDING_DIM)

X_2d = np.zeros((NUM_TIMEPOINTS, num_top, 2))
for t in range(NUM_TIMEPOINTS):
    pca = PCA(n_components=2)
    X_2d[t] = pca.fit_transform(X_top_std[t])

energy = np.linalg.norm(X_top_std, axis=2) ** 2

x_grid = np.linspace(
    X_2d[:, :, 0].min() - 2,
    X_2d[:, :, 0].max() + 2,
    100,
)
y_grid = np.linspace(
    X_2d[:, :, 1].min() - 2,
    X_2d[:, :, 1].max() + 2,
    100,
)
xx, yy = np.meshgrid(x_grid, y_grid)


def compute_landscape(xs, ys, zs):
    xy = np.vstack([xs, ys])
    kde = gaussian_kde(xy, weights=zs)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz = (zz - zz.min()) / (zz.max() - zz.min())
    return -(zz ** 1.2)


def add_star_on_top(ax, x, y, z):
    """将三维谷底投影到二维画布，并在最上层绘制红星。"""
    # 三维数据坐标 -> 当前视角下的二维投影坐标
    x_proj, y_proj, _ = proj3d.proj_transform(x, y, z, ax.get_proj())

    # 投影坐标 -> 像素坐标 -> Axes 比例坐标
    x_display, y_display = ax.transData.transform((x_proj, y_proj))
    x_axes, y_axes = ax.transAxes.inverted().transform((x_display, y_display))

    # 这里使用二维 transform，因此不参与 mplot3d 的深度排序，永远盖在曲面上方。
    # 红星保留白色描边，确保与复杂曲面分离。
    ax.add_artist(Line2D(
        [x_axes], [y_axes],
        marker="*",
        linestyle="none",
        markersize=sqrt(STAR_SIZE),
        markerfacecolor=STAR_COLOR,
        markeredgecolor="white",
        markeredgewidth=STAR_EDGE_WIDTH,
        alpha=1.0,
        transform=ax.transAxes,
        clip_on=False,
        zorder=1000,
    ))


fig = plt.figure(figsize=(24, 10))
basin_path = []
basin_markers = []
last_surface = None

for t in range(NUM_TIMEPOINTS):
    ax = fig.add_subplot(2, 3, t + 1, projection="3d")
    xs, ys = X_2d[t, :, 0], X_2d[t, :, 1]

    surface = compute_landscape(xs, ys, energy[t])
    last_surface = surface

    min_idx = np.unravel_index(np.argmin(surface), surface.shape)
    cx, cy, cz = xx[min_idx], yy[min_idx], surface[min_idx]
    basin_path.append([cx, cy, cz])

    ax.plot_surface(
        xx,
        yy,
        surface,
        cmap="terrain",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    ax.contour(
        xx,
        yy,
        surface,
        zdir="z",
        offset=surface.min() - 0.2,
        cmap="coolwarm",
        alpha=0.5,
    )

    ax.scatter(
        xs,
        ys,
        surface.min() - 0.1,
        c="black",
        s=10,
        alpha=0.3,
        depthshade=False,
    )

    # 暂存谷底位置；完成布局和视角设置后再绘制最上层二维标记。
    basin_markers.append((ax, cx, cy, cz))

    ax.set_title(f"Stage {t + 1} Landscape", fontsize=26)
    ax.set_zlim(surface.min() - 0.3, 0.2)
    ax.view_init(elev=35, azim=125)

    ax.set_xlabel("PC1", fontsize=20, labelpad=10)
    ax.set_ylabel("PC2", fontsize=20, labelpad=10)
    ax.set_zlabel("Potential", fontsize=20, labelpad=10)

    ax.tick_params(axis="x", labelsize=16, pad=2)
    ax.tick_params(axis="y", labelsize=16, pad=2)
    ax.tick_params(axis="z", labelsize=16, pad=2)

# 使用代理对象构造简洁图例
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="black",
        markeredgecolor="black",
        alpha=0.3,
        markersize=7,
        label="Gene Projections",
    ),
    Line2D(
        [0], [0],
        marker="*",
        linestyle="none",
        markerfacecolor=STAR_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.8,
        markersize=18,
        label="Energy Basin",
    ),
]

legend_ax = fig.add_subplot(2, 3, 6)
legend_ax.axis("off")
legend_ax.legend(
    handles=legend_handles,
    loc="center",
    fontsize=26,
    frameon=True,
    markerscale=1.0,
)

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
norm = plt.Normalize(vmin=last_surface.min(), vmax=0.2)
sm = plt.cm.ScalarMappable(cmap="terrain", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(
    "Potential Energy (Landscape Depth)",
    fontsize=26,
    fontweight="bold",
)
cbar.ax.tick_params(labelsize=24)

plt.subplots_adjust(right=0.9)

# 必须先完成布局并触发一次绘制，投影矩阵和坐标变换才是最终状态。
fig.canvas.draw()
for marker_ax, marker_x, marker_y, marker_z in basin_markers:
    add_star_on_top(marker_ax, marker_x, marker_y, marker_z)

plt.savefig("gene_energy_landscape_diff2_warm_halo.pdf", dpi=300)
plt.savefig("gene_energy_landscape_diff2_warm_halo.svg", dpi=300)
plt.show()
