import torch, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import matplotlib.patches as patches
from matplotlib.lines import Line2D


np.random.seed(100)

# -----------------------------
# Paths
# -----------------------------
feature_dir = "./content/drive/MyDrive/base-clip-data/models/250K_mse48_klclip1_jina_multiMpnet/features"
save_dir = "tsne_multilingual_plots2"
os.makedirs(save_dir, exist_ok=True)
eval_data_path = "./content/drive/MyDrive/base-clip-data/Cross-lingual-Test-Dataset-XTD10/merged_data.csv"

# Language embedding directories
lang_dirs = {
    "en": "XTD10_captions_en",
    "fr": "MIC_caption_fr",
    "de": "MIC_caption_de",
    "it": "XTD10_captions_it",
    "pl": "XTD10_captions_pl",
    "tr": "XTD10_captions_tr",
    "jp": "STAIR_caption_jp",
    "es": "XTD10_captions_es",
    "ko": "XTD10_captions_ko",
    "ru": "XTD10_captions_ru",
    "zh": "XTD10_captions_zh"
}

# -----------------------------
# 1. Load embeddings
# -----------------------------
def load_embedding(path):
    x = torch.load(path, map_location="cpu")
    # support dict-style or direct tensor
    if isinstance(x, dict):
        # try common keys
        for key in ["embeddings", "all_text", "all_text_clip"]:
            if key in x:
                x = x[key]
                break
    return x.numpy() if torch.is_tensor(x) else np.array(x)

# Image embeddings
image_embeddings = load_embedding(os.path.join(feature_dir, "all_images.pth"))
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
print("Images:", image_embeddings.shape)

# English embedding
en_embeddings = load_embedding(os.path.join(feature_dir, lang_dirs["en"], "all_text_clip.pth"))
en_embeddings = en_embeddings / np.linalg.norm(en_embeddings, axis=1, keepdims=True)

# Non-English embeddings
non_en_embeddings = {}
for lang, subdir in lang_dirs.items():
    #emb_path = os.path.join(feature_dir, subdir, "all_text_org.pth")
    emb_path = os.path.join(feature_dir, subdir, "all_text.pth")
    non_en_embeddings[lang] = load_embedding(emb_path)
    non_en_embeddings[lang] /= np.linalg.norm(non_en_embeddings[lang], axis=1, keepdims=True)

# -----------------------------
# 2. Image clustering to select diverse clusters
# -----------------------------
num_clusters = 100
K = 17
max_samples_per_cluster = 10

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(image_embeddings)
centroids = kmeans.cluster_centers_

# Farthest-point selection
cluster_sizes = np.bincount(cluster_labels)
valid_clusters = np.where(cluster_sizes >= 3)[0]  # clusters with at least 3 images
#remaining = set(valid_clusters)

top_clusters = np.argsort(cluster_sizes)[-50:]  # pick top 50 largest clusters
remaining = set(top_clusters)
# proceed with farthest-point sampling

# farthest-point selection among valid clusters only
selected_clusters = []
first = np.random.choice(list(remaining))
selected_clusters.append(first)
remaining.remove(first)

for _ in range(K-1):
    if len(remaining) == 0:
        break
    dists = pairwise_distances(centroids[list(remaining)], centroids[selected_clusters])
    min_dists = dists.min(axis=1)
    next_cluster = list(remaining)[np.argmax(min_dists)]
    selected_clusters.append(next_cluster)
    remaining.remove(next_cluster)


# Sample indices per selected cluster
selected_indices = []
for c in selected_clusters:
    idxs = np.where(cluster_labels == c)[0]
    if len(idxs) > max_samples_per_cluster:
        idxs = np.random.choice(idxs, size=max_samples_per_cluster, replace=False)
    selected_indices.extend(idxs.tolist())

print("secletd", selected_clusters)

remaining = [c for c in valid_clusters if c not in selected_clusters]
print("remanin", remaining)

remaining_indices = []
for c in remaining:
    idxs = np.where(cluster_labels == c)[0]
    if len(idxs) > max_samples_per_cluster:
        idxs = np.random.choice(idxs, size=max_samples_per_cluster, replace=False)
    remaining_indices.extend(idxs.tolist())


selected_indices = sorted(selected_indices)
print("Sampled images:", len(selected_indices))

remaining_indices = sorted(remaining_indices)
print("Sampled (rem) images:", len(remaining_indices))
# -----------------------------
# 3. Build t-SNE embedding set
# -----------------------------
all_embeddings = []
modalities = []  # marker type
clusters_for_plot = []

for idx in selected_indices:
    
    # Image
    '''
    all_embeddings.append(image_embeddings[idx])
    modalities.append("image")
    clusters_for_plot.append(cluster_labels[idx])
    '''

    # English text
    all_embeddings.append(en_embeddings[idx])
    modalities.append("en_clip")
    clusters_for_plot.append(cluster_labels[idx])

    # Other languages
    for lang, emb in non_en_embeddings.items():
        all_embeddings.append(emb[idx])
        modalities.append(lang)
        clusters_for_plot.append(cluster_labels[idx])

'''
for idx in remaining_indices:
    # English text
    all_embeddings.append(en_embeddings[idx])
    modalities.append("en_clip")
    clusters_for_plot.append(cluster_labels[idx])
'''


all_embeddings = np.stack(all_embeddings, axis=0)
print("Total points for t-SNE:", all_embeddings.shape[0])

# -----------------------------
# 4. t-SNE
# -----------------------------
ps = [32]
ppx = 32
tsne = TSNE(n_components=2, perplexity=ppx, random_state=42, init="pca", learning_rate="auto")
emb_2d = tsne.fit_transform(all_embeddings)
# -----------------------------
# 5. Plotting
# -----------------------------
df = pd.DataFrame({
    "x": emb_2d[:,0],
    "y": emb_2d[:,1],
    "modality": modalities,
    "cluster": clusters_for_plot
})

plt.figure(figsize=(12,10))
ax = plt.gca()
ax.set_aspect('equal', adjustable='datalim')

# Marker per cluster (shape)
unique_clusters = sorted(df["cluster"].unique())
markers = ["o","s","D","^","v","<",">","P","X","*"]
markers = ["o", "s", "x", "d", "^", "v", "<", ">", "p", "P", 
                   "X", "*", "h", "H", "+", "D"]
markers = ["o", "s", "D", "d", "^", "v", "<", ">", "p", "P", 
                   "X", "*", "h", "H", "+", "x", "."]

cluster_marker_map = {cid: markers[i % len(markers)] for i, cid in enumerate(unique_clusters)}

# Edge color per language

lang_color_map = {
    "en": "#1f77b4",  # blue
    "fr": "#2ca02c",  # green
    "de": "#9467bd",  # purple
    "it": "#17becf",  # cyan
    "pl": "#e377c2",  # pink
    "tr": "#bcbd22",  # olive
    "jp": "#8c564b",  # brown
    "es": "#7f7f7f",  # gray
    "ko": "#aec7e8",  # light blue
    "ru": "#98df8a",  # light green
    "zh": "#ffbb78",  # peach (soft, not red/orange)
}
'''
lang_color_map = {
    "en": "#1f77b4",  # bright blue
    "fr": "#2ca02c",  # bright green
    "de": "#9467bd",  # bright purple
    "it": "#17becf",  # bright cyan
    "pl": "#e377c2",  # bright pink
    "tr": "#bcbd22",  # bright lime
    "jp": "#8c564b",  # brown
    "es": "#7f7f7f",  # gray
    "ko": "#aec7e8",  # light blue
    "ru": "#98df8a",  # light green
    "zh": "#ffbf00",  # gold/yellow
}

lang_color_map = {
    "en": "#1f77b4",   # vivid blue
    "fr": "#2ca02c",   # vivid green
    "de": "#9467bd",   # vivid purple
    "it": "#17becf",   # bright cyan
    "pl": "#e377c2",   # bright pink
    "tr": "#bcbd22",   # lime green
    "jp": "#ff7f0e",   # bright orange-ish (avoid strong red)
    "es": "#8c564b",   # deep magenta / fuchsia
    "ko": "#d62728",   # bright red variant (if okay for language) → otherwise replace with teal "#1abc9c"
    "ru": "#7f7f7f",   # replace gray with bright yellow "#ffff00"
    "zh": "#17a2b8",   # teal / cyan
}

languages = ["en", "fr", "de", "it", "pl", "tr", "jp", "es", "ko", "ru", "zh"]
num_langs = len(languages)

# Get a vibrant colormap
cmap = plt.get_cmap("gist_rainbow")

# Skip the red-orange range (approx 0-0.08 in normalized cmap)
start, end = 0.08, 1.0  # start after red/orange
colors = [cmap(start + (end-start)*i/num_langs) for i in range(num_langs)]

# Build language_color_map
lang_color_map = {lang: colors[i] for i, lang in enumerate(languages)}
'''

lang_color_map = {
    "en": "#1f77b4",  # bright blue
    "fr": "#2ca02c",  # vivid green
    "de": "#9467bd",  # strong purple
    "it": "#17becf",  # cyan
    "pl": "#e377c2",  # hot pink
    "tr": "#bcbd22",  # lime
    "jp": "#7f7f7f",  # dark gray (neutral but still visible)
    "es": "#8c564b",  # earthy brown
    "ko": "#00c0a3",  # turquoise
    "ru": "#f781bf",  # magenta pink
    "zh": "#aec7e8",  # light blue
}


# Default fill (light gray)
fill_color = "black"
s = 200
L=0.5
# Scatter plot
for m in df["modality"].unique():
    m_df = df[df["modality"] == m]
    for cluster_id, c_df in m_df.groupby("cluster"):
        marker = cluster_marker_map[cluster_id]
        for _, row in c_df.iterrows():
            z = np.random.randint(1, 10)  # random zorder
            if m == "en_clip":
                # fully filled red
                plt.scatter(
                    row["x"], row["y"],
                    c="red", marker=marker,
                    s=s+20 if marker in ['x', '+'] else s, edgecolors="black",
                    linewidths=L, alpha=0.8,
                    label=None, zorder= z if np.random.choice([True, False]) else 500
                )
            else:
                edge_col = lang_color_map.get(m, "black")
                plt.scatter(
                    row["x"], row["y"],
                    c=edge_col, marker=marker,
                    s=s+20 if marker in ['x', '+'] else s, edgecolors="black",
                    linewidths=L, alpha=0.9,
                    label=None, zorder=z
                )
'''
# Annotate cluster numbers at centroids
for cluster_id in selected_clusters:
    cluster_points = df[df["cluster"] == cluster_id][["x","y"]].values
    centroid = cluster_points.mean(axis=0)
    plt.text(
        centroid[0], centroid[1], str(cluster_id),
        fontsize=20, fontweight="bold", color="black",
        ha="center", va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"),
        zorder=10000
    )
'''
# Legends
cluster_handles = [
    Line2D([0],[0], marker=cluster_marker_map[cid], color="gray", linestyle="None", label=f"Cluster {cid}")
    for cid in selected_clusters[:10]  # only show 10 clusters in legend
]
lang_handles = [
    Line2D([0],[0], marker="o", color=col, markerfacecolor=col,
           markeredgecolor=col, linestyle="None", label=lang)
    for lang, col in lang_color_map.items()
]
lang_handles.append(Line2D([0],[0], marker="o", color="red", linestyle="None",
                           markerfacecolor="red", label="en_clip"))

plt.legend(handles=lang_handles, title="Languages", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"tsne_shape_cluster_edge_lang_{ppx}.png"), bbox_inches="tight", dpi=300)

'''
for ppx in ps:
    print(f"Processing perplexity {ppx}")
    tsne = TSNE(n_components=2, perplexity=ppx, random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(all_embeddings)

    # -----------------------------
    # 5. Plotting
    # -----------------------------
    df = pd.DataFrame({
        "x": emb_2d[:,0],
        "y": emb_2d[:,1],
        "modality": modalities,
        "cluster": clusters_for_plot
    })

    plt.figure(figsize=(12,10))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    # Colors = cluster
    num_clusters_plot = df["cluster"].nunique()
    cmap = plt.get_cmap("tab20", num_clusters_plot)
    cluster_color_map = {cid: cmap(i) for i, cid in enumerate(sorted(df["cluster"].unique()))}

    # Marker per modality
    unique_modalities = sorted(df["modality"].unique())
    markers = ["o","s","D","^","v","<",">","P","X","*"]
    modality_marker_map = {m: markers[i%len(markers)] for i,m in enumerate(unique_modalities)}

    for m in unique_modalities:
        m_df = df[df["modality"]==m]
        s = 80 if m in ["image","en_clip"] else 50  # bigger for image & en_clip
        s = 60
        edge = "red" if m in ["image","en_clip"] else "black"  # highlight
        linewidths = 2.0 if m in ["image","en_clip"] else 0.3  # highlight
        alpha = 0.9 if m in ["image","en_clip"] else 0.6
        
        plt.scatter(
            m_df["x"], m_df["y"],
            c=[cluster_color_map[cid] for cid in m_df["cluster"]],
            marker=modality_marker_map[m],
            label=m,
            s=s,
            edgecolors=edge,
            linewidths=linewidths,
            alpha=alpha
        )

    # Annotate cluster numbers
    for cluster_id in selected_clusters:  # use only selected clusters
        cluster_points = df[df["cluster"] == cluster_id][["x","y"]].values
        centroid = cluster_points.mean(axis=0)
        plt.text(
            centroid[0], centroid[1], str(cluster_id),
            fontsize=12, fontweight='bold', color='black',
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
        )

    # Legend
    legend_handles = [
        Line2D([0],[0], marker=modality_marker_map[m], color="white", label=m,
               linestyle="None", markerfacecolor="gray", markersize=10,
               markeredgecolor="black", markeredgewidth=0.5)
        for m in unique_modalities
    ]
    plt.legend(handles=legend_handles, title="languages", bbox_to_anchor=(1.05,1), loc="upper left")
    #plt.title("t-SNE: Image + English + Multilingual Text Embeddings")
    #plt.xlabel("t-SNE Dim 1")
    #plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,f"tsne_image_text_multilingual_{ppx}.png"), bbox_inches='tight', dpi=300)
    plt.show()
'''
# -----------------------------
# 6. Print multilingual captions for sampled images
# -----------------------------
img_text_df = pd.read_csv(eval_data_path)
caption_cols = list(lang_dirs.values())

txt_path = os.path.join(save_dir,"sampled_captions_by_cluster.txt")

# Group selected indices by cluster
cluster_to_indices = {}
for idx in selected_indices:
    c = cluster_labels[idx]
    cluster_to_indices.setdefault(c, []).append(idx)

with open(txt_path, "w", encoding="utf-8") as f:
    for cluster_id in sorted(cluster_to_indices.keys()):
        f.write(f"\n=== Cluster {cluster_id} (size={len(cluster_to_indices[cluster_id])}) ===\n")
        for idx in cluster_to_indices[cluster_id]:
            f.write(f"\nImage {idx}:\n")
            for col in caption_cols:
                # Only write English captions
                if col != "XTD10_captions_en":
                    continue
                try:
                    val = img_text_df.loc[idx, col]
                    img_name = img_text_df.loc[idx, 'test_image_names']
                    if pd.notna(val):
                        f.write(f"{col} {img_name}: {val}\n")
                except KeyError as e:
                    raise e
            f.write("-" * 50 + "\n")

print("Saved clustered English captions at", txt_path)
