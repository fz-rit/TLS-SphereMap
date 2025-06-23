"""
Point-cloud label-refinement.
1. reads point features (.csv) and labels (.label)
2. applies spatial majority voting (k-NN) to smooth obvious errors
3. trains a Random-Forest on reliable points to fix stubborn mislabels
4. returns a clean DataFrame you can inspect or write back to disk.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tqdm
from tools.logger_setup import Logger

log = Logger()
# ────────────────────────────────────────────────────────────────
# 1. I/O
# ────────────────────────────────────────────────────────────────
def read_pc_and_labels(csv_path, label_path, *,
                       feature_cols=None, dtype=np.float32):
    """
    Parameters
    ----------
    csv_path   : str | Path  - point feature file
    label_path : str | Path  - .label file (uint32 per point)
    feature_cols : list[str] - optional explicit list of columns
    dtype      : np.dtype    - numeric dtype for features
    Returns
    -------
    pd.DataFrame with features + 'label' column (uint32)
    """
    csv_path, label_path = map(Path, (csv_path, label_path))

    if feature_cols is None:
        # assume header present
        df = pd.read_csv(csv_path).astype(dtype)
    else:
        df = pd.read_csv(csv_path, sep=',')
        if not set(feature_cols).issubset(df.columns):
            raise ValueError(f"Feature columns {feature_cols} not found in CSV.")
        df = df[feature_cols]

    labels = np.loadtxt(label_path, dtype=int)
    if len(labels) != len(df):
        raise ValueError(f"Label count {len(labels)} ≠ point count {len(df)}")

    df["label"] = labels
    return df


# ────────────────────────────────────────────────────────────────
# 2. Spatial majority-voting helpers
# ────────────────────────────────────────────────────────────────

def _vote_labels(df, k=16, majority_thresh=0.6, xyz_cols=("X", "Y", "Z"),
                 classes=6):
    """
    One-pass k-NN majority voting.

    Parameters
    ----------
    df      : DataFrame with xyz + 'label'
    k       : int   - neighbours queried (incl. the point itself)
    majority_thresh : float - if own-class support < thresh → relabel
    classes : int   - number of classes (0 … classes-1)

    Returns
    -------
    np.ndarray of new labels (same length as df)
    """
    tree = cKDTree(df[list(xyz_cols)].values)
    labels = df["label"].to_numpy()
    out    = labels.copy()

    # use tqdm for progress bar
    log.info(f"Running k-NN voting with k={k} and majority threshold={majority_thresh:.2f}")
    for i in tqdm.tqdm(range(len(df)), desc="Voting labels"):
        point_xyz = df.loc[i, list(xyz_cols)].values
        lbl = int(labels[i])
        dists, idxs = tree.query(point_xyz, k=k)
        neigh_labels = labels[idxs]

        # counts length ≥ number of classes actually present
        max_lbl = max(neigh_labels.max(), classes - 1)
        counts  = np.bincount(neigh_labels, minlength=max_lbl + 1)

        # ignore class 0 when picking dominant label
        dominant = counts[1:].argmax() + 1

        # ---------- decision rules ----------
        if lbl == 0:                          # unlabeled → adopt dominant
            if counts[1:].sum():
                out[i] = dominant
        else:                                 # already labeled
            support = counts[lbl] / k
            if support < majority_thresh:
                out[i] = dominant
    return out


# ────────────────────────────────────────────────────────────────
# 3. Random-Forest correction helpers
# ────────────────────────────────────────────────────────────────
def _train_rf(df, reliable_mask, feature_cols, n_estimators=200):
    X = df.loc[reliable_mask, feature_cols].values
    y = df.loc[reliable_mask, "label"].values
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", n_jobs=-1
    )
    clf.fit(Xtr, ytr)
    log.info(f"RF validation accuracy: {clf.score(Xval, yval):.3f}")
    return clf

def _rf_predict(df, clf, feature_cols):
    proba = clf.predict_proba(df[feature_cols].values)
    pred  = clf.classes_[proba.argmax(1)]
    conf  = proba.max(1)
    return pred, conf


# ────────────────────────────────────────────────────────────────
# 4. Public pipeline (glues everything together)
# ────────────────────────────────────────────────────────────────
def refine_labels(df, *, feature_cols, k=16,
                  majority_thresh=0.6, use_rf=True, conf_thresh=0.8):
    """
    Parameters
    ----------
    df : pd.DataFrame  - must contain feature_cols + 'label'
    feature_cols : list[str]
    k, majority_thresh : voting parameters
    use_rf : bool       - train RF on reliable pts
    conf_thresh : float - RF confidence override
    Returns
    -------
    df_out : pd.DataFrame (deep copy) with column 'label_refined'
    """
    df = df.copy(deep=True)

    # 4-1 spatial smoothing
    df["label_vote"] = _vote_labels(df, k=k, majority_thresh=majority_thresh)

    if not use_rf:
        df["label_refined"] = df["label_vote"]
        return df

    # 4-2 RF on reliable points
    reliable = (df["label"] != 0) & (df["label"] == df["label_vote"])
    clf = _train_rf(df, reliable, feature_cols)
    pred, conf = _rf_predict(df, clf, feature_cols)

    # 4-3 merge rules
    mask = (df["label"] == 0) | ((conf > conf_thresh) & (df["label"] != pred))
    df.loc[mask, "label_vote"] = pred[mask]  # overwrite
    df["label_refined"] = df["label_vote"]


    # compare original and refined labels
    label_counts = df["label"].value_counts().sort_index()
    refined_counts = df["label_refined"].value_counts().sort_index()
    diff_counts = label_counts - refined_counts
    log.info("Label counts before and after refinement:")
    log.info("Original labels:\n" + str(label_counts))
    log.info("Refined labels:\n" + str(refined_counts))
    log.info("Difference (original - refined):\n" + str(diff_counts))
    log.info(f"Refinement complete. {mask.sum()} points were relabeled.")

    return df

# make histograms (bar chart) of labels (blue) and refined labels (red), in one plot
def plot_label_histograms(df, save_path=None):
    """
    Plot histograms of original and refined labels.

    Parameters
    ----------
    df : pd.DataFrame - must contain 'label' and 'label_refined'
    save_path : str | Path - path to save the plot
    """
    import matplotlib.pyplot as plt

    label_counts = df["label"].value_counts().sort_index()
    refined_counts = df["label_refined"].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    bars1 = plt.bar(label_counts.index - 0.2, label_counts.values, width=0.4, label='Original', color='blue', alpha=0.7)
    bars2 = plt.bar(refined_counts.index + 0.2, refined_counts.values, width=0.4, label='Refined', color='red', alpha=0.7)

    # Add count values on top of bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(label_counts.values),
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(refined_counts.values),
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Label Histograms')
    plt.xticks(range(max(label_counts.index.max(), refined_counts.index.max()) + 1))
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        log.info(f"Saved histogram to {save_path}")
    
    plt.show()
    


def main():
    # 1. read data
    root_dir = Path("/home/fzhcis/mylab/gdrive/projects_with_Jan/point_cloud_segmentation/unwrap_outputs/palau_2024")
    K = 50
    majority_thresh = 0.8
    output_formats = [".label"]
    plot_hist = False
    label_paths = list(root_dir.glob('**/*seg.label'))
    csv_paths = list(root_dir.glob('**/*_color.csv'))
    label_paths.sort()
    csv_paths.sort()
    log.info(f"Found {len(label_paths)} label files and {len(csv_paths)} CSV files.")
    for label_path, csv_path in zip(label_paths[2:], csv_paths[2:]):
        pcd_dir = label_path.parent
        log.info(f"Processing {pcd_dir.name} ...")
        
        log.info(f"Reading features from {csv_path.stem}")
        log.info(f"Reading labels from {label_path.stem}")
        features = ['X','Y','Z','nx','ny','nz',
                    # 'Intensity', 'Range',
                    'curvature','anisotropy','planarity']
        df = read_pc_and_labels(csv_path,
                                label_path,
                                feature_cols=features)

        # 2. run refinement
        df_clean = refine_labels(df, feature_cols=features,
                                k=K, majority_thresh=majority_thresh,
                                use_rf=True, conf_thresh=0.85)

        # 3. write outputs
        for output_format in output_formats:
            if output_format == ".csv":
                output_path = pcd_dir / (str(label_path.stem.split('_seg')[0])+"_refined.csv")
                df_clean[["X", "Y", "Z", "label", "label_refined"]].to_csv(output_path, index=False)
            elif output_format == ".label":
                output_path = pcd_dir / (str(label_path.stem.split('_seg')[0])+"_refined.label")
                df_clean["label_refined"].to_csv(output_path, index=False, header=False)
            else:
                raise ValueError(f"Unsupported output file type: {output_format}; must be .csv or .label")
        if plot_hist:
            log.info("Plotting label histograms...")
            plot_label_histograms(df_clean, save_path=pcd_dir / "label_histograms.png")
    

if __name__ == "__main__":
    main()