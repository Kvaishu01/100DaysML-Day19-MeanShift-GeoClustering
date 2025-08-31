import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

st.set_page_config(page_title="MeanShift - Geo Clustering", layout="centered")
st.title("📍 Day 19 — Mean Shift: Cluster Geographic Data")

# Synthetic GPS-like data (lat, lon)
@st.cache_data
def make_geo(n=300, seed=42):
    rng = np.random.RandomState(seed)
    centers = [
        (28.7041, 77.1025),  # Delhi
        (19.0760, 72.8777),  # Mumbai
        (12.9716, 77.5946),  # Bangalore
        (22.5726, 88.3639)   # Kolkata
    ]
    points = []
    for c in centers:
        lat = rng.normal(c[0], 0.2, n//len(centers))
        lon = rng.normal(c[1], 0.2, n//len(centers))
        points.append(np.column_stack([lat, lon]))
    pts = np.vstack(points)
    noise = np.column_stack([rng.uniform(8, 35, n//20), rng.uniform(68, 90, n//20)])
    data = np.vstack([pts, noise])
    return pd.DataFrame(data, columns=["lat", "lon"])

df = make_geo()
st.subheader("📂 Sample Geographic Points")
st.dataframe(df.sample(8))

# Bandwidth estimate & clustering
bandwidth = st.slider("Bandwidth (search radius)", 0.05, 1.0, 0.25)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(df.values)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

df["cluster"] = labels
n_clusters = len(np.unique(labels))
st.write(f"Found **{n_clusters}** clusters")

# Plot lat/lon clusters
fig, ax = plt.subplots(figsize=(7,6))
for c in np.unique(labels):
    subset = df[df["cluster"]==c]
    ax.scatter(subset["lon"], subset["lat"], s=30, label=f"cluster {c}")
ax.scatter(cluster_centers[:,1], cluster_centers[:,0], marker="X", s=200, c="black", label="centers")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Clusters (Mean Shift)")
ax.legend()
st.pyplot(fig)

st.success("✅ Mean Shift clustering completed")