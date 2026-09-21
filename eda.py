from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("data/segthor_part1/data/segthor_part1/train")

patients = []
num_slices = []
spacing_x = []
spacing_y = []
spacing_z = []
ct_min = []
ct_max = []

class_fractions = {
    0: [],
    1: [],
    2: [],
    3: []
}

for patient_dir in sorted(ROOT.glob("Patient_*")):

    patient = patient_dir.name

    ct_path = patient_dir / f"{patient}.nii.gz"
    gt_path = patient_dir / "GT.nii.gz"

    ct = nib.load(str(ct_path))
    gt = nib.load(str(gt_path))

    ct_data = ct.get_fdata()
    gt_data = gt.get_fdata().astype(np.int16)

    patients.append(patient)
    num_slices.append(ct_data.shape[2])

    spacing = ct.header.get_zooms()[:3]
    spacing_x.append(spacing[0])
    spacing_y.append(spacing[1])
    spacing_z.append(spacing[2])

    ct_min.append(float(ct_data.min()))
    ct_max.append(float(ct_data.max()))

    labels, counts = np.unique(gt_data, return_counts=True)

    total = gt_data.size

    for label in range(4):
        if label in labels:
            idx = np.where(labels == label)[0][0]
            class_fractions[label].append(counts[idx] / total)
        else:
            class_fractions[label].append(0)


# Graph 1
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(patients))

for i in range(len(patients)):
    ax.plot(
        [x[i], x[i]],
        [ct_min[i], ct_max[i]],
        linewidth=3
    )

    ax.scatter(x[i], ct_min[i], s=45)
    ax.scatter(x[i], ct_max[i], s=45)

ax.set_xticks(x)
ax.set_xticklabels(patients, rotation=45)
ax.set_ylabel("HU")
ax.set_title("CT Intensity Range Across Patients")
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# Graph 2
fig, ax = plt.subplots(figsize=(9, 6))

data = [
    np.array(class_fractions[1]) * 100,
    np.array(class_fractions[2]) * 100,
    np.array(class_fractions[3]) * 100
]

parts = ax.violinplot(
    data,
    positions=[1, 2, 3],
    showmeans=True,
    showmedians=True
)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels([
    "Aorta + Esophagus",
    "Heart",
    "Trachea"
])

ax.set_ylabel("Percentage of voxels (%)")
ax.set_title("Variation in Organ Volume Across Patients")

ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

# Graph 3
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    spacing_z,
    num_slices,
    s=90
)

for i, patient in enumerate(patients):
    ax.annotate(
        patient.replace("Patient_", "P"),
        (spacing_z[i], num_slices[i]),
        xytext=(5, 5),
        textcoords="offset points"
    )

ax.set_xlabel("Slice thickness (mm)")
ax.set_ylabel("Number of axial slices")
ax.set_title("Variation in CT Acquisition Geometry")

ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()

# Graph 4

from matplotlib.colors import Normalize

features = np.column_stack([
    num_slices,
    spacing_x,
    spacing_y,
    spacing_z,
    ct_min,
    ct_max,
    np.array(class_fractions[0]) * 100,
    np.array(class_fractions[1]) * 100,
    np.array(class_fractions[2]) * 100,
    np.array(class_fractions[3]) * 100
])

feature_names = [
    "Slices",
    "Spacing X",
    "Spacing Y",
    "Spacing Z",
    "CT Min",
    "CT Max",
    "Background %",
    "Aorta+Eso %",
    "Heart %",
    "Trachea %"
]

# Normalize each feature to 0-1
features_norm = np.zeros_like(features, dtype=float)

for j in range(features.shape[1]):
    col = features[:, j]
    min_val = col.min()
    max_val = col.max()

    if max_val > min_val:
        features_norm[:, j] = (col - min_val) / (max_val - min_val)

fig, ax = plt.subplots(figsize=(12, 8))

im = ax.imshow(features_norm, aspect="auto")

ax.set_yticks(np.arange(len(patients)))
ax.set_yticklabels(patients)

ax.set_xticks(np.arange(len(feature_names)))
ax.set_xticklabels(
    feature_names,
    rotation=45,
    ha="right"
)

ax.set_title("Patient-Level Dataset Fingerprint")

fig.colorbar(
    im,
    ax=ax,
    label="Normalized feature value"
)

plt.tight_layout()
plt.show()