from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path("data/segthor_part1/data/segthor_part1/train")

for patient_dir in sorted(ROOT.glob("Patient_*")):

    patient = patient_dir.name
    ct_path = patient_dir / f"{patient}.nii.gz"
    gt_path = patient_dir / "GT.nii.gz"

    ct = nib.load(str(ct_path))
    gt = nib.load(str(gt_path))

    ct_data = ct.get_fdata()
    gt_data = gt.get_fdata().astype(np.int16)

    print(f"\n{patient}")
    print("CT shape:", ct_data.shape)
    print("CT spacing:", ct.header.get_zooms()[:3])
    print(
        "CT range:",
        float(ct_data.min()),
        float(ct_data.max())
    )

    labels, counts = np.unique(gt_data, return_counts=True)

    print("GT:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count:,} voxels")

    print("GT voxel fractions:")
    total = gt_data.size
    for label, count in zip(labels, counts):
        print(f"  {label}: {count / total:.6f}")