from pathlib import Path
import csv
import json

import nibabel as nib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from try_label_split import load_labels, propose_split, check_geometry


root = Path(
    input("Paste the path to the train folder: ").strip()
).expanduser()

patients = sorted(
    p for p in root.glob("Patient_*") if p.is_dir()
)

if not patients:
    raise SystemExit("No patient folders found.")

output = Path(__file__).resolve().parent / "label_split_review_v1"

if output.exists():
    raise SystemExit(
        "label_split_review_v1 already exists. "
        "Change the output name in this script before another run."
    )

output.mkdir()
rows = []

for patient in patients:
    print(f"Processing {patient.name}...", flush=True)

    try:
        image, gt = load_labels(patient / "GT.nii.gz")

        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError("Expected spatial units in millimetres.")

        if np.any(gt == 4):
            raise ValueError("Label 4 already exists; inspect separately.")

        # The splitting function does not receive GT2.
        candidate, details = propose_split(
            gt, image.header.get_zooms()[:3]
        )

        ct_image = nib.load(
            str(patient / f"{patient.name}.nii.gz")
        )
        check_geometry(image, ct_image)
        ct = np.asarray(ct_image.dataobj)

        if not np.isfinite(ct).all():
            raise ValueError("CT contains invalid numeric values.")

        patient_output = output / patient.name
        patient_output.mkdir()

        header = image.header.copy()
        header.set_data_dtype(np.uint8)

        result = nib.Nifti1Image(
            candidate, image.affine, header
        )

        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        result.set_qform(qform, int(qcode))
        result.set_sform(sform, int(scode))

        saved_path = patient_output / "GT_candidate.nii.gz"
        nib.save(result, str(saved_path))

        saved = nib.load(str(saved_path))
        check_geometry(image, saved)

        if not np.array_equal(
            np.asarray(saved.dataobj), candidate
        ):
            raise ValueError("Saved candidate differs from memory.")

        # Choose six slices spread across the merged region.
        occupied = np.flatnonzero(
            np.any(gt == 1, axis=(0, 1))
        )
        positions = np.linspace(
            0, len(occupied) - 1, 6
        ).round().astype(int)
        slices = np.unique(occupied[positions])

        low, high = np.percentile(ct, [1, 99])
        if high <= low:
            high = low + 1

        fig, axes = plt.subplots(
            len(slices), 3,
            figsize=(10, 3 * len(slices)),
            squeeze=False,
        )
        colours = ListedColormap(["cyan", "magenta"])

        for row_number, z in enumerate(slices):
            points = np.argwhere(gt[:, :, z] == 1)
            start = np.maximum(points.min(0) - 20, 0)
            stop = np.minimum(
                points.max(0) + 21, gt.shape[:2]
            )

            selection = (
                slice(start[0], stop[0]),
                slice(start[1], stop[1]),
                z,
            )

            for column in range(3):
                axis = axes[row_number, column]
                axis.imshow(
                    ct[selection].T,
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                    origin="lower",
                    interpolation="nearest",
                )

                if column > 0:
                    values = (
                        gt[selection] if column == 1
                        else candidate[selection]
                    )
                    display = np.zeros(
                        values.shape, dtype=np.uint8
                    )
                    display[values == 1] = 1
                    display[values == 4] = 2

                    axis.imshow(
                        np.ma.masked_where(
                            display.T == 0, display.T
                        ),
                        cmap=colours,
                        vmin=0.5,
                        vmax=2.5,
                        alpha=0.45,
                        origin="lower",
                        interpolation="nearest",
                    )

                title = [
                    "CT", "Original merged GT", "Candidate"
                ][column]
                axis.set_title(f"Slice {z}: {title}")
                axis.set_aspect(
                    float(
                        image.header.get_zooms()[1]
                        / image.header.get_zooms()[0]
                    )
                )
                axis.axis("off")

        fig.suptitle(
            f"{patient.name}: candidate requires review\n"
            "Cyan = label 1; magenta = label 4. "
            "Colours are proposed assignments."
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(
            patient_output / "review.png", dpi=120
        )
        plt.close(fig)

        (patient_output / "method_details.json").write_text(
            json.dumps(details, indent=2)
        )

        rows.append({
            "patient": patient.name,
            "status": "generated_needs_review",
            "basins": len(details["basins"]),
            "label_1_voxels": int(
                np.count_nonzero(candidate == 1)
            ),
            "label_4_voxels": int(
                np.count_nonzero(candidate == 4)
            ),
            "review_notes": "",
        })

    except Exception as error:
        plt.close("all")
        print(f"  CHECK REQUIRED: {error}", flush=True)
        rows.append({
            "patient": patient.name,
            "status": "failed_do_not_use",
            "basins": "",
            "label_1_voxels": "",
            "label_4_voxels": "",
            "review_notes": str(error),
        })

with (output / "review_summary.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nFinished. Review results in: {output}")
print("Original annotations were not changed.")