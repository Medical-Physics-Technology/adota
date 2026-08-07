"""Build UUID -> (anatomy, patient/CT key) map for the trainset, from source metadata."""
import os, json
import pandas as pd

R = "/RadiotherapyData/dataset_v0"
SRC = {"trainset_pelvis": "pelvic_abdominal", "initial_test_one_ct": "thorax"}
OUT = "/tmp/claude-634202/-home-mstryja-projects-adota/d51f677e-63c0-4b1d-b867-bdccd2d5e730/scratchpad/uuid_provenance_map.csv"

rows = []
for folder, anat in SRC.items():
    d0 = os.path.join(R, folder)
    n = 0
    with os.scandir(d0) as it:
        for e in it:
            if not e.name.endswith("_metadata.json"):
                continue
            uid = e.name[: -len("_metadata.json")]
            try:
                d = json.load(open(e.path))
                sz = tuple(d.get("image_size", []))
                og = tuple(round(x, 1) for x in d.get("image_origin", []))
                pkey = f"{anat}|{sz}|{og}"
            except Exception:
                pkey = f"{anat}|NA"
            rows.append((uid, anat, pkey))
            n += 1
    print(f"{folder} ({anat}): {n} records", flush=True)

m = pd.DataFrame(rows, columns=["sample_id", "anatomy", "patient_key"])
m.to_csv(OUT, index=False)
print("distinct patient_keys per anatomy:")
print(m.groupby("anatomy")["patient_key"].nunique())

res = pd.read_csv("/scratch/mstryja/adota_runs/20260707_124010/results.csv")[["sample_id"]]
mg = res.merge(m, on="sample_id", how="left")
print("\nresults.csv coverage:", round(mg.anatomy.notna().mean(), 4),
      " unmatched:", int(mg.anatomy.isna().sum()))
print("anatomy counts (results.csv):")
print(mg.anatomy.value_counts(dropna=False))
print("\n#patients (CTs) represented in results.csv per anatomy:")
print(mg.dropna().groupby("anatomy")["patient_key"].nunique())
print("beamlets per patient: median",
      int(mg.dropna().groupby("patient_key").size().median()),
      " min", int(mg.dropna().groupby("patient_key").size().min()),
      " max", int(mg.dropna().groupby("patient_key").size().max()))
print("saved:", OUT)
