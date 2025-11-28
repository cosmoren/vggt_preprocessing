from pathlib import Path

def main():
    original_dataset_root = "/mnt/nfs/SpatialAI/wai_datasets"
    output_dir = "/mnt/nfs/SpatialAI/sq/optical_flow_datavggt_ft_optical_flow"
    with open("/mnt/nfs/SpatialAI/sq/optical_flow_datavggt_ft_optical_flow/mvs_synth/ransac_failures.txt", "r") as f:
        failures = f.readlines()
    
    unfound_count = 0
    for failure in failures:
        pt_path = Path(failure.strip().replace(original_dataset_root, output_dir).replace(".png", ".pt").replace("/images", ""))
        if pt_path.exists():
            pt_path.unlink()
            print(f"Deleted {pt_path}")
        else:
            unfound_count += 1
            print(f"PT file not found: {pt_path}")
    print(f"Deleted {len(failures) - unfound_count} PT files")
if __name__ == "__main__":
    main()