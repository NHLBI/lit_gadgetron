#!/usr/bin/env python3
"""Update an existing NHLBI integration test case.

Supports updating data files, config, thresholds, and GPU/memory requirements.

Usage:
    python update_test.py imoco_vds --data-file /path/to/new_recon_data.h5
    python update_test.py imoco_vds --noise-file /path/to/new_noise.h5
    python update_test.py imoco_vds --config new_imoco_config.xml
    python update_test.py imoco_vds --value-threshold 0.05 --scale-threshold 0.05
    python update_test.py imoco_vds --gpu-memory 16384
    python update_test.py imoco_vds --regenerate-baseline
"""

import argparse
import configparser
import os
import sys
from pathlib import Path

import h5py

from get_nhlbi_data import (
    calc_sha256,
    get_container_client,
    load_manifest,
    save_manifest,
    upload_blob,
)

CASES_DIR = Path(__file__).parent / "cases"


def main():
    parser = argparse.ArgumentParser(
        description="Update an existing NHLBI integration test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('name', help="Test case name to update")
    parser.add_argument('--data-file', help="New reconstruction input HDF5 file")
    parser.add_argument('--noise-file', help="New noise calibration HDF5 file")
    parser.add_argument('--config', help="New Gadgetron XML config for reconstruction")
    parser.add_argument('--noise-config', help="New Gadgetron XML config for noise dependency")
    parser.add_argument('--description', help="Updated test description")
    parser.add_argument('--value-threshold', type=float, help="New value comparison threshold")
    parser.add_argument('--scale-threshold', type=float, help="New scale comparison threshold")
    parser.add_argument('--gpu-memory', type=int, help="New GPU memory requirement in MB")
    parser.add_argument('--system-memory', type=int, help="New system memory requirement in MB")
    parser.add_argument('--tags', help="New comma-separated tags (nhlbi is always included)")
    parser.add_argument('--regenerate-baseline', action='store_true',
                        help="Invalidate current baseline and prompt for regeneration")
    parser.add_argument('--skip-upload', action='store_true',
                        help="Skip uploading new files to Azure")

    args = parser.parse_args()
    name = args.name

    # Validate test exists
    cfg_path = CASES_DIR / f"{name}.cfg"
    if not cfg_path.exists():
        print(f"Error: Test '{name}' not found at {cfg_path}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(cfg_path)
    manifest = load_manifest()
    changes = []

    # Update data file
    if args.data_file:
        if not os.path.isfile(args.data_file):
            print(f"Error: Data file not found: {args.data_file}")
            sys.exit(1)
        try:
            h5py.File(args.data_file, 'r').close()
        except Exception as e:
            print(f"Error: Cannot read HDF5 file: {e}")
            sys.exit(1)

        sha256 = calc_sha256(args.data_file)
        if not args.skip_upload:
            container_client = get_container_client()
            upload_blob(container_client, args.data_file, f"{name}/recon_data.h5")

        # Update manifest
        for entry in manifest:
            if entry.get('test') == name and entry.get('type') == 'input':
                entry['sha256'] = sha256
                break
        else:
            manifest.append({'file': f'{name}/recon_data.h5', 'sha256': sha256, 'type': 'input', 'test': name})

        changes.append("Updated reconstruction data file")

    # Update noise file
    if args.noise_file:
        if not os.path.isfile(args.noise_file):
            print(f"Error: Noise file not found: {args.noise_file}")
            sys.exit(1)
        try:
            h5py.File(args.noise_file, 'r').close()
        except Exception as e:
            print(f"Error: Cannot read HDF5 file: {e}")
            sys.exit(1)

        sha256 = calc_sha256(args.noise_file)
        if not args.skip_upload:
            container_client = get_container_client()
            upload_blob(container_client, args.noise_file, f"{name}/noise_data.h5")

        for entry in manifest:
            if entry.get('test') == name and entry.get('type') == 'noise':
                entry['sha256'] = sha256
                break
        else:
            manifest.append({'file': f'{name}/noise_data.h5', 'sha256': sha256, 'type': 'noise', 'test': name})

        changes.append("Updated noise data file")

    # Update reconstruction config
    if args.config:
        config['reconstruction.client']['configuration'] = args.config
        config['reconstruction.test']['reference_images'] = f'{args.config}/image_0'
        config['reconstruction.test']['output_images'] = f'{args.config}/image_0'
        changes.append(f"Updated reconstruction config to {args.config}")

    # Update noise config
    if args.noise_config:
        config['dependency.client']['configuration'] = args.noise_config
        changes.append(f"Updated noise config to {args.noise_config}")

    # Update thresholds
    if args.value_threshold is not None:
        config['reconstruction.test']['value_comparison_threshold'] = str(args.value_threshold)
        changes.append(f"Updated value threshold to {args.value_threshold}")

    if args.scale_threshold is not None:
        config['reconstruction.test']['scale_comparison_threshold'] = str(args.scale_threshold)
        changes.append(f"Updated scale threshold to {args.scale_threshold}")

    # Update requirements
    if args.gpu_memory is not None:
        config['requirements']['gpu_memory'] = str(args.gpu_memory)
        changes.append(f"Updated GPU memory requirement to {args.gpu_memory} MB")

    if args.system_memory is not None:
        config['requirements']['system_memory'] = str(args.system_memory)
        changes.append(f"Updated system memory requirement to {args.system_memory} MB")

    # Update tags
    if args.tags is not None:
        tag_list = ['nhlbi'] + [t.strip() for t in args.tags.split(',') if t.strip()]
        config['tags']['tags'] = ','.join(tag_list)
        changes.append(f"Updated tags to {','.join(tag_list)}")

    # Update description
    if args.description is not None:
        config['nhlbi']['description'] = args.description
        changes.append("Updated description")

    # Regenerate baseline
    if args.regenerate_baseline:
        # Remove baseline entry from manifest
        manifest = [e for e in manifest if not (e.get('test') == name and e.get('type') == 'baseline')]
        changes.append("Invalidated baseline")

    if not changes:
        print("No changes specified. Use --help for options.")
        sys.exit(0)

    # Write changes
    with open(cfg_path, 'w') as f:
        config.write(f)
    save_manifest(manifest)

    print(f"Updated test '{name}':")
    for change in changes:
        print(f"  - {change}")

    if args.regenerate_baseline or args.data_file or args.noise_file or args.config:
        print(f"\nReminder: Run 'python generate_baseline.py --test {name}' to regenerate baseline")


if __name__ == '__main__':
    main()
