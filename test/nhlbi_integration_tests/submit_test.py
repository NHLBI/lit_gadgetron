#!/usr/bin/env python3
"""Register a new NHLBI integration test case.

Uploads data files to private Azure Blob Storage, computes SHA256 checksums,
creates a .cfg test case file, and updates the data manifest.

Usage:
    python submit_test.py \\
        --name imoco_vds \\
        --config imoco_recon_vds.xml \\
        --noise-file /path/to/noise.h5 \\
        --data-file /path/to/recon_data.h5 \\
        --noise-config default_measurement_dependencies.xml \\
        --description "iMOCO VDS 3D lung reconstruction" \\
        --gpu-memory 8192
"""

import argparse
import configparser
import os
import sys
from pathlib import Path
import os.path as op

import h5py
from test_utils import get_gadgetron_config_path
from get_nhlbi_data import (
    calc_sha256,
    get_container_client,
    load_manifest,
    save_manifest,
    upload_blob,
)

CASES_DIR = Path(__file__).parent / "cases"


def validate_hdf5(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            pass
        return True
    except Exception as e:
        print(f"Error: Cannot read HDF5 file {filepath}: {e}")
        return False


def validate_config_exists(config_name):
    """Check if the XML config exists in common gadgetron config locations."""
    search_paths = [
        Path(get_gadgetron_config_path()) / config_name,
        Path("config") / config_name,
        Path("config/config") / config_name,
    ]
    for p in search_paths:
        if p.exists():
            return True
    print(f"Warning: Config '{config_name}' not found in standard locations. "
          f"Ensure it is installed before running the test.")
    return True  # Warning only, don't block submission


def create_cfg(name, config, noise_config, description, gpu_memory, system_memory,
               value_threshold, scale_threshold, tags,optional_additional_datasets=[],optional_additional_dependency_datasets=[]):
    cfg = configparser.ConfigParser()

    cfg['dependency.siemens'] = {
        'data_file': f'{name}/noise_data.h5',
        'measurement': '0',
        'additional_arguments': 'skip_converstion',
    }
    cfg['dependency.client'] = {
        'configuration': noise_config,
    }
    for i in range (len(optional_additional_dependency_datasets)//2):
        cfg[f'dependency.siemens.{i+1}'] = {
        'data_file': f'{name}/{op.basename(optional_additional_dependency_datasets[2*i])}',
        'measurement': '0',
        'additional_arguments': 'skip_converstion',
        }
        cfg[f'dependency.client.{i+1}'] = {
        'configuration': optional_additional_dependency_datasets[2*i+1],
    }
        
    cfg['reconstruction.siemens'] = {
        'data_file': f'{name}/recon_data.h5',
        'measurement': '0',
        'additional_arguments': 'skip_converstion',
    }
    cfg['reconstruction.client'] = {
        'configuration': config,
    }
    cfg['reconstruction.test'] = {
        'reference_file': f'{name}/baseline_output.h5',
        'reference_images': f'{config}/image_0',
        'output_images': f'{config}/image_0',
        'value_comparison_threshold': str(value_threshold),
        'scale_comparison_threshold': str(scale_threshold),
    }
    cfg['requirements'] = {
        'system_memory': str(system_memory),
        'gpu_support': '1',
        'gpu_memory': str(gpu_memory),
    }

    tag_list = ['nhlbi'] + [t.strip() for t in tags.split(',') if t.strip()]
    cfg['tags'] = {
        'tags': ','.join(tag_list),
    }
    cfg['nhlbi'] = {
        'description': description,
        'noise_file': f'{name}/noise_data.h5',
    }

    if optional_additional_datasets:
        for i, dataset in enumerate(optional_additional_datasets):
            cfg['nhlbi'].update({
                f'additional_dataset_{i}': f"{name}/{op.basename(dataset)}"})
    
    cfg_path = CASES_DIR / f"{name}.cfg"
    with open(cfg_path, 'w') as f:
        cfg.write(f)

    return cfg_path


def main():
    parser = argparse.ArgumentParser(
        description="Register a new NHLBI integration test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--name', required=True, help="Test case name (e.g., imoco_vds)")
    parser.add_argument('--config', required=True, help="Gadgetron XML config for reconstruction")
    parser.add_argument('--noise-file', required=True, help="Path to noise calibration HDF5 file")
    parser.add_argument('--data-file', required=True, help="Path to reconstruction input HDF5 file")
    parser.add_argument('--noise-config', default='default_measurement_dependencies.xml',
                        help="Gadgetron XML config for noise dependency")
    parser.add_argument('--description', default='', help="Human-readable test description")
    parser.add_argument('--gpu-memory', type=int, default=8192, help="Required GPU memory in MB")
    parser.add_argument('--system-memory', type=int, default=8192, help="Required system memory in MB")
    parser.add_argument('--value-threshold', type=float, default=0.01,
                        help="Value comparison threshold for baseline validation")
    parser.add_argument('--scale-threshold', type=float, default=0.01,
                        help="Scale comparison threshold for baseline validation")
    parser.add_argument('--tags', type=str, default='',
                        help="Comma-separated additional tags (nhlbi is always included)")
    parser.add_argument('--skip-upload', action='store_true',
                        help="Skip uploading to Azure (for local-only testing)")
    parser.add_argument('--additional-files',nargs='+',type=str,default=[],
                        help='List of additional files for testing (e.g traj_bSTAR.seq traj_bSTAR.h5)')
    parser.add_argument('--additional-dependencies',nargs='+',type=str,default=[],
                        help='List of additional dependency data.h5 .xml data2 .xml2')

    args = parser.parse_args()
    print(args)
    print(type(args.additional_files))
    print(len(args.additional_files))
    print(type(args.additional_dependencies))
    print(len(args.additional_dependencies))
    print(args.additional_dependencies)
    # Check if test already exists
    cfg_path = CASES_DIR / f"{args.name}.cfg"
    if cfg_path.exists():
        print(f"Error: Test '{args.name}' already exists at {cfg_path}")
        print("Use update_test.py to modify existing tests.")
        sys.exit(1)

    
    # Validate input files
    print("Validating input files...")
    if not op.isfile(args.noise_file):
        print(f"Error: Noise file not found: {args.noise_file}")
        sys.exit(1)
    if not op.isfile(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)
    if not validate_hdf5(args.noise_file):
        sys.exit(1)
    if not validate_hdf5(args.data_file):
        sys.exit(1)

    validate_config_exists(args.config)
    
    # Validate additional files
    for additional_file in args.additional_files:
        if not op.isfile(additional_file):
            print(f"Error: Additional file not found: {additional_file}")
            sys.exit(1)
        if additional_file.endswith('.h5'):
            if not validate_hdf5(additional_file):
                sys.exit(1)
    
    # Additional dependencies
    if len(args.additional_dependencies) % 2 !=0 :
        print(f"Error additional dependencies required data.h5 and config.xml")
        sys.exit(1)
    for i in range(len(args.additional_dependencies)//2):
        dependency_file=args.additional_dependencies[2*i]
        print(dependency_file)
        if not op.isfile(dependency_file):
            print(f"Error: Dependency file not found: {dependency_file}")
            sys.exit(1)
        if dependency_file.endswith('.h5'):
            if not validate_hdf5(dependency_file):
                sys.exit(1)
    
    # Compute checksums
    print("Computing checksums...")
    noise_sha256 = calc_sha256(args.noise_file)
    data_sha256 = calc_sha256(args.data_file)
    print(f"  Noise SHA256: {noise_sha256}")
    print(f"  Data  SHA256: {data_sha256}")

    additional_files_sha256 = []
    for additional_file in args.additional_files:
        sha256 = calc_sha256(additional_file)
        additional_files_sha256.append((additional_file, sha256))
        print(f"  Additional file {additional_file} SHA256: {sha256}")
    
    additional_dependency_file_sha256 = []
    for i in range(len(args.additional_dependencies)//2):
        dependency_file=args.additional_dependencies[2*i]
        sha256 = calc_sha256(dependency_file)
        additional_dependency_file_sha256.append((dependency_file, sha256))
        print(f"  Dependency file {dependency_file} SHA256: {sha256}")
    
    # Upload to Azure
    if not args.skip_upload:
        print("Uploading to Azure Blob Storage...")
        container_client = get_container_client()
        upload_blob(container_client, args.noise_file, f"{args.name}/noise_data.h5")
        upload_blob(container_client, args.data_file, f"{args.name}/recon_data.h5")
        for additional_file, sha256 in additional_files_sha256:
            upload_blob(container_client, additional_file, f"{args.name}/{op.basename(additional_file)}")
        for additional_dependency_file, sha256 in additional_dependency_file_sha256:
            upload_blob(container_client, additional_dependency_file, f"{args.name}/{op.basename(additional_dependency_file)}")
    else:
        print("Skipping Azure upload (--skip-upload)")

    # Update manifest
    manifest = load_manifest()
    manifest.append({
        'file': f'{args.name}/noise_data.h5',
        'sha256': noise_sha256,
        'type': 'noise',
        'test': args.name,
    })
    manifest.append({
        'file': f'{args.name}/recon_data.h5',
        'sha256': data_sha256,
        'type': 'input',
        'test': args.name,
    })
    
    for additional_file, sha256 in additional_files_sha256:
        manifest.append({
            'file': f"{args.name}/{op.basename(additional_file)}",
            'sha256': sha256,
            'type': 'additional',
            'test': args.name,
        })
        
    for additional_dependency_file, sha256 in additional_dependency_file_sha256:
        manifest.append({
            'file': f"{args.name}/{op.basename(additional_dependency_file)}",
            'sha256': sha256,
            'type': 'additional',
            'test': args.name,
        }) 
        
        
    save_manifest(manifest)
    print(f"Updated manifest: {len(manifest)} entries")

    # Create .cfg file
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = create_cfg(
        name=args.name,
        config=args.config,
        noise_config=args.noise_config,
        description=args.description,
        gpu_memory=args.gpu_memory,
        system_memory=args.system_memory,
        value_threshold=args.value_threshold,
        scale_threshold=args.scale_threshold,
        tags=args.tags,
        optional_additional_datasets=args.additional_files,
        optional_additional_dependency_datasets=args.additional_dependencies
    )
    print(f"Created test case: {cfg_path}")
    print(f"\nNext step: Run 'python generate_baseline.py --test {args.name}' to create baseline")


if __name__ == '__main__':
    main()
