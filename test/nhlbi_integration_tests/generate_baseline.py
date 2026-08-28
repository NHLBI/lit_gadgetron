#!/usr/bin/env python3
"""Generate and validate baselines for NHLBI integration tests.

Runs a reconstruction, generates preview artifacts, and prompts the user
to accept or reject the output as a baseline. Accepted baselines are
uploaded to Azure Blob Storage and registered in the manifest.

Usage:
    python generate_baseline.py --test imoco_vds [--port 9002]
"""

import argparse
import configparser
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from get_nhlbi_data import download_data
import h5py
import numpy as np
from test_utils import get_gadgetron_bin_path
from get_nhlbi_data import (
    calc_sha256,
    get_container_client,
    load_manifest,
    save_manifest,
    upload_blob,
)
from test_utils import read_h5
from collections import OrderedDict
import os.path as op

CASES_DIR = Path(__file__).parent / "cases"
BASELINES_DIR = Path(__file__).parent / "baselines"

# Ensure gadgetron binaries are on PATH
_gadgetron_bin = get_gadgetron_bin_path()
if _gadgetron_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _gadgetron_bin + ":" + os.environ.get("PATH", "")

# Import from existing integration test framework
sys.path.insert(0, str(Path(__file__).parent.parent / 'integration'))
from run_gadgetron_test import (
    send_data_to_gadgetron,
    start_gadgetron_instance,
    start_storage_server,
)


def get_data_dir():
    return os.environ.get('NHLBI_DATA_CACHE', str(Path(__file__).parent / 'data'))


def echo_handler(cmd):
    print(' '.join(cmd))


class GadgetronInstance:
    def __init__(self, host, port):
        self.host = host
        self.port = port


def generate_preview(output_file, preview_dir):
    """Generate text summary and optional PNG montage of reconstruction output."""
    os.makedirs(preview_dir, exist_ok=True)
    summary_lines = []

    try:
        with h5py.File(output_file, 'r') as f:
            summary_lines.append(f"File: {output_file}")
            summary_lines.append(f"Groups: {list(f.keys())}")

            def visit_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    summary_lines.append(f"  Dataset: {name}")
                    summary_lines.append(f"    Shape: {obj.shape}")
                    summary_lines.append(f"    Dtype: {obj.dtype}")
                    if np.issubdtype(obj.dtype, np.number) and obj.size > 0:
                        data = obj[...]
                        if np.iscomplexobj(data):
                            data = np.abs(data)
                        summary_lines.append(f"    Min:  {np.min(data):.6e}")
                        summary_lines.append(f"    Max:  {np.max(data):.6e}")
                        summary_lines.append(f"    Mean: {np.mean(data):.6e}")
                        summary_lines.append(f"    Std:  {np.std(data):.6e}")

            f.visititems(visit_datasets)
    except Exception as e:
        summary_lines.append(f"Error reading output: {e}")

    summary_text = '\n'.join(summary_lines)
    summary_path = os.path.join(preview_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)

    print("\n=== Baseline Preview ===")
    print(summary_text)
    print("========================\n")

    # Attempt PNG montage of central slices
    try:
        _generate_montage(output_file, preview_dir)
    except Exception as e:
        print(f"Note: Could not generate PNG montage: {e}")
        print("Install matplotlib for visual previews: pip install matplotlib")

    return summary_path


def _generate_montage(output_file, preview_dir):
    """Generate a PNG montage showing central slices from each dimension."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    img_list,header_list=read_h5(output_file)
    
    if len(img_list) == 0:
        print("No image datasets found for montage.")
        return
    for k in range(len(img_list)):
        image_data = img_list[k].squeeze()
        print(image_data.shape)
        header= header_list[k]
        print(header[0])
        img_serie_index=header[0]['image_series_index']
        if image_data.ndim == 2:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(image_data, cmap='gray')
            ax.set_title('2D Output')
            ax.axis('off')
        elif image_data.ndim == 3:
            nslices = image_data.shape[0]
            # Show up to 9 evenly spaced slices
            n_show = min(9, nslices)
            indices = np.linspace(0, nslices - 1, n_show, dtype=int)
            cols = min(3, n_show)
            rows = (n_show + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.atleast_2d(axes)
            for i, idx in enumerate(indices):
                r, c = divmod(i, cols)
                axes[r, c].imshow(image_data[idx], cmap='gray')
                axes[r, c].set_title(f'Slice {idx}')
                axes[r, c].axis('off')
            for i in range(n_show, rows * cols):
                r, c = divmod(i, cols)
                axes[r, c].axis('off')
        elif image_data.ndim >= 4:
            # Show central slice of last two dims across first dimension
            shape = image_data.shape
            # Flatten to 3D: combine all leading dims
            idx_0=image_data.shape[0]
            flat = image_data.reshape(-1, shape[-2], shape[-1],order="C")
            nslices = flat.shape[0]
            central_slices=(nslices/idx_0) // 2
            indices = np.arange(central_slices,nslices,idx_0).astype(np.int32)
            n_show = min(9, len(indices))
            if len(indices) > n_show:
                indices = indices[:n_show]
            cols = min(3, n_show)
            rows = (n_show + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.atleast_2d(axes)
            for i, idx in enumerate(indices):
                r, c = divmod(i, cols)
                axes[r, c].imshow(flat[idx], cmap='gray')
                axes[r, c].set_title(f'Frame {idx}')
                axes[r, c].axis('off')
            for i in range(n_show, rows * cols):
                r, c = divmod(i, cols)
                axes[r, c].axis('off')
        else:
            print("Data is 1D or scalar, skipping montage")
            continue

        fig.suptitle(f'Baseline Preview image {img_serie_index}', fontsize=14)
        fig.tight_layout()
        preview_path = os.path.join(preview_dir, f'preview_{img_serie_index}.png')
        fig.savefig(preview_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Preview montage saved: {preview_path}")


def run_reconstruction(test_name, port, storage_port):
    """Run noise dependency + reconstruction and return the output file path."""
    cfg_path = CASES_DIR / f"{test_name}.cfg"
    if not cfg_path.exists():
        print(f"Error: Test case not found: {cfg_path}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read_dict({
        "DEFAULT": {
            'parameter_xml': 'IsmrmrdParameterMap_Siemens.xml',
            'parameter_xsl': 'IsmrmrdParameterMap_Siemens.xsl',
            'value_comparison_threshold': '0.01',
            'scale_comparison_threshold': '0.01',
        }
    })
    config.read(cfg_path)

    data_dir = get_data_dir()
    
    test_dir = str(BASELINES_DIR / test_name)
    os.makedirs(test_dir, exist_ok=True)

    # Resolve data file paths
    noise_file = os.path.join(data_dir, config['dependency.siemens']['data_file'])
    recon_file = os.path.join(data_dir, config['reconstruction.siemens']['data_file'])

    if not os.path.isfile(noise_file):
        print(f"Error: Noise file not found: {noise_file}")
        print("Run 'python get_nhlbi_data.py download --test {}' first.".format(test_name))
        sys.exit(1)
    if not os.path.isfile(recon_file):
        print(f"Error: Recon data file not found: {recon_file}")
        print("Run 'python get_nhlbi_data.py download --test {}' first.".format(test_name))
        sys.exit(1)

    # additional dependency files 
    
    add_dep_sections_number=[dep_sec.split(".")[-1] for dep_sec in config.sections() if "dependency.siemens." in dep_sec]

    for num_dep in add_dep_sections_number:
        dep_file=os.path.join(data_dir,config[f"dependency.siemens.{num_dep}"]['data_file'])
        if not os.path.isfile(dep_file):
            print(f"Error: Recon data file not found: {dep_file}")
            print("Run 'python get_nhlbi_data.py download --test {}' first.".format(test_name))
            sys.exit(1)
    
    output_file = os.path.join(test_dir, 'output.h5')

    noise_config = config['dependency.client']['configuration']
    recon_config = config['reconstruction.client']['configuration']

    gadgetron_instance = GadgetronInstance("localhost", str(port))

    with tempfile.TemporaryDirectory() as storage_folder:
        storage_log = open(os.path.join(test_dir, 'storage.log'), 'w')
        try:
            storage_proc = start_storage_server(
                log=storage_log,
                port=str(storage_port),
                storage_folder=storage_folder,
            )
        except Exception as e:
            storage_log.close()
            print(f"Error starting storage server: {e}")
            sys.exit(1)

        try:
            gt_log_out = open(os.path.join(test_dir, 'gadgetron.log.out'), 'w')
            gt_log_err = open(os.path.join(test_dir, 'gadgetron.log.err'), 'w')
            storage_address = f"http://localhost:{storage_port}"

            gt_proc = start_gadgetron_instance(
                log_stdout=gt_log_out,
                log_stderr=gt_log_err,
                port=str(port),
                storage_address=storage_address,
            )

            try:
                # Wait briefly for gadgetron to start
                time.sleep(2)

                if len(add_dep_sections_number)>0:
                    print("\n--- Sending additional dependencies ---")
                    print(add_dep_sections_number)
                    for num_dep in add_dep_sections_number:
                        dep_siem=f"dependency.siemens.{num_dep}"
                        input_dep_file=os.path.join(data_dir,config[f"dependency.siemens.{num_dep}"]['data_file'])
                        input_config_file=config[f"dependency.client.{num_dep}"]['configuration']
                        dep_name=f"dep_{num_dep}"
                        dep_log = open(os.path.join(test_dir, f'{dep_name}.log'), 'w')
                        print("\n--- Sending dependency {input_dep_file} with config {input_config_file} ---")
                        send_data_to_gadgetron(
                            echo_handler, gadgetron_instance,
                            input=input_dep_file,
                            output=os.path.join(test_dir, f'{dep_name}_output.h5'),
                            configuration=['-c', input_config_file],
                            group=noise_config,
                            log=dep_log,
                            additional_arguments=config[f"dependency.siemens.{num_dep}"].get('additional_arguments'),
                        )
                        dep_log.close()
                # Send noise data
                print(f"\n--- Sending noise data ({noise_config}) ---")
                noise_log = open(os.path.join(test_dir, 'noise_client.log'), 'w')
                send_data_to_gadgetron(
                    echo_handler, gadgetron_instance,
                    input=noise_file,
                    output=os.path.join(test_dir, 'noise_output.h5'),
                    configuration=['-c', noise_config],
                    group=noise_config,
                    log=noise_log,
                    additional_arguments=config['dependency.siemens'].get('additional_arguments'),
                )
                noise_log.close()

                # Send reconstruction data
                print(f"\n--- Sending reconstruction data ({recon_config}) ---")
                recon_log = open(os.path.join(test_dir, 'recon_client.log'), 'w')
                start_time = time.time()
                send_data_to_gadgetron(
                    echo_handler, gadgetron_instance,
                    input=recon_file,
                    output=output_file,
                    configuration=['-c', recon_config],
                    group=recon_config,
                    log=recon_log,
                    additional_arguments=config['reconstruction.siemens'].get('additional_arguments'),
                )
                recon_log.close()
                elapsed = time.time() - start_time
                print(f"Reconstruction completed in {elapsed:.1f}s")

            finally:
                gt_proc.kill()
                gt_log_out.close()
                gt_log_err.close()
        finally:
            storage_proc.kill()
            storage_log.close()

    if not os.path.isfile(output_file):
        print("Error: No output file was produced.")
        print(f"Check logs in {test_dir}/")
        sys.exit(1)

    return output_file, elapsed


def update_cfg(cfg_path, baseline_file, recon_time):
    """Update the .cfg file with the new baseline reference and recon time."""
    test_name=op.basename(cfg_path).replace('.cfg','')
    config = configparser.ConfigParser()
    config.read(cfg_path)
    images,headers=read_h5(baseline_file)
    print(config.sections())
    reconstruction_tests=[section_name for section_name in config.sections() if section_name.startswith('reconstruction.test')]
    value_comparison_threshold_initial=config[reconstruction_tests[0]]['value_comparison_threshold']
    scale_comparison_threshold_initial=config[reconstruction_tests[0]]['scale_comparison_threshold']
    for section_name in reconstruction_tests:
        config.remove_section(section_name)
    
    key_names=[f'reconstruction.test.{i+1}' for i in range(len(images))]
    if len(key_names)==1:
        key_names=['reconstruction.test']
    for key_name,image,header in zip(key_names,images,headers):
            image_serie_index=header[0]['image_series_index']
            config[key_name] = {
                'reference_file': f'{test_name}/baseline_output.h5',
                'reference_images': f"{config['reconstruction.client']['configuration']}/image_{image_serie_index}",
                'output_images': f"{config['reconstruction.client']['configuration']}/image_{image_serie_index}",
                'value_comparison_threshold': value_comparison_threshold_initial,
                'scale_comparison_threshold': scale_comparison_threshold_initial,
            }
    config['nhlbi']['baseline_recon_time'] = str(round(recon_time, 1))
    # Sorting keys to ensure deterministic order in the .cfg file
    desired_order = [section_name for section_name in config.sections() if section_name.startswith('dependency')]
    desired_order.extend(['reconstruction.siemens','reconstruction.client'])
    desired_order.extend([section_name for section_name in config.sections() if section_name.startswith('reconstruction.test')])
    desired_order.extend(['requirements','tags','nhlbi'])
    
    ordered_config = configparser.ConfigParser()
    
    section_order = config.sections()
    for section in desired_order:
        if config.has_section(section):
            ordered_config.add_section(section)
            for key, value in config.items(section):
                ordered_config.set(section, key, value)
    
    # Optionally, add any remaining sections not in the desired order
    for section in config.sections():
        if section not in desired_order:
            print(f"Warning: Section '{section}' not in desired order list, adding at the end.")
            ordered_config.add_section(section)
            for key, value in config.items(section):
                ordered_config.set(section, key, value)
    
    with open(cfg_path, 'w') as f:
        ordered_config.write(f)

def main():
    parser = argparse.ArgumentParser(
        description="Generate and validate NHLBI test baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--test', required=True, help="Test case name")
    parser.add_argument('--port', type=int, default=9003, help="Gadgetron port")
    parser.add_argument('--storage-port', type=int, default=9113, help="Storage server port")
    parser.add_argument('--auto-accept', action='store_true',
                        help="Accept baseline without interactive prompt (for CI)")
    parser.add_argument('--skip-upload', action='store_true',
                        help="Skip uploading baseline to Azure")
    parser.add_argument('--accept-existing', action='store_true',
                        help="Accept an existing output in baselines/<test>/ without re-running reconstruction")
    parser.add_argument('--recon-time', type=float, default=None,
                        help="Override reconstruction time in seconds (use with --accept-existing)")

    args = parser.parse_args()

    test_name = args.test
    preview_dir = str(BASELINES_DIR / test_name)

    if args.accept_existing:
        # Accept an already-generated baseline without re-running
        output_file = str(BASELINES_DIR / test_name / 'output.h5')
        if not os.path.isfile(output_file):
            print(f"Error: No existing output at {output_file}")
            print("Run without --accept-existing to generate it first.")
            sys.exit(1)

        # Try to extract recon time from gadgetron log
        recon_time = args.recon_time
        if recon_time is None:
            recon_time = _extract_recon_time_from_log(test_name)
        if recon_time is None:
            recon_time = 0.0
            print("Warning: Could not determine reconstruction time. Use --recon-time to set it.")

        print(f"Using existing output: {output_file}")
        print(f"Reconstruction time: {recon_time:.1f}s")
        generate_preview(output_file, preview_dir)
    else:
        # Download test data if needed
        print(f"Ensuring test data is available for '{test_name}'...")
        dl_args = argparse.Namespace(
            destination=get_data_dir(),
            test=test_name,
            list=str(Path(__file__).parent / 'nhlbi_data.json'),
        )
        try:
            download_data(dl_args)
        except Exception as e:
            print(f"Warning: Could not download data: {e}")
            print("Continuing with locally available data...")

        # Run reconstruction
        print(f"\nRunning reconstruction for '{test_name}'...")
        output_file, recon_time = run_reconstruction(test_name, args.port, args.storage_port)

        # Generate preview
        generate_preview(output_file, preview_dir)

    # Interactive validation
    if args.auto_accept:
        accept = True
    else:
        print(f"\nOutput file: {output_file}")
        print(f"Preview dir: {preview_dir}/")
        print(f"Reconstruction time: {recon_time:.1f}s")
        response = input("Accept this output as baseline? [y/N] ").strip().lower()
        accept = response in ('y', 'yes')

    if not accept:
        print("\nBaseline rejected. Output kept for inspection at:")
        print(f"  {preview_dir}/")
        print(f"\nTo re-run: python generate_baseline.py --test {test_name}")
        sys.exit(0)

    # Upload baseline
    baseline_sha256 = calc_sha256(output_file)

    if not args.skip_upload:
        print("\nUploading baseline to Azure...")
        container_client = get_container_client()
        upload_blob(container_client, output_file, f"{test_name}/baseline_output.h5")
    else:
        print("Skipping Azure upload (--skip-upload)")

    # Update manifest
    manifest = load_manifest()
    # Remove any existing baseline entry for this test
    manifest = [e for e in manifest if not (e.get('test') == test_name and e.get('type') == 'baseline')]
    manifest.append({
        'file': f'{test_name}/baseline_output.h5',
        'sha256': baseline_sha256,
        'type': 'baseline',
        'test': test_name,
        'validated_by': os.environ.get('USER', 'unknown'),
        'validated_date': datetime.now().strftime('%Y-%m-%d'),
        'git_sha': _get_git_sha(),
        'recon_time_seconds': round(recon_time, 1),
    })
    save_manifest(manifest)

    # Update .cfg with reference_file and baseline timing
    
    cfg_path = CASES_DIR / f"{test_name}.cfg"
    update_cfg(cfg_path, output_file, recon_time)
    
    """config = configparser.ConfigParser()
    config.read(cfg_path)
    
    # Set the reference file to point to the baseline output in Azure Blob Storage
    
    
    config['reconstruction.test']['reference_file'] = f'{test_name}/baseline_output.h5'
    config['nhlbi']['baseline_recon_time'] = str(round(recon_time, 1))
    with open(cfg_path, 'w') as f:
        config.write(f)
    """
    print(f"\nBaseline accepted and registered for '{test_name}'")
    print(f"Baseline reconstruction time: {recon_time:.1f}s")
    print(f"Run 'python run_nhlbi_tests.py cases/{test_name}.cfg' to verify")


def _extract_recon_time_from_log(test_name):
    """Try to extract reconstruction time from gadgetron server log timestamps."""
    log_path = BASELINES_DIR / test_name / 'gadgetron.log.err'
    if not log_path.exists():
        return None
    try:
        import re
        timestamps = []
        with open(log_path, 'r') as f:
            for line in f:
                m = re.match(r'^(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)', line)
                if m:
                    timestamps.append(m.group(1))
        if len(timestamps) >= 2:
            from datetime import datetime
            fmt = '%m-%d %H:%M:%S.%f'
            start = datetime.strptime(timestamps[0], fmt)
            end = datetime.strptime(timestamps[-1], fmt)
            elapsed = (end - start).total_seconds()
            if elapsed > 0:
                return elapsed
    except Exception:
        pass
    return None


def _get_git_sha():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    main()
