#!/usr/bin/env python3
"""Remove an NHLBI integration test case.

Deletes the .cfg file, removes manifest entries, and optionally deletes
the associated Azure Blob Storage data.

Usage:
    python delete_test.py imoco_vds
    python delete_test.py imoco_vds --keep-data
"""

import argparse
import sys
from pathlib import Path

from get_nhlbi_data import get_container_client, load_manifest, save_manifest

CASES_DIR = Path(__file__).parent / "cases"
BASELINES_DIR = Path(__file__).parent / "baselines"


def main():
    parser = argparse.ArgumentParser(
        description="Remove an NHLBI integration test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('name', help="Test case name to delete")
    parser.add_argument('--keep-data', action='store_true',
                        help="Keep Azure blobs; only remove local .cfg and manifest entries")
    parser.add_argument('--yes', '-y', action='store_true',
                        help="Skip confirmation prompt")

    args = parser.parse_args()
    name = args.name

    cfg_path = CASES_DIR / f"{name}.cfg"
    manifest = load_manifest()
    test_entries = [e for e in manifest if e.get('test') == name]

    if not cfg_path.exists() and not test_entries:
        print(f"Error: Test '{name}' not found")
        sys.exit(1)

    # Show what will be deleted
    print(f"Test: {name}")
    if cfg_path.exists():
        print(f"  Config: {cfg_path}")
    if test_entries:
        print(f"  Manifest entries: {len(test_entries)}")
        for entry in test_entries:
            print(f"    - {entry['file']} ({entry.get('type', 'unknown')})")
    if not args.keep_data and test_entries:
        print(f"  Azure blobs: {len(test_entries)} will be deleted")
    else:
        print(f"  Azure blobs: kept")

    baseline_dir = BASELINES_DIR / name
    if baseline_dir.exists():
        print(f"  Local baselines: {baseline_dir}")

    # Confirm
    if not args.yes:
        action = "and Azure blobs" if not args.keep_data else "(keeping Azure data)"
        response = input(f"\nDelete test '{name}' {action}? [y/N] ").strip().lower()
        if response not in ('y', 'yes'):
            print("Cancelled.")
            sys.exit(0)

    # Delete Azure blobs
    if not args.keep_data and test_entries:
        try:
            container_client = get_container_client()
            for entry in test_entries:
                blob_name = entry['file']
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.delete_blob()
                    print(f"  Deleted blob: {blob_name}")
                except Exception as e:
                    print(f"  Warning: Could not delete blob {blob_name}: {e}")
        except Exception as e:
            print(f"  Warning: Could not connect to Azure: {e}")
            print("  Local files will still be removed.")

    # Remove from manifest
    remaining = [e for e in manifest if e.get('test') != name]
    save_manifest(remaining)
    print(f"  Removed {len(test_entries)} manifest entries")

    # Delete .cfg
    if cfg_path.exists():
        cfg_path.unlink()
        print(f"  Deleted {cfg_path}")

    # Delete local baseline directory
    if baseline_dir.exists():
        import shutil
        shutil.rmtree(baseline_dir)
        print(f"  Deleted {baseline_dir}")

    print(f"\nTest '{name}' removed successfully.")


if __name__ == '__main__':
    main()
