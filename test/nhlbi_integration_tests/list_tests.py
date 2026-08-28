#!/usr/bin/env python3
"""List all registered NHLBI integration tests and their status."""

import configparser
import json
import sys
from pathlib import Path

from get_nhlbi_data import MANIFEST_FILE

CASES_DIR = Path(__file__).parent / "cases"
BASELINES_DIR = Path(__file__).parent / "baselines"

_codes = {
    'red': '\033[91m',
    'green': '\033[92m',
    'cyan': '\033[96m',
    'yellow': '\033[93m',
    'bold': '\033[1m',
    'end': '\033[0m',
}


def color(text, c):
    return f"{_codes.get(c, '')}{text}{_codes.get('end', '')}"


def main():
    cfg_files = sorted(CASES_DIR.glob("*.cfg"))

    if not cfg_files:
        print("No tests registered. Use submit_test.py to add one.")
        sys.exit(0)

    # Load manifest
    try:
        with open(MANIFEST_FILE, 'r') as f:
            manifest = json.load(f)
    except Exception:
        manifest = []

    def get_manifest_entry(test_name, entry_type):
        return next((e for e in manifest if e.get('test') == test_name and e.get('type') == entry_type), None)

    print(color(f"{'Test':<25} {'Config':<40} {'Baseline':<15} {'Recon Time':<12} {'Tags'}", 'bold'))
    print("-" * 110)

    for cfg_file in cfg_files:
        name = cfg_file.stem
        config = configparser.ConfigParser()
        config.read(cfg_file)

        # Config XML
        recon_config = config.get('reconstruction.client', 'configuration', fallback='?')

        # Tags
        tags = config.get('tags', 'tags', fallback='')

        # Baseline status
        baseline_entry = get_manifest_entry(name, 'baseline')
        has_local_output = (BASELINES_DIR / name / 'output.h5').exists()

        if baseline_entry:
            validated_by = baseline_entry.get('validated_by', '?')
            validated_date = baseline_entry.get('validated_date', '?')
            baseline_status = color(f"yes ({validated_date})", 'green')
        elif has_local_output:
            baseline_status = color("local only", 'yellow')
        else:
            baseline_status = color("missing", 'red')

        # Recon time
        recon_time = config.get('nhlbi', 'baseline_recon_time', fallback=None)
        if recon_time:
            recon_time_str = f"{float(recon_time):.1f}s"
        elif baseline_entry and baseline_entry.get('recon_time_seconds'):
            recon_time_str = f"{baseline_entry['recon_time_seconds']:.1f}s"
        else:
            recon_time_str = "-"

        # Description
        description = config.get('nhlbi', 'description', fallback='')

        print(f"{name:<25} {recon_config:<40} {baseline_status:<27} {recon_time_str:<12} {tags}")
        if description:
            print(f"  {color(description, 'cyan')}")

    # Data summary
    noise_count = sum(1 for e in manifest if e.get('type') == 'noise')
    input_count = sum(1 for e in manifest if e.get('type') == 'input')
    baseline_count = sum(1 for e in manifest if e.get('type') == 'baseline')
    print(f"\n{len(cfg_files)} tests, {baseline_count} baselines, "
          f"{noise_count + input_count} data files in manifest")


if __name__ == '__main__':
    main()
