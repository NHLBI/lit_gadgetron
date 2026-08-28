#!/usr/bin/env python3
"""NHLBI Integration Test Orchestrator.

Thin wrapper around the existing run_gadgetron_test.py that handles
NHLBI-specific concerns: downloading test data from private Azure Blob
Storage and filtering by NHLBI tags.

Usage:
    python run_nhlbi_tests.py cases/*.cfg
    python run_nhlbi_tests.py cases/imoco_vds.cfg
    python run_nhlbi_tests.py cases/*.cfg --only fast
"""

import argparse
import configparser
import csv
import glob
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from test_utils import get_gadgetron_bin_path
# Ensure gadgetron binaries are on PATH
_gadgetron_bin = get_gadgetron_bin_path()
if _gadgetron_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _gadgetron_bin + ":" + os.environ.get("PATH", "")

# Reuse tag/requirement parsing from the existing test runner.
# In dev: test/nhlbi_integration_tests/../integration  i.e. test/integration
# In RT container: /opt/nhlbi-integration-test -> sibling is /opt/integration-test
_dev_integration = Path(__file__).parent.parent / 'integration'
_rt_integration = Path('/opt/integration-test')
INTEGRATION_DIR = _dev_integration if _dev_integration.is_dir() else _rt_integration
sys.path.insert(0, str(INTEGRATION_DIR))
from run_tests import (
    _colors_disabled,
    _colors_enabled,
    output_csv,
    output_log_file,
    query_gadgetron_capabilities,
    ignore_gadgetron_capabilities,
    read_test_details,
    should_skip_test,
    split_tag_list,
)

SCRIPT_DIR = Path(__file__).parent
RUN_TEST_SCRIPT = INTEGRATION_DIR / 'run_gadgetron_test.py'


def get_data_dir():
    return os.environ.get('NHLBI_DATA_CACHE', str(SCRIPT_DIR / 'data'))


def download_test_data(test_names):
    """Download data for the specified tests from Azure Blob Storage."""
    from get_nhlbi_data import download_data
    import argparse as _argparse

    data_dir = get_data_dir()
    for name in test_names:
        dl_args = _argparse.Namespace(
            destination=data_dir,
            test=name,
            list=str(SCRIPT_DIR / 'nhlbi_data.json'),
        )
        try:
            download_data(dl_args)
        except Exception as e:
            print(f"Warning: Could not download data for test '{name}': {e}")


def get_test_name_from_cfg(cfg_path):
    """Extract the test name from a .cfg file path."""
    return Path(cfg_path).stem


def check_baseline_exists(cfg_path):
    """Check if the test has a baseline registered in the manifest."""
    test_name = get_test_name_from_cfg(cfg_path)
    manifest_path = SCRIPT_DIR / 'nhlbi_data.json'
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        return any(
            e.get('test') == test_name and e.get('type') == 'baseline'
            for e in manifest
        )
    except Exception:
        return False


def get_baseline_recon_time(cfg_path):
    """Read the baseline reconstruction time from the .cfg file."""
    config = configparser.ConfigParser()
    config.read(cfg_path)
    try:
        return float(config['nhlbi']['baseline_recon_time'])
    except (KeyError, ValueError):
        return None


def check_speed_regression(test_file, actual_time, color_handler, speed_threshold):
    """Compare actual reconstruction time against baseline and report."""
    baseline_time = get_baseline_recon_time(test_file)
    if baseline_time is None:
        return None

    ratio = actual_time / baseline_time
    pct_change = (ratio - 1.0) * 100

    if ratio > speed_threshold:
        print(color_handler(
            f"  SPEED REGRESSION: {actual_time:.1f}s vs baseline {baseline_time:.1f}s "
            f"({pct_change:+.1f}%, threshold {(speed_threshold - 1) * 100:.0f}%)",
            'red',
        ))
        return False
    elif pct_change < -5:
        print(color_handler(
            f"  Speed improved: {actual_time:.1f}s vs baseline {baseline_time:.1f}s ({pct_change:+.1f}%)",
            'green',
        ))
    else:
        print(f"  Speed OK: {actual_time:.1f}s vs baseline {baseline_time:.1f}s ({pct_change:+.1f}%)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="NHLBI Integration Test Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-p', '--port', type=int, default=9003, help="Port for Gadgetron instance")
    parser.add_argument('-a', '--host', type=str, default="localhost", help="Address of Gadgetron host")

    parser.add_argument('-e', '--external', action='store_const', const=['-e'], default=[],
                        help="Use external Gadgetron; don't start a new instance each test.")

    parser.add_argument('-d', '--data-folder', type=str, default=None,
                        help="Look for test data in the specified folder (default: NHLBI_DATA_CACHE or ./data)")
    parser.add_argument('-t', '--test-folder', type=str, default='test',
                        help="Save Gadgetron and Client output to specified folder")

    parser.add_argument('-F', '--ignore-failures', action='store_true', default=False,
                        help="Continue running tests after failures")
    parser.add_argument('-s', '--stats', type=str, default=None,
                        help="Output individual test stats to CSV file")

    parser.add_argument('--timeout', type=int, default=None,
                        help="Fail test if it runs longer than timeout seconds")

    parser.add_argument('--echo-log-on-failure', action='store_true', default=False,
                        help="Send test logs to stdout on failure")

    parser.add_argument('--disable-color', dest='color_handler', action='store_const',
                        const=_colors_disabled, default=_colors_enabled,
                        help="Disable colors in output")

    parser.add_argument('--disable-capability-query', action='store_const',
                        dest='capability_query_function',
                        const=ignore_gadgetron_capabilities,
                        default=query_gadgetron_capabilities,
                        help="Disable querying Gadgetron capabilities")

    parser.add_argument('--ignore-requirements', type=split_tag_list, default='none', metavar='tags',
                        help="Run tests with specified tags regardless of capabilities")
    parser.add_argument('--only', type=split_tag_list, default='all', metavar='tags',
                        help="Only run tests with the specified tags")
    parser.add_argument('--exclude', type=split_tag_list, default='none', metavar='tags',
                        help="Do not run tests with the specified tags")

    parser.add_argument('--skip-download', action='store_true', default=False,
                        help="Skip automatic data download from Azure")

    parser.add_argument('--speed-threshold', type=float, default=1.5,
                        help="Fail if reconstruction takes longer than this multiple of baseline time (e.g., 1.5 = 50%% slower)")
    parser.add_argument('--no-speed-check', action='store_true', default=False,
                        help="Disable speed regression checking")

    parser.add_argument('tests', type=str, nargs='+', help="Test case .cfg files or glob patterns")

    args = parser.parse_args()

    data_dir = args.data_folder or get_data_dir()

    # Resolve test files
    files = sorted(set(itertools.chain(*[glob.glob(pattern) for pattern in args.tests])))
    if not files:
        print("No test files found matching the specified patterns.")
        sys.exit(1)

    # Check for missing baselines
    missing_baselines = []
    for f in files:
        if not check_baseline_exists(f):
            missing_baselines.append(f)
            print(args.color_handler(
                f"Warning: No baseline for {f} — test will be skipped",
                'cyan',
            ))

    # Filter out tests without baselines
    files = [f for f in files if f not in missing_baselines]
    if not files:
        print("No tests with baselines to run.")
        sys.exit(0)

    # Download test data
    if not args.skip_download:
        test_names = [get_test_name_from_cfg(f) for f in files]
        print("Downloading test data...")
        download_test_data(test_names)

    # Read test details and filter by capabilities/tags
    tests = [read_test_details(f) for f in files]
    capabilities = args.capability_query_function(args)

    stats = []
    passed = []
    failed = []
    skipped = []
    speed_regressions = []

    def skip_handler(test, message):
        skipped.append((test, message))

    tests = [t for t in tests if not should_skip_test(t, capabilities, args, skip_handler)]

    if skipped:
        print("\nSkipped tests:")
        for test, message in skipped:
            print(f"\t{test.get('file')} ({message})")

    # Run each test
    for i, test in enumerate(tests, start=1):
        print(args.color_handler(f"\nTest {i} of {len(tests)}: {test.get('file')}\n", 'bold'))

        disable_color = ['--disable-colors'] if args.color_handler == _colors_disabled else []

        command = [
            sys.executable, str(RUN_TEST_SCRIPT),
            '-a', str(args.host),
            '-d', str(data_dir),
            '-t', str(args.test_folder),
            '-p', str(args.port),
        ] + args.external + disable_color + [test.get('file')]

        with subprocess.Popen(command) as proc:
            try:
                import time as _time
                test_start = _time.time()
                proc.wait(timeout=args.timeout)
                test_elapsed = _time.time() - test_start

                if proc.returncode == 0:
                    passed.append(test)
                    try:
                        with open('test/stats.json') as sf:
                            stat = json.loads(sf.read())
                            stats.append(stat)
                            actual_time = stat.get('processing_time', test_elapsed)
                    except FileNotFoundError:
                        actual_time = test_elapsed

                    # Check speed regression
                    if not args.no_speed_check:
                        speed_ok = check_speed_regression(
                            test.get('file'), actual_time,
                            args.color_handler, args.speed_threshold,
                        )
                        if speed_ok is False:
                            speed_regressions.append(test)
                else:
                    if args.echo_log_on_failure:
                        for log in glob.glob(os.path.join(args.test_folder, '*.log')):
                            output_log_file(log)
                    failed.append(test)
                    try:
                        with open('test/stats.json') as sf:
                            stats.append(json.loads(sf.read()))
                    except FileNotFoundError:
                        pass
                    if not args.ignore_failures:
                        break
            except subprocess.TimeoutExpired:
                print(f"Timeout during test: {test.get('file')}")
                proc.kill()
                failed.append(test)
                if not args.ignore_failures:
                    break

    if args.stats and stats:
        output_csv(stats, args.stats)

    # Summary
    if failed:
        print("\nFailed tests:")
        for test in failed:
            print(f"\t{test.get('file')}")

    if speed_regressions:
        print("\nSpeed regressions:")
        for test in speed_regressions:
            print(f"\t{test.get('file')}")

    if missing_baselines:
        print("\nTests skipped (no baseline):")
        for f in missing_baselines:
            print(f"\t{f}")

    print(f"\n{len(passed)} tests passed. {len(failed)} tests failed. "
          f"{len(skipped)} tests skipped. {len(missing_baselines)} missing baselines. "
          f"{len(speed_regressions)} speed regressions.")

    if stats:
        print(f"Total processing time: {sum(s['processing_time'] for s in stats):.2f} seconds.")

    sys.exit(bool(failed))


if __name__ == '__main__':
    main()