#!/usr/bin/env python3
"""Download and upload NHLBI test data from/to private Azure Blob Storage.

Authentication uses DefaultAzureCredential (picks up `az login` for developers,
managed identity or service principal for CI). Fallback: set NHLBI_AZURE_SAS_TOKEN
environment variable for SAS-token-based access.

Dependencies: pip install azure-storage-blob azure-identity
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STORAGE_ACCOUNT = "gadgetrondata"
CONTAINER_NAME = "nhlbitestdata"
ACCOUNT_URL = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
DEFAULT_DATA_DIR = "data"
MANIFEST_FILE = Path(__file__).parent / "nhlbi_data.json"


def calc_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def is_valid(filepath, expected_sha256):
    if not os.path.isfile(filepath):
        return False
    return expected_sha256 == calc_sha256(filepath)


def get_container_client():
    sas_token = os.environ.get("NHLBI_AZURE_SAS_TOKEN")
    if sas_token:
        from azure.storage.blob import ContainerClient
        return ContainerClient(
            account_url=ACCOUNT_URL,
            container_name=CONTAINER_NAME,
            credential=sas_token,
        )
    else:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import ContainerClient
        credential = DefaultAzureCredential()
        return ContainerClient(
            account_url=ACCOUNT_URL,
            container_name=CONTAINER_NAME,
            credential=credential,
        )


def download_blob_public(blob_name, destination, retries=3):
    """Download a blob via public URL (no auth required)."""
    import urllib.request
    import urllib.error
    import socket

    url = f"{ACCOUNT_URL}/{CONTAINER_NAME}/{blob_name}"
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with open(destination, 'wb') as f:
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        f.write(chunk)
            return
        except (urllib.error.URLError, ConnectionResetError, socket.timeout) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to download {blob_name} after {retries} attempts: {e}")
            print(f"Retry {attempt + 1} for {blob_name}: {e}")


def upload_blob(container_client, local_path, blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    print(f"Uploading {local_path} -> {blob_name}")
    with open(local_path, 'rb') as f:
        blob_client.upload_blob(f, overwrite=True)
    print(f"Upload complete: {blob_name}")


def load_manifest():
    with open(MANIFEST_FILE, 'r') as f:
        return json.load(f)


def save_manifest(entries):
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(entries, f, indent=2)
        f.write('\n')


def download_data(args):
    entries = load_manifest()

    if args.test:
        entries = [e for e in entries if e.get('test') == args.test]
        if not entries:
            print(f"No data entries found for test '{args.test}'")
            sys.exit(1)

    data_dir = args.destination

    def download_entry(entry):
        destination = os.path.join(data_dir, entry['file'])
        if is_valid(destination, entry['sha256']):
            print(f"Verified: {destination}")
            return
        print(f"Downloading: {entry['file']}")
        download_blob_public(entry['file'], destination)
        if not is_valid(destination, entry['sha256']):
            actual = calc_sha256(destination)
            raise RuntimeError(
                f"Downloaded file {destination} failed validation. "
                f"Expected SHA256 {entry['sha256']}. Actual SHA256 {actual}"
            )
        print(f"Saved: {destination}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download_entry, entries))


def upload_data(args):
    container_client = get_container_client()
    upload_blob(container_client, args.local_path, args.remote_path)
    sha256 = calc_sha256(args.local_path)
    print(f"SHA256: {sha256}")
    return sha256


def main():
    parser = argparse.ArgumentParser(
        description="NHLBI Integration Test Data Manager",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    dl_parser = subparsers.add_parser('download', help='Download test data')
    dl_parser.add_argument('-d', '--destination', type=str,
                           default=os.environ.get('NHLBI_DATA_CACHE', DEFAULT_DATA_DIR),
                           help="Local folder for downloaded data")
    dl_parser.add_argument('-t', '--test', type=str, default=None,
                           help="Download data only for the specified test case")
    dl_parser.add_argument('-l', '--list', type=str, default=str(MANIFEST_FILE),
                           help="Path to data manifest file")

    ul_parser = subparsers.add_parser('upload', help='Upload data to Azure')
    ul_parser.add_argument('local_path', type=str, help="Local file to upload")
    ul_parser.add_argument('remote_path', type=str, help="Blob path in container")

    args = parser.parse_args()

    if args.command == 'download':
        download_data(args)
    elif args.command == 'upload':
        upload_data(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
