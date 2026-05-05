#!/usr/bin/env python3
"""Fast DALI index builder for TFRecord files.

DALI's ``fn.readers.tfrecord`` needs an index file (``.idx``) per shard
listing the byte offset and size of every record. The ``tfrecord2idx`` CLI
that ships with DALI imports TensorFlow for each invocation (~20 s startup),
which makes indexing 1024 shards take hours. This script reads the binary
TFRecord framing directly and processes shards in parallel — ~40 s for the
full 1024-shard ImageNet train split.

TFRecord on-disk format (per record):
    uint64 length
    uint32 length_crc
    <length> bytes of data
    uint32 data_crc

DALI index format: ``"offset size\n"`` per record where ``offset`` is the
first byte of the length field and ``size`` is the total bytes on disk.

Usage:
    python scripts/build_tfrecord_index.py <src_dir> <out_dir>

If src_dir == out_dir, index files are written alongside each .tfrecord
as ``<stem>.tfrecord.idx`` — this is what ``DALIImageNetLoader`` expects.
"""

import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def build_index(tfrec_path: str, idx_path: str) -> int:
    recs = []
    with open(tfrec_path, 'rb') as f:
        size = os.fstat(f.fileno()).st_size
        pos = 0
        while pos < size:
            hdr = f.read(12)  # uint64 length + uint32 length_crc
            if len(hdr) < 12:
                break
            length = struct.unpack('<Q', hdr[:8])[0]
            f.seek(length + 4, 1)  # skip payload + data_crc
            rec_size = 12 + length + 4
            recs.append((pos, rec_size))
            pos += rec_size
    with open(idx_path, 'w') as o:
        for off, sz in recs:
            o.write(f'{off} {sz}\n')
    return len(recs)


def _worker(args):
    src, dst = args
    return dst, build_index(src, dst)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <src_dir> <out_dir>", file=sys.stderr)
        sys.exit(1)
    src_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob('*.tfrecord'))
    if not files:
        print(f"No .tfrecord files in {src_dir}", file=sys.stderr)
        sys.exit(1)
    print(f'{len(files)} shards')

    jobs = [(str(f), str(out_dir / (f.name + '.idx'))) for f in files]
    total = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for i, (dst, n) in enumerate(ex.map(_worker, jobs)):
            total += n
            if (i + 1) % 64 == 0:
                print(f'  [{i+1}/{len(files)}] {n} records in {Path(dst).name}')
    print(f'done: {total} records across {len(files)} shards')


if __name__ == '__main__':
    main()
