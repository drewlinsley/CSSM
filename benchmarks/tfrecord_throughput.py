"""
Pure data-loader throughput test: iterate the TFRecord ImageNet loader
(no model, no JAX) for N seconds, report batches/sec.

Usage:
    python benchmarks/tfrecord_throughput.py --batch_size 1024 --seconds 60
    python benchmarks/tfrecord_throughput.py --batch_size 1024 --max_intraop 4 --seconds 60
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Keep TF off GPU — we only want to measure CPU data pipeline
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf

from src.data.imagenet_tfdata import TFRecordImageNetLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tfrecord_dir', type=str,
                   default='/oscar/scratch/dlinsley/imagenet_tfrecords')
    p.add_argument('--batch_size', type=int, default=1024,
                   help='Global batch (same as training: batch_size * num_devices)')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--repeated_aug', action='store_true', default=True)
    p.add_argument('--seconds', type=float, default=60.0,
                   help='Duration to run the loader')
    p.add_argument('--max_intraop', type=int, default=1,
                   help='options.threading.max_intra_op_parallelism (current default: 1)')
    p.add_argument('--threadpool_size', type=int, default=48,
                   help='options.threading.private_threadpool_size')
    p.add_argument('--warmup', type=int, default=3,
                   help='Batches to discard before measurement')
    args = p.parse_args()

    # Monkey-patch TFRecordImageNetLoader's options before __iter__
    original_iter = TFRecordImageNetLoader.__iter__

    def patched_iter(self):
        aug = (self.split == 'train')
        options = tf.data.Options()
        options.threading.max_intra_op_parallelism = args.max_intraop
        options.threading.private_threadpool_size = args.threadpool_size
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True

        files_ds = tf.data.Dataset.from_tensor_slices(
            [str(f) for f in self.tfrecord_files]
        )
        if aug:
            files_ds = files_ds.shuffle(len(self.tfrecord_files))

        from src.data.imagenet_tfdata import _parse_tfrecord_fn
        ds = files_ds.interleave(
            lambda f: tf.data.TFRecordDataset(
                f, buffer_size=8 * 1024 * 1024,
                num_parallel_reads=tf.data.AUTOTUNE,
            ),
            cycle_length=min(self.num_parallel_reads, len(self.tfrecord_files)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not aug,
        )
        if aug and self.repeated_aug:
            ds = ds.repeat(3)
        if aug:
            ds = ds.shuffle(self.shuffle_buffer)
        ds = ds.map(
            lambda x: _parse_tfrecord_fn(x, self.image_size, aug),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.ignore_errors(log_warning=True)
        ds = ds.batch(self.batch_size, drop_remainder=self.drop_remainder)
        ds = ds.with_options(options)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        for images, labels in ds:
            yield images.numpy(), labels.numpy()

    TFRecordImageNetLoader.__iter__ = patched_iter

    print(f"TFRecord throughput test")
    print(f"  dir={args.tfrecord_dir}")
    print(f"  batch_size={args.batch_size}  image_size={args.image_size}")
    print(f"  max_intraop={args.max_intraop}  threadpool_size={args.threadpool_size}")
    print(f"  target duration={args.seconds:.0f}s  warmup={args.warmup} batches")
    print()

    loader = TFRecordImageNetLoader(
        tfrecord_dir=args.tfrecord_dir,
        split='train',
        batch_size=args.batch_size,
        image_size=args.image_size,
        repeated_aug=args.repeated_aug,
    )

    it = iter(loader)

    # Warmup
    for _ in range(args.warmup):
        imgs, lbls = next(it)
    print(f"Warmup done. Image batch shape: {imgs.shape}, dtype: {imgs.dtype}")

    start = time.perf_counter()
    deadline = start + args.seconds
    n_batches = 0
    last_report = start
    while time.perf_counter() < deadline:
        imgs, lbls = next(it)
        n_batches += 1
        now = time.perf_counter()
        if now - last_report > 10.0:
            elapsed = now - start
            print(f"  [{elapsed:5.1f}s] {n_batches} batches, "
                  f"{n_batches / elapsed:.2f} batches/s, "
                  f"{n_batches * args.batch_size / elapsed:.0f} samples/s")
            last_report = now

    elapsed = time.perf_counter() - start
    bps = n_batches / elapsed
    sps = n_batches * args.batch_size / elapsed
    ms_per_batch = 1000.0 / bps
    print()
    print(f"=== RESULTS ===")
    print(f"  Elapsed:      {elapsed:.2f}s")
    print(f"  Batches:      {n_batches}")
    print(f"  Batches/sec:  {bps:.2f}")
    print(f"  Samples/sec:  {sps:.0f}")
    print(f"  ms/batch:     {ms_per_batch:.1f}")
    print()
    print("Interpretation:")
    print(f"  Current training step time: ~597 ms/step")
    if ms_per_batch < 400:
        print(f"  → Data loader is FAST enough ({ms_per_batch:.0f} ms << 597 ms). Not the bottleneck.")
    elif ms_per_batch < 597:
        print(f"  → Data loader is CLOSE to step time ({ms_per_batch:.0f} ms ~ 597 ms).")
        print(f"     May contribute meaningfully. Try bumping --max_intraop.")
    else:
        print(f"  → Data loader IS the bottleneck ({ms_per_batch:.0f} ms >= 597 ms).")


if __name__ == '__main__':
    main()
