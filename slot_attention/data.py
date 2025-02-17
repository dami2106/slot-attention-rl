"""Data utils for loading a dataset from .npy files."""
import tensorflow as tf
import numpy as np


def preprocess_npy_image(image, resolution, apply_augmentation=False):
    """Preprocess an image from the npy dataset."""
    image = tf.cast(image, dtype=tf.float32)
    image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1]
    image = tf.image.resize(image, resolution, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.clip_by_value(image, -1., 1.)

    if apply_augmentation:
        image = augment_image(image)

    return {"image": image}


def augment_image(image):
    """Apply random augmentation to the image."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image


def build_npy_dataset(npy_file, resolution=(128, 128), split="train", shuffle=True,
                      num_eval_examples=2, apply_augmentation=True):
    """Load a dataset from a npy file and preprocess it with train/eval split."""
    images = np.load(npy_file)  # Shape: (N, 64, 64, 3)
    total_samples = images.shape[0]

    # Define train/eval split (default: 80% train, 20% eval)
    train_split = total_samples - num_eval_examples

    if split == "train":
        images = images[:train_split]
    elif split == "train_eval":
        images = images[train_split:]  # Subset of training for evaluation
    elif split == "eval":
        images = images[train_split:]
    else:
        raise ValueError(f"Invalid split '{split}', choose from ['train', 'train_eval', 'eval']")

    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(lambda x: preprocess_npy_image(x, resolution, apply_augmentation))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(images))

    return ds


def build_npy_iterator(npy_file, batch_size, split="train", **kwargs):
    """Create an infinite iterator for training/evaluation batches."""
    ds = build_npy_dataset(npy_file, split=split, **kwargs)
    ds = ds.repeat(-1)  # Infinite looping
    ds = ds.batch(batch_size, drop_remainder=True)
    return iter(ds)
