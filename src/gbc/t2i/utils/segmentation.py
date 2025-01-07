# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import numpy as np
from skimage.segmentation import felzenszwalb


def otsu_1d(array):
    """
    Find the Otsu threshold for a 1D array using vectorized operations.

    Parameters:
    array (array-like): 1D input array

    Returns:
    float: Otsu threshold value
    """

    # Sort the array
    sorted_array = np.sort(array)
    total_weight = len(sorted_array)

    # Calculate cumulative sums
    cumsum = np.cumsum(sorted_array)

    # Calculate class probabilities
    weight_left = np.arange(1, total_weight)
    weight_right = total_weight - weight_left

    # Calculate class means
    # Exclude the last element
    mean_left = cumsum[:-1] / weight_left
    mean_right = (cumsum[-1] - cumsum[:-1]) / weight_right

    # Calculate inter-class variance
    variance = weight_left * weight_right * (mean_left - mean_right) ** 2

    # Find the threshold that maximizes the inter-class variance
    idx = np.argmax(variance)
    threshold = (sorted_array[idx] + sorted_array[idx + 1]) / 2

    return threshold


def felzenszwalb_segmentation_intersection(image: np.ndarray, mask: np.ndarray):
    if image.ndim == 2:  # Single channel, no need for intersection
        segmented_image = felzenszwalb(image)
        return segmented_image[mask]

    # Segment each channel separately
    segmented_channels = [felzenszwalb(image[..., i]) for i in range(image.shape[-1])]

    # Initialize combined labels array with zeros
    combined_labels = np.zeros_like(segmented_channels[0], dtype=int)

    # Define a multiplier for each channel to create unique labels when combined
    max_label_per_channel = [
        np.max(segmented_channel) + 1 for segmented_channel in segmented_channels
    ]
    label_multipliers = np.cumprod([1] + max_label_per_channel[:-1])

    # Combine labels from each channel by
    # creating a unique label based on each channel's segmented regions
    for i, segmented_channel in enumerate(segmented_channels):
        combined_labels += segmented_channel * label_multipliers[i]
    combined_labels = combined_labels[mask]

    # Get unique labels present in the combined labels array
    unique_labels, inverse = np.unique(combined_labels, return_inverse=True)

    # Create a new compressed label array
    compressed_labels = np.arange(len(unique_labels))
    compressed_combined_labels = compressed_labels[inverse].reshape(
        combined_labels.shape
    )

    # Apply the mask and return the compressed segmented image
    return compressed_combined_labels


def felzenszwalb_segmentation(image: np.ndarray, mask: np.ndarray):
    segmented_image = felzenszwalb(image)
    return segmented_image[mask]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # Generate a sample bimodal distribution
    np.random.seed(0)
    data = np.concatenate([np.random.normal(-2, 1, 1000), np.random.normal(2, 1, 1000)])

    # Find the Otsu threshold
    threshold = otsu_1d(data)

    print(f"Otsu threshold: {threshold}")

    img_size = 32
    x, y = np.indices((img_size, img_size))

    center1 = (7, 6)
    center2 = (10, 12)
    center3 = (16, 15)
    center4 = (6, 18)

    radius1, radius2, radius3, radius4 = 5, 4, 3, 2

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1**2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2**2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3**2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4**2

    img = circle1 + circle2 + circle3 + circle4
    mask = img.astype(bool)
    img = img.astype(float)
    img += 1 + 0.2 * np.random.randn(*img.shape)

    img_for_segmentation = (
        np.stack([circle1, circle2, circle3, circle4])
        .astype(float)
        .transpose([1, 2, 0])
    )
    img_for_segmentation += 1 + 0.2 * np.random.randn(*img_for_segmentation.shape)
    print(img_for_segmentation.shape)

    cluster_labels = felzenszwalb_segmentation_intersection(img_for_segmentation, mask)
    label_im = np.full(img.shape, -1.0)
    label_im[mask] = cluster_labels
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].matshow(img)
    axs[1].matshow(label_im)
    plt.show()
