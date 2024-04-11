from typing import List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from subspaces import VectorSpace


def vizualize_images(vector_space: VectorSpace,
                     components: Union[int, slice],
                     image_shape: Union[Tuple, List]) -> None:
    """
    Visualizes the vector at index `idx` as a 2D image.
    """
    if type(image_shape) is list:
        image_shape = tuple(image_shape)

    images = vector_space[components].detach()
    if images.dim() == 1:
        images = images.unsqueeze(0)

    images = images.reshape((-1,)+image_shape).numpy()

    N = images.shape[0]  # Number of images
    cols = int(np.ceil(np.sqrt(N)))  # Determine the number of columns (and rows) for the plot
    rows = N // cols + (N % cols > 0)  # Ensure enough rows to display all images

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    ax = np.atleast_2d(ax)  # Ensure ax is 2D

    for i in range(rows * cols):
        row, col = divmod(i, cols)
        if i < N:
            if rows == 1 or cols == 1:  # Check if the subplot array is 1D
                ax_flat = ax.flatten()  # Flatten to 1D
                ax_flat[i].imshow(images[i], cmap='gray')
                ax_flat[i].axis('off')
            else:
                ax[row, col].imshow(images[i], cmap='gray')
                ax[row, col].axis('off')
        else:
            if rows == 1 or cols == 1:  # Hide empty subplots
                ax_flat = ax.flatten()
                ax_flat[i].axis('off')
            else:
                ax[row, col].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Simple test script
    import torch
    from torchvision import transforms as T
    from torchvision.datasets import MNIST

    from subspaces.transform import PCATransform
    from subspaces.generators import IdentityGenerator

    dataset = MNIST("~/datasets", download=True, train=False,
                    transform=T.Compose([T.PILToTensor(), torch.flatten]))
    generator = IdentityGenerator()

    vector_spaces = generator.generate(dataset, batch_size=32)

    pca_transform = PCATransform(n_components=10)

    pca_vector_space = pca_transform.transform(vector_spaces[0])

    vizualize_images(pca_vector_space, 1, image_shape=[28, 28])
    vizualize_images(pca_vector_space, slice(1, 5), image_shape=[28, 28])
