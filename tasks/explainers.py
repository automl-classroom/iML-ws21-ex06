import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
from utils.styled_plot import plt
from utils.dataset import load_test_image, preprocess_image, normalize_image, convert_idx_to_label
from classifiers.cnn_classifier import ImageNetClassifier
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel


def get_gradient(model, image):
    """
    Uses captum's 'Saliency' method to generate a saliency map based on the gradient w.r.t. the model's prediction as the target. See also: https://captum.ai/api/saliency.html

    Parameters:
        model (ImageNetClassifier): Image classification model. Has a 'predict' method that returns the predicted label index for an image.
        image (torch.tensor): Single image with shape (1, 3, ?, ?).

    Returns:
        attribution (torch.tensor): The gradient, of the same shape as the image.
    """
    return None


def get_integrated_gradients(model, image):
    """
    Uses captum's IntegratedGradients method to generate a attribution map w.r.t. the model's prediction as the target. Uses zeros (black image) as the baseline, that are normalized using 'normalize_image'.
    See also: https://captum.ai/api/integrated_gradients.html

    Parameters:
        model (ImageNetClassifier): Image classification model. Has a 'predict' method that returns the predicted label index for an image.
        image (torch.tensor): Single image with shape (1, 3, ?, ?).

    Returns:
        attributions (torch.tensor): The integrated gradients, of the same shape as the image.
    """
    return None


def get_smoothgrad(model, image, num_samples=10, stdevs=0.3):
    """
    Uses captum's NoiseTunnel and Saliency method to generate a saliency map using SmoothGrad, based on the gradient w.r.t. the model's prediction as the target. See also: https://captum.ai/api/noise_tunnel.html

    Parameters:
        model (ImageNetClassifier): Image classification model. Has a 'predict' method.
        image (torch.tensor): Single image with shape (1, 3, ?, ?).
        num_samples (int): Number of SmoothGrad samples to use.
        stdevs (float): Standard deviation for the smoothgrad samples

    Returns:
        attributions (torch.tensor): The gradient, of the same shape as the image.
    """
    return None


def aggregate_attribution(attribution):
    """
    Aggregates the channel dimension of a feature attribution tensor via summation.
    Additionally, removes the batch dimension (dim 0).

    Parameters:
        attribution (torch.tensor): Feature attribution of shape (1, 3, ?, ?)

    Returns:
        attribution (torch.tensor): The aggregated attribution of shape (?, ?)
    """
    return None


def normalize_attribution(attribution):
    """
    Takes the absolute value of the feature attribution, then normalizes to the range [0, 1] by first subtracting the minimum and then dividing by the maximum afterwards.

    Parameters:
        attribution (torch.tensor): Feature attribution of shape (?, ?)

    Returns:
        attribution (torch.tensor): The absolute, normalized attribution of shape (?, ?)
    """
    return None


def plot_attributions(plt, image, attributions, method_names):
    """
    Visualizes an image and a list of corresponding feature attributions by plotting them in a single row.

    Parameters:
        image (torch.tensor): Single image with shape (3, ?, ?)
        attributions (List[torch.tensor]): List of feature attributions, each of shape (?, ?)
        method_names (List[str]): List of method names corresponding to the attributions. Used as subfigure titles.

    Returns:
        None

    Hint: iterate over the axes. Use imshow() to plot images. Matplotlib expects a channels last format. Optionally turn of the axis labeling using ax.axis('off') .
    """
    fig, axes = plt.subplots(len(attributions) + 1, 1)



if __name__ == "__main__":
    image = load_test_image()
    image_preprocessed = preprocess_image(image)
    image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

    model = ImageNetClassifier()
    
    y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
    print(f'Predicted class: "{convert_idx_to_label(y_pred.item())}". Confidence: {y_prob.item() * 100:.2f}%')
    assert y_pred == torch.tensor([13])
    assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)

    print('Run `get_gradient` ...')
    gradient = get_gradient(model, image_preprocessed_norm)

    print('Run `get_integrated_gradients` ...')
    ig = get_integrated_gradients(model, image_preprocessed_norm)

    print('Run `get_smoothgrad` ...')
    smoothgrad = get_smoothgrad(model, image_preprocessed_norm)

    print('Run `plot_attributions` ...')
    attributions = [gradient, ig, smoothgrad]
    attributions = [aggregate_attribution(attr) for attr in attributions]
    attributions = [normalize_attribution(attr) for attr in attributions]
    plot_attributions(plt, image_preprocessed, attributions, ['Gradient', 'IG', 'SmoothGrad'])
    plt.show()
