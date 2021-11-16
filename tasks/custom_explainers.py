import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
from utils.styled_plot import plt
from utils.dataset import load_test_image, preprocess_image, normalize_image, convert_idx_to_label
from classifiers.cnn_classifier import ImageNetClassifier
from tasks.explainers import plot_attributions, aggregate_attribution, normalize_attribution


def get_custom_gradient(model, image, absolute=False):
    """
    Generates a saliency map based on the input gradient w.r.t. the model's prediction as the target.

    Parameters:
        model (ImageNetClassifier): Image classification model. Has a 'predict' method that returns the predicted label index for an image.
        image (torch.tensor): Single image with shape (1, 3, ?, ?).
        absolute (bool): If True, return the absolute value of the gradients. If False, return the signed gradients.

    Returns:
        attribution (torch.tensor): The gradient, of the same shape as the image.

    Hint: Use torch.autograd.grad . The model is a torch.nn.Module, so you can call model(x) to get the network's outputs.
    """
    return None


def get_path(image, baseline, num_samples):
    """
    Generate a attribution map based on the Integrated Gradients method, w.r.t. the model's prediction.
    Uses zeros (black image) as the baseline, that are normalized using 'normalize_image'.

    Parameters:
        image (torch.tensor): Single image with shape (1, 3, ?, ?).
        baseline (torch.tensor): Baseline image with same shape as image.
        num_samples (int): The nuber of samples on the path.

    Returns:
        path (List[torch.tensor]): A list of length num_samples, containing the images on the path starting from the baseline and ending with the image.

    Hint: Create alphas using torch.linspace.
    """
    return None


def get_custom_integrated_gradients(model, image, num_samples):
    """
    Generate an attribution map based on the Integrated Gradients method, w.r.t. the model's prediction.
    Uses zeros (black image) as the baseline, that are normalized using 'normalize_image'.

    Parameters:
        model (ImageNetClassifier): Image classification model. Has a 'predict' method that returns the predicted label index for an image.
        image (torch.tensor): Single image with shape (1, 3, ?, ?).
        num_samples (int):
    Returns:
        attributions (torch.tensor): The integrated gradients, of the same shape as the image.

    Hint: Iterate over the path of images, remember what you did in 'get_custom_gradient'.
    """
    return None


if __name__ == "__main__":
    image = load_test_image()
    image_preprocessed = preprocess_image(image)
    image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

    model = ImageNetClassifier()
    
    y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
    print(f'Predicted class: "{convert_idx_to_label(y_pred.item())}". Confidence: {y_prob.item() * 100:.2f}%')
    assert y_pred == torch.tensor([13])
    assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)

    print('Run `get_custom_gradient` ...')
    gradient = get_custom_gradient(model, image_preprocessed_norm.clone())
    gradient_abs = get_custom_gradient(model, image_preprocessed_norm.clone(), absolute=True)

    print('Run `get_custom_integrated_gradients` ...')
    ig = get_custom_integrated_gradients(model, image_preprocessed_norm.clone(), num_samples=50)

    print('Run `plot_attributions` ...')
    attributions = [gradient, gradient_abs, ig]
    attributions = [aggregate_attribution(attr) for attr in attributions]
    attributions = [normalize_attribution(attr) for attr in attributions]
    plot_attributions(plt, image_preprocessed, attributions, ['Gradient', 'abs. Gradient', 'IG'])
    plt.show()
