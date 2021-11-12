import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

import torch
torch.manual_seed(0)

from classifiers.cnn_classifier import ImageNetClassifier
from tests.config import WORKING_DIR
from utils.dataset import load_test_image, preprocess_image, normalize_image
import matplotlib


matplotlib.use('Agg')

module = __import__(f"{WORKING_DIR}.explainers", fromlist=[
    'get_gradient', 'get_integrated_gradients', 'get_smoothgrad', 'aggregate_attribution', 'normalize_attribution', 'plot_attributions'])

image = load_test_image()
image_preprocessed = preprocess_image(image)
image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

model = ImageNetClassifier()

y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
assert y_pred == torch.tensor([13])
assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)


def test_get_gradient():
    gradient = module.get_gradient(model, torch.zeros_like(image_preprocessed_norm))
    assert gradient.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(gradient[0, 0, 0, 0], torch.tensor(0.0003), atol=1e-4)
    assert torch.allclose(gradient.mean(), torch.tensor(0.0001), atol=1e-4)
    assert torch.allclose(gradient.sum(), torch.tensor(22.3855), atol=1e-3)

    gradient = module.get_gradient(model, image_preprocessed_norm)
    assert torch.allclose(gradient[0, 0, 0, 0], torch.tensor(0.0004), atol=1e-4)
    assert torch.allclose(gradient.mean(), torch.tensor(0.0004), atol=1e-4)
    assert torch.allclose(gradient.sum(), torch.tensor(56.3818), atol=1e-3)


def test_get_integrated_gradients():
    ig = module.get_integrated_gradients(model, image_preprocessed_norm)
    assert ig.shape == torch.Size([1, 3, 224, 224])
    assert ig.dtype == torch.double
    assert torch.allclose(ig[0, 0, 0, 0], torch.tensor(0.0011).double(), atol=1e-4)
    assert torch.allclose(ig.mean(), torch.tensor(0.0).double(), atol=1e-4)
    assert torch.allclose(ig.sum(), torch.tensor(0.9821).double(), atol=1e-3)


def test_get_smoothgrad():
    smoothgrad = module.get_smoothgrad(model, image_preprocessed_norm)
    assert smoothgrad.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(smoothgrad[0, 0, 0, 0], torch.tensor(0.0002), atol=1e-4)
    assert torch.allclose(smoothgrad.mean(), torch.tensor(0.0003), atol=1e-4)
    assert torch.allclose(smoothgrad.sum(), torch.tensor(47.2648), atol=1e-3)


def test_aggregate_attribution():
    attribution = torch.arange(0, 3*64*64, dtype=torch.float).view(1, 3, 64, 64)
    agg = module.aggregate_attribution(attribution)
    assert agg.shape == torch.Size([64, 64])
    assert torch.allclose(agg[0, 0], torch.tensor(12288.0), atol=1e-3)
    assert torch.allclose(agg.mean(), torch.tensor(18430.5), atol=1e-3)

    attribution = torch.arange(-9, 3*64*64-9, dtype=torch.float).view(1, 3, 64, 64)
    agg = module.aggregate_attribution(attribution)
    assert agg.shape == torch.Size([64, 64])
    assert torch.allclose(agg[0][0], torch.tensor(12261.0), atol=1e-3)
    assert torch.allclose(agg.mean(), torch.tensor(18403.5), atol=1e-3)


def test_normalize_attribution():
    attribution = torch.arange(0, 64*64, dtype=torch.float).view(64, 64)
    agg = module.normalize_attribution(attribution)
    assert agg.shape == torch.Size([64, 64])
    assert torch.allclose(agg.min(), torch.tensor(0.0))
    assert torch.allclose(agg.max(), torch.tensor(1.0))
    assert torch.allclose(agg[1, 1], torch.tensor(0.0159), atol=1e-3)
    assert torch.allclose(agg.mean(), torch.tensor(0.5))

    attribution = torch.arange(-9, 64*64-9, dtype=torch.float).view(64, 64)
    agg = module.normalize_attribution(attribution)
    assert agg.shape == torch.Size([64, 64])
    assert torch.allclose(agg.min(), torch.tensor(0.0))
    assert torch.allclose(agg.max(), torch.tensor(1.0))
    assert torch.allclose(agg[1, 1], torch.tensor(0.0137), atol=1e-3)
    assert torch.allclose(agg.mean(), torch.tensor(0.4989), atol=1e4)


def test_plot_attributions():
    from utils.styled_plot import plt
    image = torch.linspace(0, 1, 3*64*64).view(3, 64, 64)
    attributions = [
        torch.linspace(0, 1, 64*64).view(64, 64),
        torch.linspace(0, 1, 64*64).view(64, 64).T
    ]
    method_names = ['Method1', 'Method2']
    module.plot_attributions(plt, image, attributions, method_names)

    fig = plt.gcf()

    # test if correct number of subplots
    assert len(fig.axes) == 1 + len(attributions)

    # test if image is plotted correctly:
    aximg = [x for x in fig.axes[0].get_children() if isinstance(x, matplotlib.image.AxesImage)][0]
    assert torch.allclose(torch.tensor(aximg.get_array()).permute(2, 0, 1)[0], image[0])

    # test if last attribution is plotted correctly:
    aximg = [x for x in fig.axes[-1].get_children() if isinstance(x, matplotlib.image.AxesImage)][0]
    assert torch.allclose(torch.tensor(aximg.get_array())[0], attributions[-1][0])

    # test if method names are used as titles of subplots
    for i, ax in enumerate(fig.axes[1:]):
        assert ax.get_title() == method_names[i]


if __name__ == "__main__":
    test_get_gradient()
    test_get_integrated_gradients()
    test_get_smoothgrad()
    test_aggregate_attribution()
    test_normalize_attribution()
    test_plot_attributions()
