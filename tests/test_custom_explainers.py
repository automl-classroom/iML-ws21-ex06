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

module = __import__(f"{WORKING_DIR}.custom_explainers", fromlist=[
    'get_custom_gradient', 'get_path', 'get_custom_integrated_gradients'])

image = load_test_image()
image_preprocessed = preprocess_image(image)
image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

model = ImageNetClassifier()

y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
assert y_pred == torch.tensor([13])
assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)


def test_get_custom_gradient():
    # test without absolute
    gradient = module.get_custom_gradient(model, torch.zeros_like(image_preprocessed_norm.clone()))
    assert gradient.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(gradient[0, 0, 0, 0], torch.tensor(0.0003), atol=1e-4)
    assert torch.allclose(gradient.mean(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(gradient.sum(), torch.tensor(0.0177), atol=1e-3)

    gradient = module.get_custom_gradient(model, image_preprocessed_norm.clone())
    assert torch.allclose(gradient[0, 0, 0, 0], torch.tensor(-0.0004), atol=1e-4)
    assert torch.allclose(gradient.mean(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(gradient.sum(), torch.tensor(-0.0659), atol=1e-4)

    # test with absolute
    gradient = module.get_custom_gradient(model, torch.zeros_like(image_preprocessed_norm.clone()), absolute=True)
    assert gradient.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(gradient[0, 0, 0, 0], torch.tensor(0.0003), atol=1e-4)
    assert torch.allclose(gradient.mean(), torch.tensor(0.0001), atol=1e-4)
    assert torch.allclose(gradient.sum(), torch.tensor(22.3855), atol=1e-3)

    gradient = module.get_custom_gradient(model, image_preprocessed_norm.clone(), absolute=True)
    assert torch.allclose(gradient[0, 0, 0, 0], torch.tensor(0.0004), atol=1e-4)
    assert torch.allclose(gradient.mean(), torch.tensor(0.0004), atol=1e-4)
    assert torch.allclose(gradient.sum(), torch.tensor(56.3818), atol=1e-3)


def test_get_path():
    image = image_preprocessed_norm.clone()
    baseline = torch.zeros_like(image)

    num_samples = 3
    path = module.get_path(image_preprocessed_norm.clone(), baseline, num_samples)
    assert len(path) == num_samples
    for p in path:
        assert p.shape == torch.Size([1, 3, 224, 224])
    assert torch.equal(path[0], baseline)
    assert torch.equal(path[-1], image)
    assert torch.allclose(path[1][0,0,0,:3], torch.tensor([0.1912, 0.1826, 0.1740]), atol=1e-4)

    num_samples = 10
    path = module.get_path(image_preprocessed_norm.clone(), baseline, num_samples)
    assert len(path) == num_samples
    for p in path:
        assert p.shape == torch.Size([1, 3, 224, 224])
    assert torch.equal(path[0], baseline)
    assert torch.equal(path[-1], image)
    assert torch.allclose(path[1][0,0,0,:3], torch.tensor([0.0425, 0.0406, 0.0387]), atol=1e-4)


def test_get_custom_integrated_gradients():
    ig = module.get_custom_integrated_gradients(model, image_preprocessed_norm.clone(), 10)
    assert ig.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(ig[0, 0, 0, 0], torch.tensor(0.0009), atol=1e-4)
    assert torch.allclose(ig.mean(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(ig.sum(), torch.tensor(1.5907), atol=1e-3)

    ig = module.get_custom_integrated_gradients(model, image_preprocessed_norm.clone(), 50)
    assert ig.shape == torch.Size([1, 3, 224, 224])
    assert torch.allclose(ig[0, 0, 0, 0], torch.tensor(0.0010), atol=1e-4)
    assert torch.allclose(ig.mean(), torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(ig.sum(), torch.tensor(0.9154), atol=1e-3)


if __name__ == "__main__":
    test_get_custom_gradient()
    test_get_path()
    test_get_custom_integrated_gradients()
