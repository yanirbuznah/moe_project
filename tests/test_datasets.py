# import unittest
from unittest.mock import patch, MagicMock
import torchvision.datasets
import pytest
from datasets_and_dataloaders.utils import get_dataset

@pytest.mark.parametrize("dataset_name", ["mnist","tinyimagenet"])
def test_huggingface_get_dataset(dataset_name):
    with patch('datasets_and_dataloaders.utils.load_dataset') as mock_load_dataset:
        mock_load_dataset.return_value = MagicMock()
        train = get_dataset(dataset_name, True)
        val = get_dataset(dataset_name, False)
        assert isinstance(train, MagicMock)
        assert isinstance(val, MagicMock)


@pytest.mark.parametrize("dataset_name", ["cifar10","cifar100"])
def test_torchvision_get_dataset(dataset_name):
    with patch.object(torchvision.datasets, dataset_name.upper()) as mock_imagefolder:
        mock_imagefolder.return_value = MagicMock()
        train = get_dataset(dataset_name,True)
        val = get_dataset(dataset_name,False)
        assert isinstance(train, MagicMock)
        assert isinstance(val, MagicMock)

def test_invalid_dataset():
    with pytest.raises(NotImplementedError):
        get_dataset('foo',True)

# class TestGetDataset(unittest.TestCase):
#     def test_mnist_size(self):
#         train, val = get_dataset('mnist')
#         self.assertEqual(len(train), 60000)
#         self.assertEqual(len(val), 10000)

#     def test_cifar10_size(self):
#         train, val = get_dataset('cifar10')
#         self.assertEqual(len(train), 50000)
#         self.assertEqual(len(val), 10000)

#     def test_cifar100_size(self):
#         train, val = get_dataset('cifar100')
#         self.assertEqual(len(train), 50000)
#         self.assertEqual(len(val), 10000)

#     def test_tinyimagenet_size(self):
#         train, val = get_dataset('tinyimagenet')
#         self.assertEqual(len(train), 100000)
#         self.assertEqual(len(val), 10000)

#     # def test_imagenet_size(self):
#     #     train, val = get_dataset('imagenet')
#     #     self.assertEqual(len(train), 1000)
#     #     self.assertEqual(len(val), 1000)

#     def test_invalid_dataset(self):
#         with self.assertRaises(NotImplementedError):
#             get_dataset('invalid_dataset')

#     @patch('utils.load_dataset')
#     def test_mnist(self, mock_load_dataset):
#         mock_load_dataset.return_value = MagicMock()
#         train, val = get_dataset('mnist')
#         self.assertIsInstance(train, MagicMock)
#         self.assertIsInstance(val, MagicMock)

#     @patch.object(torchvision.datasets, 'CIFAR10')
#     def test_cifar10(self, mock_cifar10):
#         mock_cifar10.return_value = MagicMock()
#         train, val = get_dataset('cifar10')
#         self.assertIsInstance(train, MagicMock)
#         self.assertIsInstance(val, MagicMock)

#     @patch.object(torchvision.datasets, 'CIFAR100')
#     def test_cifar100(self, mock_cifar100):
#         mock_cifar100.return_value = MagicMock()
#         train, val = get_dataset('cifar100')
#         self.assertIsInstance(train, MagicMock)
#         self.assertIsInstance(val, MagicMock)

#     @patch.object(torchvision.datasets, 'ImageFolder')
#     def test_tinyimagenet(self, mock_imagefolder):
#         mock_imagefolder.return_value = MagicMock()
#         train, val = get_dataset('tinyimagenet')
#         self.assertIsInstance(train, MagicMock)
#         self.assertIsInstance(val, MagicMock)

#     @patch.object(torchvision.datasets, 'ImageFolder')
#     def test_imagenet(self, mock_imagefolder):
#         mock_imagefolder.return_value = MagicMock()
#         train, val = get_dataset('imagenet')
#         self.assertIsInstance(train, MagicMock)
#         self.assertIsInstance(val, MagicMock)

#     def test_not_implemented(self):
#         with self.assertRaises(NotImplementedError):
#             get_dataset('foo')


# if __name__ == '__main__':
#     unittest.main()

