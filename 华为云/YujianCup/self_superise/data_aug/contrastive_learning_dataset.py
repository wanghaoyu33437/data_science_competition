from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from view_generator import ContrastiveLearningViewGenerator

from self_superise.exceptions.exceptions import InvalidDatasetSelection
from data_helper import MyDataset
import sys
sys.path.append('../../')

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder
    def get_dataset(self, name, n_views):
        train_data_dir = './data/new_split_data/train'
        val_data_dir = './data/new_split_data/val'
        train_dataset = MyDataset(train_data_dir, 'train', input_size=384, resize=400)
        val_dataset = MyDataset(val_data_dir, 'val', input_size=384, resize=400)
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
