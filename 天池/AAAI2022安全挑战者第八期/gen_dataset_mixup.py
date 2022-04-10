"""
Import necessary libraries to train a network using mixup
The code is mainly developed using the PyTorch library
"""
import numpy as np
import pickle
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader



"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Dataset and Dataloader creation
All data are downloaded found via Graviti Open Dataset which links to CIFAR-10 official page
The dataset implementation is where mixup take place
"""

class CIFAR_Dataset(Dataset):
    def __init__(self, data_dir, train, transform):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []

        # Loading all the data depending on whether the dataset is training or testing
        if self.train:
            print('train')
            for i in range(5):
                with open(data_dir + 'data_batch_' + str(i+1), 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    self.data.append(entry['data'])
                    self.targets.extend(entry['labels'])
        else:
            print('test')
            with open(data_dir + 'test_batch', 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        # Reshape it and turn it into the HWC format which PyTorch takes in the images
        # Original CIFAR format can be seen via its official page
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Create a one hot label
        label = torch.zeros(10)
        label[self.targets[idx]] = 1.

        # Transform the image by converting to tensor and normalizing it
        if self.transform:
            image = transform(self.data[idx])

        # If data is for training, perform mixup, only perform mixup roughly on 1 for every 5 images

        print('Mixup')
        # Choose another image/label randomly
        mixup_idx = random.randint(0, len(self.data)-1)
        mixup_label = torch.zeros(10)
        mixup_label[self.targets[mixup_idx]] = 1.
        if self.transform:
            mixup_image = transform(self.data[mixup_idx])

        # Select a random number from the given beta distribution
        # Mixup the images accordingly
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        image = lam * image + (1 - lam) * mixup_image
        label = lam * label + (1 - lam) * mixup_label

        return image, label

"""
Define the hyperparameters, image transform components, and the dataset/dataloaders
"""
transform = transforms.Compose(
    [transforms.ToTensor()])

BATCH_SIZE = 64
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30


train_dataset = CIFAR_Dataset('./data/cifar-10-batches-py/', 0, transform)
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
images = []
soft_labels = []
for image, label in train_dataset:
    image = np.array(image * 255).astype(np.uint8)
    images.append(image)
    label= label.numpy()
    soft_labels.append(label)
images = np.array(images)
images = images.transpose(0, 2, 3, 1)
soft_labels = np.array(soft_labels)
print(images.shape, images.dtype, soft_labels.shape, soft_labels.dtype)

np.save('ADVImage/data_mixup_020.npy', images)
np.save('ADVImage/label_mixup_020.npy', soft_labels)
# test_dataset = CIFAR_Dataset('./data/cifar-10-batches-py/', 0, transform)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# images_ = images.transpose((0, 2, 3, 1))
# image1 = images[100]
# cv.imshow('test', image1)
# cv.waitKey()
# cv.imwrite('test1_aug.jpg', image1)
#
# print()


# 50000,3,32,32
#
# 50000,32,32,3

