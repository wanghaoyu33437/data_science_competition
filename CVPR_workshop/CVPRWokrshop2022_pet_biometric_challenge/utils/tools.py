import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
from torch import optim

from utils.patch_utils import  un_normalize
from torchvision.utils import save_image as t_save_img

def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20
    config["data_path"] = ""
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = ""
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    elif config["dataset"] == "imagenet_ResNet50":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_100":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "little_ImageNet":
        config["topK"] = 50
        config["n_class"] = 10
    elif config["dataset"] == "little_ImageNet_CIBA":
        config["topK"] = 50
        config["n_class"] = 10
    elif config["dataset"] == "little_ImageNet_BadNets":
        config["topK"] = 50
        config["n_class"] = 10
    elif config["dataset"] == "ImageNet_100_posion":
        config["topK"] = 100
        config["n_class"] = 100
    elif config["dataset"] == "MS-COCO_VGG11":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_ResNet50":
        config["topK"] = 1000
        config["n_class"] = 80
    # 投毒样本数据集CSQ-HashNet
    elif config["dataset"] == "ImageNet_HashNet_ResNet50_M1":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_ResNet50_M2":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_HashNet_ResNet50_M2":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_ResNet50_M3":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_ResNet50_M4":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_ResNet50_M5":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_HashNet_ResNet50_M5":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_VGG11_M6":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_VGG11_M7":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_VGG11_M8":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_HashNet_VGG11_M9":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "MS-COCO_HashNet_ResNet50_M10":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_HashNet_ResNet50_M11":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_HashNet_ResNet50_M12":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_CSQ_VGG11_M15":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_HashNet_VGG11_M14":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_HashNet_VGG11_M15":
        config["topK"] = 1000
        config["n_class"] = 80

    # 投毒样本数据集HashNet-CSQ
    elif config["dataset"] == "ImageNet_CSQ_ResNet50_M1":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_ResNet50_M2":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_ResNet50_M3":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_CSQ_ResNet50_M2":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_CSQ_ResNet50_M3":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_ResNet50_M4":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_ResNet50_M5":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_VGG11_M6":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_VGG11_M7":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_VGG11_M8":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "ImageNet_CSQ_VGG11_M9":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_CSQ_VGG11_M6":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_CSQ_VGG11_M7":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_CSQ_VGG11_M8":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "M_ImageNet_CSQ_VGG11_M9":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "MS-COCO_CSQ_ResNet50_M10":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_CSQ_ResNet50_M11":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_CSQ_ResNet50_M12":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_CSQ_VGG11_M13":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_CSQ_VGG11_M14":
        config["topK"] = 1000
        config["n_class"] = 80
    elif config["dataset"] == "MS-COCO_CSQ_VGG11_M15":
        config["topK"] = 1000
        config["n_class"] = 80


    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    config["optimizer"] = {
        "type": optim.RMSprop,
        "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}}

    config["device"] = torch.device("cuda:0")
    return config

class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


# 这里是我加上去的，处理patch贴在图片上
class PatchedImageList(object):
    def __init__(self, data_path, image_list, transform, patch_path):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.patch = self.patch_transform(Image.open(patch_path).convert('RGB'))

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # patched_img = img
        x_location = 200 - self.patch.shape[1]
        y_location = 200 - self.patch.shape[2]
        img[:, x_location: x_location + self.patch.shape[1], y_location: y_location + self.patch.shape[2]] = self.patch
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MyCIFAR10(root='/dataset/cifar/',
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root='/dataset/cifar/',
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root='/dataset/cifar/',
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def inject_backdoor(images, applied_patch, mask):
    new_shape = list(mask.shape)
    new_shape.insert(0, images.shape[0])
    perturbated_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul(1 - mask.expand(new_shape ).type(torch.FloatTensor),  images.type(torch.FloatTensor))
    return perturbated_images


def return_data(config, patch, mask):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    portion = config['portion']
    batch = config['batch_size']

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        count = 0
        path = './save/CSQ/test'

        if data_set == "train_set":
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                          batch_size=data_config[data_set]["batch_size"],
                                                          shuffle=True, num_workers=4)

            for train_image, train_label, idx in dset_loaders[data_set]:
                if count < len(dset_loaders[data_set]) * portion :
                    train_image = inject_backdoor(train_image, patch, mask)

                    for i in range(train_image.shape[0]):
                        if i == 0:
                            t_save_img(un_normalize(train_image[0]),
                                       path +'/pert_' + str(data_set) + '_' + str(count) + '.JPEG')
                    print("Data: {} - Progress: {} - idx: {}".format(data_set, "poison", count+1))
                else:
                    train_image = train_image
                    for i in range(train_image.shape[0]):
                        if i == 0:
                            t_save_img(un_normalize(train_image[0]),
                                       path + '/nomal_' + str(data_set) + '_'  + str(count) + '.JPEG')

                    print("Data: {} - Progress: {} - idx: {}".format(data_set, "nomal", count+1))

                count = count + 1
        else:
            count = 0
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                          batch_size=data_config[data_set]["batch_size"],
                                                          shuffle=False, num_workers=4)
            for train_image, train_label, idx in dset_loaders[data_set]:
                if count < len(dset_loaders[data_set]) * portion :
                    train_image = inject_backdoor(train_image, patch, mask)
                    for i in range(train_image.shape[0]):
                        if i == 0:
                            t_save_img(un_normalize(train_image[0]),
                                       path + '/pert_' + str(data_set) + '_'  + str(count) + '.JPEG')

                    print("Data: {} - Progress: {} - idx: {}".format(data_set, "poison", count+1))
                else:
                    train_image = train_image
                    print("Data: {} - Progress: {} - idx: {}".format(data_set, "nomal", count+1))
                    for i in range(train_image.shape[0]):
                        if i == 0:
                            t_save_img(un_normalize(train_image[0]),
                                       path + '/nomal_' + str(data_set) + '_' + str(count) + '.JPEG')

                count = count + 1

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

def compute_result_(images, labels, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls in tqdm(zip(images, labels)):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img .to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        if data_set == "train_set":
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                          batch_size=data_config[data_set]["batch_size"],
                                                          shuffle=True, num_workers=4)
        else:
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                          batch_size=data_config[data_set]["batch_size"],
                                                          shuffle=False, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

def load_init_data(args):
    dsets = {}
    data_args = args["data"]
    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(args["data_path"],
                                    open(data_args[data_set]["list_path"]).readlines(),
                                    transform=image_transform(args["resize_size"], args["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))

    return dsets["train_set"], dsets["test"], dsets["database"]

def apply_loader(config, train_data, test_data, database_data):
    data_config = config["data"]
    train_data_loader = util_data.DataLoader(train_data,batch_size=data_config["train_set"]["batch_size"],
                                                          shuffle=True, num_workers=4)
    test_data_loader = util_data.DataLoader(test_data,batch_size=data_config["test"]["batch_size"],
                                                          shuffle=True, num_workers=4)
    database_data_loader = util_data.DataLoader(database_data,batch_size=data_config["database"]["batch_size"],
                                                          shuffle=True, num_workers=4)
    return train_data_loader, test_data_loader, database_data_loader



