import albumentations  as A
from albumentations.pytorch import ToTensorV2
import numpy as  np
import random
from torchvision import transforms
def gen_transform(input_size = 384,resize = 400,mode='train'):
    if mode =='train':
        # transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((resize, resize)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(20),
            # transforms.ToTensor(),
        # ])
        transform =A.Compose([
            # A.RandomScale(scale_limit=(-0.5,1)),
            # A.RandomResizedCrop(resize,resize,scale=(0.5, 3.0)),
            # A.CenterCrop(resize,resize),
            A.Resize(resize,resize),
            A.CenterCrop(input_size,input_size),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            # A.Rotate(limit=30, p=0.3),
            # A.RandomGridShuffle((3,3),p=0.3),
            # A.GridDistortion(p=0.3),
            # A.Transpose(p=0.3),
            # A.CoarseDropout(30,15,15,p=0.3),
            # A.ElasticTransform(0.5),
        # ]),
        A.OneOf([
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomContrast(p=0.3),
                # SPNoise(p=0.2),
                # A.Equalize(p=0.5),
                # A.GaussNoise(p=0.3),
                # A.GaussianBlur(p=0.3),
                # A.JpegCompression(79, 90, p=0.3),
            ],p=0.5),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    elif  mode =='ssl':
        transform  = A.Compose([
            A.RandomResizedCrop(input_size, input_size),
            # A.Resize(resize, resize),
            # A.CenterCrop(input_size, input_size),
            A.HorizontalFlip(),
            A.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1,p=0.8),
            A.ToGray(p=0.2),
            A.GaussianBlur(),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            # A.RandomScale(scale_limit=(-0.5, 0.5)),
            # A.RandomResizedCrop(resize,resize,scale=(0.5, 3.0)),
            # A.CenterCrop(resize, resize),
            A.Resize(input_size,input_size),
            # A.Resize(resize, resize),
            # A.CenterCrop(input_size, input_size),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
        # transform=transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((224,224)),
        #     # transforms.CenterCrop(384),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        # transform = transforms.Compose([
        #     A.Resize(resize, resize),
            # transforms.ToPILImage(),
            # transforms.Resize((resize,resize)),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
    return transform


class TTA(object):
    def __init__(self,prob=0.5):
        self.prob = prob


    def HorizontalFlip(self,img):
        img = A.HorizontalFlip(1).apply(img)
        return img
    def VerticalFlip(self,img):
        img=A.VerticalFlip(1).apply(img)
        return img
    def Transpose(self, img):
        img = A.Transpose(1).apply(img)
        return img
    def resize(self,img,size):
        img = A.Resize(size,size).apply(img)
        return img




class SPNoise(A.transforms.ImageOnlyTransform):
    """Apply Salt perpper noise to the input image.

    Args:

    Targets:
        image
    Image types:
        uint8,
    """

    def __init__(self, prob=(0.02, 0.2), always_apply=False, p=0.5):
        super(SPNoise, self).__init__(always_apply, p)
        self.prob = prob

    def apply(self, img, prob, **params):
        shape = img.shape
        image = img.copy()
        output = np.zeros(shape, np.uint8)
        if len(shape) == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):

                    rdn = random.random()
                    if rdn < prob:
                        noise = random.randint(0, 255)
                        output[i, j, :] = (np.random.rand(3) * 255).astype(np.uint8)
                    else:
                        output[i][j] = image[i][j]
        elif len(shape) == 4:
            for i in range(shape[1]):
                for j in range(shape[2]):
                    rdn = random.random()
                    if rdn < prob:
                        output[:, i, j, :] = (np.random.rand(3) * 255).astype(np.uint8)
                    else:
                        output[:, i, j, :] = image[:, i, j, :]
        return output

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        prob = random.uniform(self.prob[0], self.prob[1])
        return {"prob": prob}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("prob")