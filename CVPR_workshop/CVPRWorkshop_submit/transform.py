import albumentations  as A
from albumentations.pytorch import ToTensorV2
import numpy as  np
import random
from torchvision import transforms

def gen_transform(inputsize = 256,resize = 224,mode='train'):
    if mode =='train':
        transform =A.Compose([
            A.Resize(resize, resize),
            A.SomeOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.GridDistortion(p=0.5),
                A.Transpose(p=0.5),
                A.CoarseDropout(30,15,15,p=0.5),
                A.ElasticTransform(0.5),
        ],n=2),
        A.SomeOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomContrast(p=0.5),
                A.Equalize(p=0.5),
                A.GaussNoise(var_limit=1000,p=1),
                SPNoise(p=0.5),
                # A.ISONoise((0.2,1),p=1),
                # A.MultiplicativeNoise(p=1),
                A.RandomSnow(p=1),
                A.Downscale(p=1),
                A.RandomFog(p=1),
                A.RandomRain(p=1),
                A.GlassBlur(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
                A.JpegCompression(39, 40, p=1),
            ],p=1,n=2),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    elif mode =='pretrain':
        transform =A.Compose([
            A.Resize(resize, resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90,p=0.5),
            A.SomeOf([
                A.GridDistortion(p=0.5),
                A.Transpose(p=0.5),
                A.CoarseDropout(p=0.5),
                A.ElasticTransform(0.5),
            ],n=2,p=1),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomContrast(p=0.5),
            ],p=0.5),
            A.SomeOf([
                    A.GaussNoise(var_limit=1000,p=1),
                    A.ISONoise(p=1),
                    A.MultiplicativeNoise(p=1),
                    A.RandomSnow(p=1),
                    A.RandomFog(p=1),
                    A.RandomRain(p=1),
                    A.GlassBlur(p=1),
                    A.GaussianBlur(p=1),
                    A.MotionBlur(p=1),
                    A.JpegCompression(19, 20, p=1),
            ],n=3,p=1),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    elif mode == 'adv':
        transform = A.Compose([
            A.Resize(resize, resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30,p=0.5),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.Transpose(p=0.5),
                A.CoarseDropout(p=0.5),
                A.ElasticTransform(0.5),
            ],p=1),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RandomContrast(p=0.5),
            ],p=0.3),
            A.OneOf([
                    A.GaussNoise(var_limit=500),
                    A.ISONoise(),
                    A.MultiplicativeNoise(),
                    A.RandomSnow(),
                    A.RandomFog(),
            ],p=1),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    elif (mode == 'mix' or mode=='smi'):
        transform = A.Compose([
            # A.Resize(resize, resize),
            # A.Resize(256,256,interpolation=2),
            # A.ToFloat(),
            # A.CenterCrop(resize,resize),
            A.Flip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.1),
            # A.OneOf([
            #     A.MotionBlur(p=0.1),
            #     A.GaussianBlur(p=0.1),
            # ],p=0.2),
            # A.OneOf([
            #     A.ToGray(p=0.1),
            #     A.HueSaturationValue(p=0.1),
            #     A.RandomBrightnessContrast(p=0.1)
            # ], p=0.2),
            # A.Normalize((0., 0., 0.), (1, 1, 1)),
            # A.Normalize(),
            # A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
        # transform = transforms.Compose([
        #     # A.Resize(resize, resize),
        #     transforms.ToPILImage(),
        #     # transforms.Resize((resize, resize)),
        #     transforms.CenterCrop(resize),
        #     transforms.RandomHorizontalFlip(p=0.2),
        #     transforms.RandomVerticalFlip(p=0.2),
        #     transforms.RandomGrayscale(p=0.1),
        #     transforms.RandomRotation(20),
        #         # A.GridDisRatortion(p=0.5),
        #     transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])
    else:
        # transform = A.Compose([
        #     A.Resize(resize, resize),
        #     A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ToTensorV2(),
        # ])
        transform = transforms.Compose([
            # A.Resize(resize, resize),
            transforms.ToPILImage(),
            # transforms.Resize((256, 256),F.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
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