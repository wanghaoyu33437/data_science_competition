import albumentations  as A
from albumentations.pytorch import ToTensorV2

def gen_transform(resize = 512,mode='train'):
    if mode =='train':
        transform =A.Compose([
            A.Resize(resize, resize),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                #     A.RandomRotate90(p=0.5),
                #     A.RandomBrightnessContrast(p=0.5),
                #     A.HueSaturationValue(p=0.5),
                #     A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                #     A.CoarseDropout(p=0.2),
                #     A.Transpose(p=0.5),
                A.RandomRotate90(p=0.5),
            ]),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    elif mode == 'val':
        transform = A.Compose([
            A.Resize(resize, resize),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(resize, resize),
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ToTensorV2(),
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




