from imagecorruptions import corrupt
import random
import cv2
import numpy as np

from PIL import Image,ImageDraw
corruption_names= ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur',
 'snow','frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression','speckle_noise', 'gaussian_blur', 'spatter', 'saturate','rain']
special_corruptions=['elastic_transform','pixelate','jpeg_compression','spatter'] #  severity 3~5
robust_corruptions=['defocus_blur','motion_blur','pixelate'] # 'jpeg_compression','draw'
class ImageCorrupter(object):
    def __init__(self,prob=0.5,n=1):
        self.prob = prob
        self.n = n
    def __call__(self,image):
        if random.random() < self.prob:
            corruptions = []
            while len(corruptions)<self.n:
                i = random.randint(0, len(robust_corruptions)-1)
                if robust_corruptions[i] not in corruptions:
                    corruptions.append(robust_corruptions[i])
            for c in corruptions:
                if c.endswith('blur'):
                    severity = random.randint(1,1)
                else :
                    severity = random.randint(1,3)
                image = corrupt(image,severity=severity,corruption_name=c)
        return image





# bg_img =Image.open('./Morning.png').convert('RGBA')
# bg_img = Image.new("RGBA", (256, 256), (0, 0, 255, 255))
def drawLineAll(img):
    startx= 0
    H,W,_=img.shape
    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = Image.fromarray(img_rgba)
    while startx<W:
        width = random.randint(0,1)
        img=drawRect(img, (startx,0,startx+width,H), fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255),random.randint(10,30)))#random.randint(0,150)
        startx +=width+3
    return np.array(img)
def drawRect(img, pos, **kwargs):
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")
    draw.rectangle(pos, **kwargs)
    img.paste(Image.alpha_composite(img, transp))
    return img
def drawSquare(img):
    H,W,_=img.shape
    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = Image.fromarray(img_rgba)
    num = random.randint(10,50)
    for _ in range(num):
        width = random.randint(20,60)
        startx,starty= random.randint(0,W-width),random.randint(0,H-width)
        img=drawRect(img, (startx,starty,startx+width,starty+width), fill=(random.randint(0,255),random.randint(0,255),random.randint(0,255),random.randint(10,30)))#random.randint(0,150)
    return np.array(img)
def drawWhiteSquare(img):
    H,W,_=img.shape
    img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = Image.fromarray(img_rgba)
    num = random.randint(7,25)
    for _ in range(num):
        width = random.randint(20,60)
        startx,starty= random.randint(0,W-width),random.randint(0,H-width)
        img=drawRect(img, (startx,starty,startx+width,starty+width), fill=(255,255,255,random.randint(10,30)))#random.randint(0,150)
    return np.array(img)

def draw(img):
    img = drawLineAll(img)
    if random.random() < 0.5:
        img = drawSquare(img)
        # img = drawWhiteSquare(img)
    return img