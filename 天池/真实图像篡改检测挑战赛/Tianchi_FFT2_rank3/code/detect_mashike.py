#!/usr/bin/env python3
# # -*- coding: utf-8 -*-


#モザイクの箇所を検出し、白塗りする。
#以下参考資料。
#  ・Template Matching
#    http://docs.opencv.org/3.2.0/d4/dc6/tutorial_py_template_matching.html
#    http://opencv.jp/cookbook/opencv_img.html#id32


import cv2
import numpy as np
import os
from tqdm import tqdm
# test_path = '../../baseline/data/test_decopose_512/'
test_path = '../../baseline/data/test/img/'
test_paths = sorted(os.listdir(test_path))
templates=[]
for i in range(11, 20 + 1):
    pattern_filename = "../user_data/pattern/pattern" + str(i) + "x" + str(i) + ".png"
    templates.append( cv2.imread(pattern_filename, 0))
for path in tqdm(test_paths):
    img_rgb = cv2.imread(test_path+path)
    img = img_rgb.copy()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) #グレースケールに
    img_gray = cv2.Canny(img_gray,10,20) #エッジ検出

    cv2.imwrite('../user_data/canny/'+path, img_gray)
    img_gray = 255-img_gray #白黒反転
    img_gray = cv2.GaussianBlur(img_gray,(3,3),0) #少しぼかす
    mask = np.zeros_like(img_rgb)
    # print(path)
    for template in templates:
        w, h = template.shape[::-1]
        img_kensyutu_kekka = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.4
        loc = np.where(img_kensyutu_kekka >= threshold)
        for pt in zip(*loc[::-1]):
            #cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (255,255,255), 1)
            cv2.rectangle(mask, pt, (pt[0] + w, pt[1] + h), (255,255,255), -1)
        # cv2.imwrite('output_progress_'+str(i)+'.png', img_rgb)
    if (mask[:,:,0]==255).sum()>= 3000:
       cv2.imwrite('../save_out/msk/'+path[:-4]+'.png', mask)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#モザイクのサイズが11から20までだった場合のためのパターン画像ファイルを生成。
#
#以下参考資料。
#  http://qiita.com/suto3/items/5181b4a3b9ebc206f579

# from PIL import Image
#
# def make_image(masksize, filename):
#     picturesize = 2+masksize+masksize-1+2
#     screen = (picturesize, picturesize)
#
#     img = Image.new('RGB', screen, (0xff,0xff,0xff))
#
#     pix = img.load()
#
#     for i in range(2,picturesize,masksize-1):
#         for j in range(2,picturesize,masksize-1):
#             for k in range(0,picturesize):
#                 pix[i, k] = (0,0,0)
#                 pix[k, j] = (0,0,0)
#
#     img.save(filename)
#     return
#
# for i in range(5, 20+1):
#     make_image(i, "../user_data/pattern/pattern"+str(i)+"x"+str(i)+".png")
