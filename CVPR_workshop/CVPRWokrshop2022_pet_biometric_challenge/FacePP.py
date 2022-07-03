
import requests
from json import JSONDecoder
from tqdm import tqdm
def compareIm(faceId1, faceId2):
    # 传送两个本地图片地址 例如："D:/Downloads/wt.jpg"
    try:
        # 官方给你的接口地址
        compare_url = "https://api-cn.faceplusplus.com/imagepp/v2/dognosecompare"
        # 创建应用分配的key和secret
        key = "xjsGAwjTuc2sdSSWij3qI9D5oDZ1IEhh"
        secret = "MWoP8LEdCo8pTxR-99kzq4Fh6VKSyQ9W"
        # 创建请求数据
        data = {"api_key": key, "api_secret": secret}
        files = {"image_file": open(faceId1, "rb"), "image_ref_file": open(faceId2, "rb")}
        # 通过接口发送请求
        response = requests.post(compare_url, data=data, files=files)
        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)

        confindence = req_dict['confidence']
        return confindence
    except Exception:
        return 0

import pandas as pd
val_csv = pd.read_csv('dataset/pet_biometric_challenge_2022/validation/new_valid_data.csv')
result  = pd.read_csv('dataset/pet_biometric_challenge_2022/validation/valid_data.csv')
path = 'dataset/pet_biometric_challenge_2022/validation/images/'

for i in tqdm(range(len(val_csv))):
    imageA = val_csv['imageA'][i]
    imageB = val_csv['imageB'][i]
    confindence = compareIm(path+imageA,path+imageB)
    result.loc[i,'prediction']= confindence
    print(confindence)
