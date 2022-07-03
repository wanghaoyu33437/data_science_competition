import numpy as np
from category_id_map import lv2id_to_category_id

r1 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_7_1_MyBert_advTrain/result_pred.npy',allow_pickle=True)
r2 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_10_1_MyBert_advTrainFGM/0/result_pred.npy',allow_pickle=True)
r3 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_10_1_MyBert_advTrainFGM/1/result_pred.npy',allow_pickle=True)
r4 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_10_1_MyBert_advTrainFGM/2/result_pred.npy',allow_pickle=True)
r5 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_10_1_MyBert_advTrainFGM/3/result_pred.npy',allow_pickle=True)
r6 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_10_1_MyBert_advTrainFGM/4/result_pred.npy',allow_pickle=True)
r7 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_9_1_MyBert_advTrainFGM/0/result_pred.npy',allow_pickle=True)
r8 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_9_1_MyBert_advTrainFGM/1/result_pred.npy',allow_pickle=True)
r9 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_9_1_MyBert_advTrainFGM/2/result_pred.npy',allow_pickle=True)
r10 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_9_1_MyBert_advTrainFGM/3/result_pred.npy',allow_pickle=True)
r11 = np.load('/home/dell/HARD-DATA/WHY/07_Wechat_challenge/model_data/v1/Jun_9_1_MyBert_advTrainFGM/4/result_pred.npy',allow_pickle=True)
with open('model_data/v1/Jun_10_1_MyBert_advTrainFGM/result.csv', 'w') as f:
    for (ann_id,r11),(_,r22),(_,r33),(_,r44) ,(_,r55),(_,r66),(_,r77),(_,r88),(_,r99),(_,r100),(_,r111) in zip(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11):
        # print(r11.argmax(),r22.argmax(),r33.argmax(),(r11+r22+r33).argmax())
        pred_label_id =(r11+r22+r33++r44+r55+r66+r77+r88+r99+r100+r111).argmax()
        category_id = lv2id_to_category_id(pred_label_id)
        f.write(f'{ann_id},{category_id}\n')