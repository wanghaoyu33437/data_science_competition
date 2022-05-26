python gen_perturbation.py --attack PGD --GPU 0 --batch_size 32
python gen_perturbation.py --attack DeepFool --GPU 0 --batch_size 32
python gen_perturbation.py --attack Square --GPU 0 --batch_size 32

python gen_advpatch.py --GPU 0 --batch_size 32