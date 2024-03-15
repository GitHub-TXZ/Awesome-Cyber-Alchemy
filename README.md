# CBCT-King-Queen
410 Cyber Segmentation Laboratory


##第一步，准备环境
In your server:
`scp -P 10522 -r tanxz@122.207.108.8:/home/tanxz/.conda/envs/nnunet /home/txz/anaconda3/envs/`
`cd /home/txz/anaconda3/envs/nnunet/bin`
`vim pip`, 将第一行的地址改为自己服务器的地址

##第二步，设置路径, i.e.
`vim ./nnunetv2/paths.py`
`
nnUNet_raw = "/home/txz/Codes/CBCT-King-Queen/data/nnUNet_raw/"
nnUNet_preprocessed = "/home/txz/Codes/CBCT-King-Queen/data/nnUNet_preprocessed/"
nnUNet_results = "/home/txz/Codes/CBCT-King-Queen/data/nnUNet_results/"
`
