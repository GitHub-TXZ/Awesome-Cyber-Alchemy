# CBCT-King-Queen
410 Cyber Segmentation Laboratory


## 第一步，准备环境
In your server:
`scp -P 10522 -r tanxz@122.207.108.8:/home/tanxz/.conda/envs/nnunet /home/txz/anaconda3/envs/`
`cd /home/txz/anaconda3/envs/nnunet/bin`
`vim pip`, 将第一行的地址改为自己服务器的地址
`find /home/txz/anaconda3/envs/nnunet/bin/ -type f -exec sed -i '1s|#!/home/tanxz/.conda/envs/nnunet/bin/python|#!/home/txz/anaconda3/envs/nnunet/bin/python|' {} + ` 将所有shebang修改
clone code, then `pip install -e ."


## 第二步，设置路径, i.e.
`vim ./nnunetv2/paths.py`
`
nnUNet_raw = "/home/txz/Codes/CBCT-King-Queen/data/nnUNet_raw/"
nnUNet_preprocessed = "/home/txz/Codes/CBCT-King-Queen/data/nnUNet_preprocessed/"
nnUNet_results = "/home/txz/Codes/CBCT-King-Queen/data/nnUNet_results/"
`
## 第三步，预处理数据集，i.e.
`nnUNetv2_plan_and_preprocess -d 18 19 20 --verify_dataset_integrity`

## 第四部，自己划分数据集，训练和验证


