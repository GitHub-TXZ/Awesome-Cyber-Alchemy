410 Cyber Segmentation Laboratory

# 如何贡献？
Two_d 和 Three_d 分别是2d和3d的模型or模块，对应（B,C,H,W） or (B,C,D,H,W),后期考虑加上（B,N,C）的1D模型or模块。
## 1.若只有一个.py,则在改文件夹中写好输入输出测试即可，若有多个.py, 则在Three_d 或者 Two_d 下新建一个文件夹放所有文件，将测试写到 Three_d/model_test 下面. 
## 2.在py文件里注明给出模块相应的链接 or 论文.


___
___
___
___
***
***
***

# CBCT-King-Queen
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

## 第四，自己划分数据集，训练和验证


