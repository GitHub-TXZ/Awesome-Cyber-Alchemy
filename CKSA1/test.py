import os

import SimpleITK as sitk
import pandas as pd


path = r"C:\Users\yimoo\Desktop\ASPECTS_KSSnew(1).xlsx"

def process(file_path, idx, left, right):
    #D:\Data\KSR Dataset\trainACUTE\Kmd001
    file_path = os.path.join(file_path, f"{idx}")
    file_path = os.path.join(file_path, "ASPECTS_Region.nii.gz")
    if not os.path.exists(file_path):
        return
    print(file_path)

    M = [9,10,11,7,8,12,14,16,13,15]
    ta = ["M1","M2","M3","M4","M5","M6","L","I","C","IC"]
    a = "_right"
    if left == 1:
        a = "_left"
        for i in range(len(M)):
            M[i] += 10
    else:
        pass

    itk = sitk.ReadImage(file_path)
    arr = sitk.GetArrayFromImage(itk)

    for i in range(10):
        M1_arr = arr.copy()
        M1_arr[M1_arr == M[i]] = 1
        M1_arr[M1_arr != M[i]] = 0
        M1_itk = sitk.GetImageFromArray(M1_arr)
        M1_itk.SetOrigin(itk.GetOrigin())
        M1_itk.SetDirection(itk.GetDirection())
        M1_itk.SetSpacing(itk.GetSpacing())

        #保存图像
        save_path = file_path.replace("\ASPECTS_Region.nii.gz", "")
        save_path = save_path.replace("KSR Dataset", "KSR")
        os.makedirs(save_path, exist_ok=True)
        sitk.WriteImage(M1_itk, file_path.replace('Region', ta[i] + a).replace("KSR Dataset", "KSR"))






def readexcel1(path):
    raw_data = pd.read_excel(path, header=0)
    data = raw_data.values
    return data

def accse():
    info = readexcel1(path)
    info
    print(info[:,0])


def preacess(data):
    pass







if __name__ == '__main__':
    accse()