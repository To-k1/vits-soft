from warnings import catch_warnings
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import fft
from scipy.io import wavfile
import os
import torch, torchaudio



def create_fft(in_path, out_path):
    #sample_rate采样率；X为音乐文件本身
    sample_rate, X = wavfile.read(in_path)
    #对音乐文件本身进行fft快速傅里叶变化，取前256赫兹数据，进行取绝对值，得到fft_features傅里叶变换的特征
    fft_features = abs(fft(X)[:256])
    #存储特征，存储的是.fft格式，但是最终生成的是.fft.npy格式，这是numpy自动生成的
    np.save(out_path, fft_features)

# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()


op_path = "F:/shixi/CS/vits3/vits/wav/"
for root, dirs, files in os.walk(op_path):
    for file in files:
        if(file[-3:] == "wav"):
            in_path = os.path.join(root, file)
            out_path = in_path[:-3] + "npy"
            # create_fft(in_path, out_path)
            # Load audio, 2nd para is sample_rate
            wav, sr = torchaudio.load(in_path)
            # assert sr == 16000
            wav = wav.unsqueeze(0).cuda()

            # Extract speech units
            with torch.inference_mode():
                units = hubert.units(wav)
            np.save(out_path, units.squeeze().cpu().numpy())
            print("Created: " + out_path)



    
