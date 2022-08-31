from distutils import filelist
import os

filelist_f = open("filelists/lanzhu_train_filelist.txt", mode='w', encoding="utf-8")
opath = "F:\\shixi\\CS\\spleeter\\lanzhu\\1"
for root, dirs, files in os.walk(opath):
    for file in files:
        if(file[-3:] != "wav"):
            continue
        fp = os.path.join(root, file)
        filelist_f.write(fp + '|' + fp[:-3] + "npy\n")
        print(fp + '|' + fp[:-3] + "npy\n")
