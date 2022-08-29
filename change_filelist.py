fi = open("filelists/shioriko_train_filelist.txt", mode='r', encoding="utf-8")
fo = open("filelists/shioriko_train_filelist_c.txt", mode='w', encoding="utf-8")
for line in fi.readlines():
    line_part1 = line.split('|')[0].split('.')[0]
    fo.write(line_part1 + ".wav|" + line_part1 + ".npy\n")
