import numpy as np
import os
import torch, torchaudio



# Load checkpoint (either hubert_soft or hubert_discrete)
hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").cuda()


# op_path = "F:/shixi/CS/vits3/vits/wav/"
op_path = "F:/shixi/CS/spleeter/lanzhu/1"
for root, dirs, files in os.walk(op_path):
    for file in files:
        if(file[-3:] == "wav"):
            try:
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
            except RuntimeError:
                os.remove(in_path)



    
