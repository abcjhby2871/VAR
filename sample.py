import os
import sys
sys.path.append('.')
from models import build_vae_var
from PIL import Image
import numpy as np 
import torch 
import time
import argparse
import os.path as osp
import torch
import numpy as np
import json

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cuda:6')
parser.add_argument('--batch_size',type=int,default=25)
parser.add_argument('MODEL_DEPTH',type=int)
args=parser.parse_args()
MODEL_DEPTH=args.MODEL_DEPTH
device=args.device
batch_size=args.batch_size


setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed


assert MODEL_DEPTH in {16, 20, 24, 30, 36}


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
if not osp.exists(f'weight/{vae_ckpt}'): os.system(f'wget {hf_home}/{vae_ckpt} -O weight/{vae_ckpt}')
if not osp.exists(f'weight/{var_ckpt}'): os.system(f'wget {hf_home}/{var_ckpt} -O weight/{var_ckpt}')

# build vae, var
FOR_512_px = MODEL_DEPTH == 36
if FOR_512_px:
    patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
else:
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

vae, var = build_vae_var(
    V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
    device=device, patch_nums=patch_nums,
    num_classes=1000, depth=MODEL_DEPTH, shared_aln=FOR_512_px,
)

# load checkpoints
vae.load_state_dict(torch.load(f'weight/{vae_ckpt}', map_location='cpu'), strict=True)
var.load_state_dict(torch.load(f'weight/{var_ckpt}', map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'preparation finished.')
print('begin sample!')
os.makedirs(f"result/d{MODEL_DEPTH}",exist_ok=True)
st=time.time()
for i in range(1000):
    p=0
    while p<50:
        num=min(50-p,batch_size)
        imgs=var.autoregressive_infer_cfg(num,i, cfg=1.5, top_p=0.96, top_k=900, more_smooth=False).permute(0,2,3,1).mul(255).cpu().numpy().astype(np.uint8)
        for k in range(num):
            Image.fromarray(imgs[k]).save(f'result/d{MODEL_DEPTH}/{i}_{p+k}.png')
        p+=batch_size
    if (i+1)%100==0:
        print(f'{(i+1)//10}% complete, using {time.time()-st:.4f} s')

print(f'complete, using {time.time()-st:.4f} s')

if os.path.exists('log/time.json'):
    with open('log/time.json','r') as f:
        T=json.load(f)
else:
    T={}
T[MODEL_DEPTH]=time.time()-st
with open('log/time.json', 'w') as f:
    json.dump(T, f)



