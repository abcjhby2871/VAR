import sys
sys.path.append('.')
from models import build_vae_var
from models.helpers import sample_with_top_k_top_p_
from PIL import Image
import numpy as np 
import os
import torch 
import argparse
import os.path as osp
import torch
import numpy as np
import json
import torch.nn.functional as F
from typing import Optional,Union
from torchvision import transforms
from PIL import Image
import random

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cuda:1')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--sample_num',type=int,default=32)
parser.add_argument('MODEL_DEPTH',type=int)
args=parser.parse_args()
MODEL_DEPTH=args.MODEL_DEPTH
device=args.device
batch_size=args.batch_size
sample_num=args.sample_num



@torch.no_grad()
def scaling_infer(
    self,imgs:torch.Tensor,label_B: Optional[Union[int, torch.LongTensor]], #imgs:Bximg
    g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
   
    B = imgs.shape[0]
    ground_token_maps=self.vae_proxy[0].img_to_idxBl(imgs,self.patch_nums) #真实的词元图序列
    Err = []
    L = []
    if g_seed is None: rng = None
    else: self.rng.manual_seed(g_seed); rng = self.rng
    label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device) # 注释：类别要与图片真实的类别相同
    sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
    lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
    next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
    cur_L = 0
    f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
    for b in self.blocks: b.attn.kv_caching(True)
    for si, pn in enumerate(self.patch_nums):   # si: i-th segment
        ratio = si / self.num_stages_minus_1
        cur_L += pn*pn
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map
        for b in self.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = self.get_logits(x, cond_BD)
        t = cfg * ratio
        logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:] # 注释，无分类器引导
        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
        Err.append(torch.mean((idx_Bl!=ground_token_maps[si]).float()).item()) # 注释，计算损失与错误率

        L.append(F.cross_entropy(logits_BlV.view(-1,logits_BlV.shape[-1]),ground_token_maps[si].view(-1)).item())
        idx_Bl = ground_token_maps[si] # 注释，将预测的token换成真实的token
        h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
        h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
        if si != self.num_stages_minus_1:   # prepare for next stage
            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
            next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
    for b in self.blocks: b.attn.kv_caching(False)
    result = {"Err_all":np.mean(Err),"Err_last":Err[-1],"L_all":np.mean(L),"L_last":L[-1]}
    return result


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
    hw=512
else:
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    hw=256
    
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

results={"Err_last":[],"Err_all":[],"L_last":[],"L_all":[]}
folder='data/imagenet/ILSVRC/Data/CLS-LOC/train'
img_processor=transforms.Compose([transforms.Resize(hw),transforms.RandomCrop(hw),transforms.ToTensor()])
with open('data/imagenet/LOC_synset_mapping.txt') as f:
    for idx,line in enumerate(f.readlines()):
        name=line.split()[0]
        img_paths=[p for p in os.listdir(f'{folder}/{name}') if p.endswith('JPEG')]
        random.shuffle(img_paths)
        p=0
        temp={"Err_last":[],"Err_all":[],"L_last":[],"L_all":[]}
        samples=min(len(img_paths),sample_num)
        while p<samples:
            num=min(samples-p,batch_size)
            imgs=[]
            for i in range(num):
                imgs.append(img_processor(Image.open(f'{folder}/{name}/{img_paths[i+p]}').convert("RGB")))
            imgs=torch.stack(imgs)
            imgs=imgs.to(device)
            result=scaling_infer(var,imgs,idx,cfg=1.5)
            for key,value in result.items():
                temp[key].append(value)
            p+=batch_size
        for key,value in temp.items():
            results[key].append(np.mean(value))
        if (idx+1)%10==0:
            for key,value in results.items():
                print(f'{key}:{np.mean(value)}')
            print(f'{(idx+1)//10}% complete')
print("complete!")
print(results)
if os.path.exists('log/scale.json'):
    with open('log/scale.json','r') as f:
        T=json.load(f)
else:
    T={}
for key,value in results.items():
    results[key]=np.mean(value)
T[MODEL_DEPTH]=results
with open('log/scale.json', 'w') as f:
    json.dump(T, f)




