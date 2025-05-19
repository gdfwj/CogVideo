"""
train_t2v_latent_reward.py
--------------------------
只使用 t2v_latents_x.pt → reward 训练 TextVideoReward
"""

import os, json, math
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
    CogVideoXPipeline,
)

# ================== ⚙️ 路径与超参 ==================
JSON_FILE      = "/home/zihao/CogVideo/finetune/data.json"
PROMPT_TXT     = "/data/zihao/one_data/prompt.txt"      # 存那句长 prompt 的纯文本
MODEL_ID       = "THUDM/cogvideox-2b"               # text encoder（任意同构 T5-large）
SAVE_DIR       = "ckpt_reward"
DEVICE         = "cuda:0"
FP16           = True
BATCH_SIZE     = 8
EPOCHS         = 100
LR             = 2e-4
FIXED_FRAMES   = 16          # 抽/补到固定 T
VIDEO_CHANNELS = 13           # latent 的通道数
os.makedirs(SAVE_DIR, exist_ok=True)
# ===================================================


# ---------- 1. 读 JSON & 过滤 t2v ----------
with open(JSON_FILE, encoding="utf-8") as f:
    full_map = json.load(f)

t2v_items = {k: v for k, v in full_map.items()
             if "/t2v_latents_" in k or "t2v_latents_" in k}

assert len(t2v_items) > 0, "⚠️ 没找到任何 t2v_latents_x.pt 样本！检查 JSON 路径"

model_id = "THUDM/cogvideox-2b"
prompt = (
    "A man rides a horse along a dusty trail, surrounded by a vast desert landscape. "
    "The sun sets on the horizon, casting a warm golden glow across the sky, with scattered "
    "clouds adding texture to the scene. Tall cacti stand like sentinels on either side of "
    "the path, their silhouettes stark against the fading light. As the horse trots steadily "
    "forward, the rider takes in the serene beauty of the open wilderness, the gentle breeze "
    "rustling through the sparse vegetation. Birds can be seen flying in the distance, adding "
    "a sense of tranquility to the moment."
)
# image_path = "/data/zihao/one_data/images/image.jpg"  # 参考图
text_candidates = [prompt]                            # ViCLIP 查询文本

# ========== 1. 载入视频生成模型 （只做一次） ==========
print("Loading CogVideoX2b …")
transformer = CogVideoXTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.float16, cache_dir="/home/zihao/.hf_cache"
).to(DEVICE)

text_encoder = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=torch.float16, cache_dir="/home/zihao/.hf_cache"
).to(DEVICE)

vae = AutoencoderKLCogVideoX.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float16, cache_dir="/home/zihao/.hf_cache"
).to(DEVICE)

pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.float16,
).to(DEVICE)

pipe.enable_sequential_cpu_offload()
   # [1,L,4096]
TEXT_LAT = pipe(
        prompt=prompt,
        guidance_scale=1.0,
        use_dynamic_cfg=False,
        num_inference_steps=1,
    )[2].to(DEVICE)
TEXT_LAT = TEXT_LAT.squeeze(0)                                    # [L,4096]


# ---------- 3. 数据集 ----------
class T2VLatentDataset(Dataset):
    def __init__(self, mapping: dict, fixed_frames: int):
        # mapping: {path: reward}
        self.items = list(mapping.items())
        self.T = fixed_frames

    def __len__(self): return len(self.items)

    def _fix_T(self, lat):
        print(lat.shape)
        T, C, H, W = lat.shape
        if T >= self.T:
            idx = torch.linspace(0, T-1, self.T).round().long()
            lat = lat[idx]
        else:
            rep = math.ceil(self.T / T)
            lat = torch.cat([lat]*rep, dim=0)[:self.T]
        return lat

    def __getitem__(self, idx):
        path, reward = self.items[idx]
        lat = torch.load(path, map_location="cpu").float()   # [T,4,h,w]
        # lat = self._fix_T(lat)
        return lat.squeeze(0), torch.tensor(float(reward), dtype=torch.float32)

def collate(batch):
    vids, rews = zip(*batch)
    return torch.stack(vids), torch.tensor(rews).unsqueeze(1)

dataset = T2VLatentDataset(t2v_items, FIXED_FRAMES)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, pin_memory=True, collate_fn=collate)


# ---------- 4. 奖励网络 ----------
class TextVideoReward(nn.Module):
    def __init__(self, embed_dim=512, video_channels=VIDEO_CHANNELS):
        super().__init__()
        self.video_backbone = nn.Sequential(
            nn.Conv3d(video_channels, 64, kernel_size=(3,7,7),
                      stride=(1,2,2), padding=(1,3,3)),
            nn.ReLU(True),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        self.text_proj = nn.Linear(4096, embed_dim)
        self.attn      = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.mlp       = nn.Sequential(
            nn.Linear(embed_dim,128), nn.ReLU(True), nn.Linear(128,1)
        )
        self.sig = nn.Sigmoid()
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m,(nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0.)
        nn.init.constant_(self.mlp[-1].bias, 1.0)

    def forward(self, video, text_lat):     # video:[B,T,C,H,W]
        v = video.permute(0,2,1,3,4)        # [B,C,T,H,W]
        v_lat = self.video_backbone(v)      # [B,E]
        q = self.text_proj(text_lat).transpose(0,1)   # [L,B,E]
        kv = v_lat.unsqueeze(0)
        attn,_ = self.attn(q, kv, kv)
        pooled = attn.mean(0)               # [B,E]
        return self.sig(self.mlp(pooled))   # [B,1]

model = TextVideoReward().to(DEVICE)

opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
loss = nn.MSELoss()
if FP16: scaler = torch.cuda.amp.GradScaler()


# ---------- 5. 训练 ----------
step = 0
for epoch in range(1, EPOCHS+1):
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for vids, gts in pbar:
        vids = vids.to(DEVICE)           # [B,T,C,H',W']
        gts  = gts.to(DEVICE)            # [B,1]
        B    = vids.size(0)
        txt_lat = TEXT_LAT.unsqueeze(0).expand(B, -1, -1)  # [B,L,4096]

        with torch.cuda.amp.autocast(enabled=FP16):
            preds = model(vids, txt_lat)
            loss_val = loss(preds, gts)

        opt.zero_grad(set_to_none=True)
        if FP16:
            scaler.scale(loss_val).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss_val.backward(); opt.step()

        step += 1
        pbar.set_postfix({"loss": f"{loss_val.item():.4f}"})

    torch.save({"epoch":epoch,
                "model": model.state_dict(),
                "opt":   opt.state_dict()},
               os.path.join(SAVE_DIR, f"epoch{epoch:02d}.pt"))
    print(f"✓ saved epoch {epoch}")

print("🎉  Training finished (t2v only).")
