from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from torch.optim import Adam
from ..utils import register

import torch.nn.functional as F
from finetune.ddpo_pytorch.ddpo_pytorch.rewards import *

class TextVideoReward(nn.Module):
    """
    3-D CNN + 文本注意力的简单奖励网络
    video  : [B, T, C, H, W]  --> C = 14
    text   : [B, L, 4096]
    output : [B, 1]  (0 ~ 1 之间)
    """
    def __init__(self,
                 embed_dim: int = 512,
                 use_film: bool = False,
                 video_channels: int = 13        # ← 改成 14，与你的数据一致
                 ):
        super().__init__()

        # ---------- 3-D CNN -------------------------------------------------
        self.video_backbone = nn.Sequential(
            nn.Conv3d(video_channels, 64,
                      kernel_size=(3, 7, 7),
                      stride=(1, 2, 2),
                      padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),   # [B,64,1,1,1]
            nn.Flatten(),                      # [B,64]
            nn.Linear(64, embed_dim)           # [B,embed_dim]
        )

        # ---------- 文本投影 -------------------------------------------------
        self.text_proj = nn.Linear(4096, embed_dim)

        # ---------- 融合策略 -------------------------------------------------
        self.use_film = use_film
        if self.use_film:
            self.film = nn.Linear(embed_dim, embed_dim * 2)
        else:
            # MultiheadAttention 的 key/value 与 query 长度允许不同
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=8)

        # ---------- 输出 MLP -------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    # ----------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
        # 把输出层偏置初始化成 1，方便前期训练
        nn.init.constant_(self.fc[-1].bias, 1.0)

    # ----------------------------------------------------------------------
    def forward(self,
                video: torch.Tensor,        # [B, T, C, H, W]
                text_latent: torch.Tensor   # [B, L, 4096]
                ) -> torch.Tensor:

        B, T, C, H, W = video.shape     # C = 13
        # print(B, T, C, H, W)

        # ---- 1. 提取视频特征 ---------------------------------------------
        v = video.permute(0, 2, 1, 3, 4)           # [B, C, T, H, W]
        v_lat = self.video_backbone(v)             # [B, embed_dim]

        # ---- 2. 文本-视频融合 --------------------------------------------
        if self.use_film:
            gamma, beta = self.film(v_lat).chunk(2, dim=-1)  # [B, embed_dim] × 2
            t_lat = self.text_proj(text_latent)               # [B, L, embed_dim]
            t_lat = gamma.unsqueeze(1) * t_lat + beta.unsqueeze(1)
            pooled = t_lat.mean(dim=1)                        # [B, embed_dim]
        else:
            # MultiheadAttention 期望:  q, k, v -> [S, B, E]
            q = self.text_proj(text_latent).transpose(0, 1)   # [L, B, E]
            kv = v_lat.unsqueeze(0)                           # [1, B, E]
            attn_out, _ = self.attn(q, kv, kv)                # [L, B, E]
            pooled = attn_out.mean(dim=0)                     # [B, E]

        # ---- 3. 打分 -----------------------------------------------------
        score = self.sigmoid(self.fc(pooled))                 # [B, 1]
        return score
    
from diffusers.utils import export_to_video
class CogVideoXT2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", cache_dir="/home/zihao/.hf_cache")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder", cache_dir="/home/zihao/.hf_cache")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer", cache_dir="/home/zihao/.hf_cache")

        components.vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-5b", subfolder="vae", cache_dir="/home/zihao/.hf_cache")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler", cache_dir="/home/zihao/.hf_cache")
        
        # self.pipe = CogVideoXPipeline.from_pretrained(
        #     model_path,
        #     text_encoder=components.text_encoder,
        #     transformer=components.transformer,
        #     vae=components.vae,
        #     torch_dtype=torch.float16,
        # ).to("cuda")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipeline:
        pipe = CogVideoXPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "prompt": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            ret["prompt"].append(sample["prompt"])

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample
        # timesteps = torch.randint(
        #     0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        # )
        # timesteps = timesteps.long()
        num_inference_steps = 10  # 你想跑多少步都行，越多越逼真但越慢
        self.components.scheduler.set_timesteps(num_inference_steps)
        t_start = 999
        timesteps = torch.tensor(t_start).repeat(batch_size).to(self.accelerator.device)
        
        if not hasattr(self, "reward_model"):
            self.reward_model = ViClipReward()
            # self.reward_model = VideoScoreTensorReward(device=self.accelerator.device) # .to(self.accelerator.device, dtype=torch.bfloat16)  # Use the default VideoScoreTensorReward
            
            self.reward_model_3d = TextVideoReward().to(self.accelerator.device, dtype=torch.bfloat16)

        # Add noise to latent
        # latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
        # noise = torch.randn_like(latent)
        # latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)
        

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )
        
        shape = (
            batch_size, num_frames, num_channels, height, width
        )
        latent_noisy = torch.randn(shape, device=self.accelerator.device, dtype=prompt_embedding.dtype) * self.components.scheduler.init_noise_sigma
        print(latent_noisy.shape)
        
        if not hasattr(self, "global_step"):  # count epochs
            self.components.vae = self.components.vae.to(self.accelerator.device, dtype=torch.bfloat16)
            self.global_step = 0
        else:
            self.global_step += 1
            
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        
        timesteps = self.components.scheduler.timesteps
        
        latents = latent_noisy
        extra_step_kwargs = {"eta": 0.0}
        old_pred_original_sample = None
        # do_cfg = True
        # prompt_in = prompt_embedding
        # guidance_scale = 1
        # do_classifier_free_guidance = True
        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        # print(type(self.components.scheduler))
        # exit(0)
        # 直接调用pipe输出一个全黑的视频
        # with torch.autocast(device_type="cuda"):
        #     self.pipe.enable_sequential_cpu_offload()
        #     video = self.pipe(prompt=batch["prompt"], guidance_scale=1, use_dynamic_cfg=False, num_inference_steps=10).frames[0]
        #     export_to_video(video, f"t2v.mp4", fps=8)  # 排除是否是模型不同
        
        with torch.autocast(device_type="cuda"):
            for i, t in enumerate(timesteps):
                # print(t)
                self._current_timestep = t
                t_b = t.repeat(batch_size).to(self.accelerator.device)
                latents_input = self.components.scheduler.scale_model_input(latents, t)
                noise_pred = self.components.transformer(
                    hidden_states        = latents_input,
                    encoder_hidden_states= prompt_embedding,
                    timestep             = t_b,
                    ofs                  = ofs_emb,
                    image_rotary_emb     = rotary_emb,
                    return_dict          = False,
                )[0].float()
                latents, old_pred_original_sample = self.components.scheduler.step(
                    noise_pred, old_pred_original_sample, t, timesteps[i-1] if i>0 else None,
                    latents, **extra_step_kwargs, return_dict=False
                )

                latents = latents.to(prompt_embedding.dtype)
                
                
        latent_pred = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W] -> [B, C, F, H, W]
        # 确保形状匹配
        latent_pred = latent_pred.contiguous()

        # 确保 latent_pred 和 vae 的权重在同一设备和数据类型
        # latent_pred = latent_pred.to(self.components.vae.device, dtype=self.components.vae.dtype)
        
        # OOM
        print(latent_pred.shape)
        with torch.no_grad():
            decoded_frame = self.components.vae.decode(latent_pred / self.components.vae.config.scaling_factor).sample  # [B, C, F, H, W]
            print(decoded_frame.shape)
            exit()
            decoded_frame = decoded_frame.float()
            reward = self.reward_model(decoded_frame, batch["prompt"], step=self.global_step)

        # # 检查 decoded_frame 是否包含 NaN 值
        # if torch.any(torch.isnan(decoded_frame)):
        #     print("Warning: decoded_frame contains NaN values")
            
        # if not hasattr(self, "reward_model_3d_optimizer"):
        #     self.reward_model_3d_optimizer = Adam(self.reward_model_3d.parameters(), lr=1e-4)

        # # 修复警告，使用 clone().detach()
        reward = reward.clone().detach().to(self.accelerator.device, dtype=torch.bfloat16)
        reward = reward.unsqueeze(1)
        
        # reward_predicted = self.reward_model_3d(latent_pred.detach(), prompt_embedding)
        
        # reward_loss = F.mse_loss(reward, reward_predicted)
        reward_loss = F.l1_loss(reward, reward_predicted)
        self.reward_model_3d_optimizer.zero_grad()
        # 注意这里要保留计算图，否则等会儿要继续用 latent_pred 做 PPO 时会报错
        # reward_loss.backward(retain_graph=True)
        reward_loss.backward()
        # print(compress_reward_reshaped, predicted_reward_3d)
        print(reward_loss)
        reward_loss = 0
        self.reward_model_3d_optimizer.step()
        reward_new = self.reward_model_3d(latent_pred, prompt_embedding).squeeze(1)
        reward_predicted = reward_new
        print("reward_loss, last, new: ", reward_loss, reward_predicted, reward, reward_new)
        
        loss = -reward_new.mean()
        print(reward_new.mean())
        return loss, reward_loss, reward, reward_new

        # Predict noise
        # predicted_noise = self.components.transformer(
        #     hidden_states=latent_added_noise,
        #     encoder_hidden_states=prompt_embedding,
        #     timestep=timesteps,
        #     image_rotary_emb=rotary_emb,
        #     return_dict=False,
        # )[0]

        # # Denoise
        # latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_added_noise, timesteps)

        # alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        # weights = 1 / (1 - alphas_cumprod)
        # while len(weights.shape) < len(latent_pred.shape):
        #     weights = weights.unsqueeze(-1)

        # loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        # loss = loss.mean()

        # return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-t2v", "lora", CogVideoXT2VLoraTrainer)
