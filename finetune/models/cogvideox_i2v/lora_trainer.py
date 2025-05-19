from typing import Any, Dict, List, Tuple

from finetune.ddpo_pytorch.ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override
from concurrent import futures

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.ddpo_pytorch.ddpo_pytorch.rewards import *
from finetune.ddpo_pytorch.ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

from ..utils import register
from torch import nn
import wandb
import torch.nn.functional as F
from torch.optim import Adam

import torch

import torch
import torch.nn as nn
import math
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

class TextVideoReward(nn.Module):
    def __init__(self, embed_dim: int = 512,  #
                 num_timesteps: int = 1000,   # <= 训练时用的最大 diffusion T
                 use_film: bool = False       # True 时改用 FiLM 调制
                 ):
        super().__init__()

        # ---- 3D‑CNN 提取视频特征 ----------------------------------------
        self.video_backbone = nn.Sequential(
            nn.Conv3d(13, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # -> [B,64,1,1,1]
            nn.Flatten(),                     # -> [B,64]
            nn.Linear(64, embed_dim)          # -> [B,embed_dim]
        )

        # ---- 文本投影 ---------------------------------------------------
        self.text_proj = nn.Linear(4096, embed_dim)

        # ---- timestep 嵌入 ----------------------------------------------
        # 方式 A：直接 Embedding
        self.t_embed = nn.Embedding(num_timesteps, embed_dim)

        # 若想用正余弦再投影，可以把上面换成：
        # self.t_embed = nn.Sequential(
        #     SinCosPosEnc(dim=embed_dim//2),  # 自定义正余弦 PE
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(embed_dim, embed_dim),
        # )

        # ---- Cross‑Attention & FiLM ‑‑ 二选一 ----------------------------
        self.use_film = use_film
        if self.use_film:
            # FiLM：γ, β 由 t_embed 产生
            self.film = nn.Linear(embed_dim, embed_dim * 2)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=8)

        # ---- 输出 MLP ---------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    # ---------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
        nn.init.constant_(self.fc[-1].bias, 1.0)

    # ---------------------------------------------------------------------
    def forward(
        self,
        video: torch.Tensor,        # [B, T, C, H, W]
        text_latent: torch.Tensor,  # [B, L, 4096]
        # timesteps: torch.Tensor,    # [B]  int64  (当前 diffusion t)
    ) -> torch.Tensor:
        """
        Returns: reward ∈ (0,1), shape [B,1]
        """
        B, T, C, H, W = video.shape

        # ---- 1. 视频特征 ------------------------------------------------
        v = video.permute(0, 2, 1, 3, 4)       # -> [B, C, T, H, W]
        v_lat = self.video_backbone(v)          # -> [B, embed_dim]

        # ---- 2. timestep 嵌入 ------------------------------------------
        # t_emb = self.t_embed(timesteps)         # -> [B, embed_dim]

        if self.use_film:
            # FiLM：γ, β 调制 video_latent
            # gamma, beta = self.film(t_emb).chunk(2, dim=-1)
            v_lat = gamma * v_lat + beta
            kv = v_lat.unsqueeze(0)             # [1,B,embed_dim]
            q = self.text_proj(text_latent).transpose(0, 1)  # [L,B,embed_dim]
            # 普通注意力（不含t），可选
            attn_out, _ = self.attn(q, kv, kv)
        else:
            # 直接把 t_emb 加到 video_latent，再做 cross‑attention
            kv = v_lat.unsqueeze(0)  #  + t_emb).unsqueeze(0)   # [1,B,embed_dim]
            q = self.text_proj(text_latent).transpose(0, 1)  # [L,B,embed_dim]
            attn_out, _ = self.attn(q, kv, kv)

        # ---- 3. pooling & 打分 -----------------------------------------
        pooled = attn_out.mean(dim=0)           # [B, embed_dim]
        score = self.sigmoid(self.fc(pooled))   # [B,1]   (0~1)

        return score




class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXImageToVideoPipeline

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        components.transformer = CogVideoXTransformer3DModel.from_pretrained(model_path, subfolder="transformer")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        
        # pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        #     model_path,
        #     text_encoder=components.text_encoder,
        #     transformer=components.transformer,
        #     vae=components.vae,
        #     torch_dtype=torch.float16,
        # )
        
        # components.ddim_scheduler = pipe.scheduler
        
        # components.reward_model = ViClipReward()   # VideoScoreTensorReward(device=components.transformer) # .to(self.accelerator.device, dtype=torch.bfloat16)  # Use the default VideoScoreTensorReward
            
        # components.reward_model_3d = TextVideoReward().to(components.transformer, dtype=torch.bfloat16)
        
        # components.reward_model_3d_optimizer = Adam(components.reward_model_3d.parameters(), lr=1e-5)
        
        # components.pipeline = CogVideoXImageToVideoPipeline(
        #     tokenizer=components.tokenizer,
        #     text_encoder=components.text_encoder,
        #     vae=components.vae,
        #     transformer=unwrap_model(self.accelerator, components.transformer),
        #     scheduler=self.components.scheduler,
        # )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXImageToVideoPipeline:
        self.pipeline = CogVideoXImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return self.pipeline

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
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": [], "prompt": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            ret["prompt"].append(sample["prompt"])

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        # 还原原始 prompt 文本
        # prompt = self.components.tokenizer.batch_decode(batch["prompt_embedding"], skip_special_tokens=True)
        # print(batch["prompt"])
        # exit(0)
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        images = batch["images"]
        # import torch.nn.functional as F
        # print(F.mse_loss(images[0], images[1]))
        # print(F.mse_loss(latent[0], latent[1]))
        # print(F.mse_loss(prompt_embedding[0], prompt_embedding[1]))
        # exit(0)

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

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

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        image_latent_dist = self.components.vae.encode(noisy_images.to(dtype=self.components.vae.dtype)).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor
        
        num_inference_steps = 10  # 你想跑多少步都行，越多越逼真但越慢
        self.components.scheduler.set_timesteps(num_inference_steps)
        # Sample a random timestep for each sample
        # timesteps = torch.randint(
        #     0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        # )
        t_start = torch.randint(
            low=1,
            high=self.components.scheduler.config.num_train_timesteps,
            size=(1,),
            device=self.accelerator.device,
            dtype=torch.long,
        ).item() 
        # t_start = 999
        timesteps = torch.tensor(t_start).repeat(batch_size).to(self.accelerator.device)
        
        if not hasattr(self, "reward_model"):
            self.reward_model = ViClipReward()
            # self.reward_model = VideoScoreTensorReward(device=self.accelerator.device) # .to(self.accelerator.device, dtype=torch.bfloat16)  # Use the default VideoScoreTensorReward
            
            self.reward_model_3d = TextVideoReward().to(self.accelerator.device, dtype=torch.bfloat16)

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise = torch.randn_like(latent)
        # noise = torch.ones_like(latent) * 0.35
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # 检查 latent_noisy 是否包含 NaN 值
        if torch.any(torch.isnan(latent_noisy)):
            print("Warning: latent_noisy contains NaN values")
            print("latent_noisy:", latent_noisy)
        

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
        
        if not hasattr(self, "global_step"):  # count epochs
            self.global_step = 0
            self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
        else:
            self.global_step += 1

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        
        # Concatenate latent and image_latents in the channel dimension
        # latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)
        latents_store = []
        timesteps_store = []
        full_ts   = self.components.scheduler.timesteps                        # [N] 递减
        # 找到 <= t_start 的索引起点
        print("t_start: ", t_start)
        start_idx = (full_ts >= t_start).nonzero(as_tuple=True)[0][-1]
        print("start_idx: ", start_idx)
        timesteps_loop = full_ts[start_idx:]
        
        print("timestpes_loop: ", timesteps_loop)
        
        latents = latent_noisy
        extra_step_kwargs = {"eta": 1.0} if "eta" in self.components.scheduler.step.__code__.co_varnames else {}
        old_pred_original_sample = None
        # do_cfg = True
        # prompt_in = prompt_embedding
        # guidance_scale = 1
        # do_classifier_free_guidance = True
        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        # print(type(self.components.scheduler))
        # exit(0)
        with torch.autocast(device_type="cuda"):
            for i, t in enumerate(timesteps_loop):
                latents_concat = torch.cat([latents, image_latents], dim=2)
                self._current_timestep = t
                t_b = t.repeat(batch_size).to(self.accelerator.device)
                noise_pred = self.components.transformer(
                    hidden_states        = latents_concat,
                    encoder_hidden_states= prompt_embedding,
                    timestep             = t_b,
                    ofs                  = ofs_emb,
                    image_rotary_emb     = rotary_emb,
                    return_dict          = False,
                )[0].float()
                # if do_cfg:
                #     uncond, cond = noise_pred.chunk(2)
                #     noise_pred = uncond.float() + guidance_scale.float() * (cond.float() - uncond.float())
                    
                # next_t = timesteps_loop[i+1] if i < len(timesteps_loop)-1 else None
                t_int      = int(t.item())                                      # or t.item()
                next_t_int = int(timesteps_loop[i+1].item()) if i < len(timesteps_loop)-1 else None
                latents, old_pred_original_sample = self.components.scheduler.step(
                    noise_pred, old_pred_original_sample, t_int, next_t_int,
                    latents, **extra_step_kwargs, return_dict=False
                )

                latents = latents.to(prompt_embedding.dtype)
                # latent_store = latents
                # latent_store = latent_store.permute(0, 2, 1, 3, 4)
                # latent_store = latent_store.contiguous()
                # latents_store.append(latent_store)
                # timesteps_store.append(t_b)
                break
        latent0 = latents.permute(0, 2, 1, 3, 4).contiguous().detach()  # [B, F, C, H, W] -> [B, C, F, H, W]
            
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                for i, t in enumerate(timesteps_loop):
                    if i == 0:
                        continue  # jump over first step
                    latents_concat = torch.cat([latents, image_latents], dim=2)
                    self._current_timestep = t
                    t_b = t.repeat(batch_size).to(self.accelerator.device)
                    noise_pred = self.components.transformer(
                        hidden_states        = latents_concat,
                        encoder_hidden_states= prompt_embedding,
                        timestep             = t_b,
                        ofs                  = ofs_emb,
                        image_rotary_emb     = rotary_emb,
                        return_dict          = False,
                    )[0].float()
                    # if do_cfg:
                    #     uncond, cond = noise_pred.chunk(2)
                    #     noise_pred = uncond.float() + guidance_scale.float() * (cond.float() - uncond.float())
                        
                    # next_t = timesteps_loop[i+1] if i < len(timesteps_loop)-1 else None
                    t_int      = int(t.item())                                      # or t.item()
                    next_t_int = int(timesteps_loop[i+1].item()) if i < len(timesteps_loop)-1 else None
                    latents, old_pred_original_sample = self.components.scheduler.step(
                        noise_pred, old_pred_original_sample, t_int, next_t_int,
                        latents, **extra_step_kwargs, return_dict=False
                    )

                    latents = latents.to(prompt_embedding.dtype)
        #             latent_store = latents
        #             latent_store = latent_store.permute(0, 2, 1, 3, 4)
        #             latent_store = latent_store.contiguous()
        #             latents_store.append(latent_store)
        #             timesteps_store.append(t_b)
                    # print(latents.shape)
            
        
        latent_pred = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W] -> [B, C, F, H, W]
        # 确保形状匹配
        latent_pred = latent_pred.contiguous()
        
        # print(latent_pred.shape)
        
        # OOM
        with torch.no_grad():
            decoded_frame = self.components.vae.decode(latent_pred / self.components.vae.config.scaling_factor).sample  # [B, C, F, H, W]
            decoded_frame = decoded_frame.float()
            # print(decoded_frame.shape)
            # if self.global_step % 100 ==0:
            #     video0 = self.video_processor.postprocess_video(video=decoded_frame[0].unsqueeze(0), output_type='pil')
            #     export_to_video(video0, output_video_path=f"video_mid_{self.global_step}_0.mp4", fps=8)
            #     video1 = self.video_processor.postprocess_video(video=decoded_frame[1].unsqueeze(0), output_type='pil')
            #     export_to_video(video1.permute(1, 0, 2, 3).cpu().numpy(), output_video_path=f"video_mid_{self.global_step}_1.mp4", fps=8)
            # print(decoded_frame.shape)
            reward = self.reward_model(decoded_frame, batch["prompt"], step=self.global_step)

        # 检查 decoded_frame 是否包含 NaN 值
        if torch.any(torch.isnan(decoded_frame)):
            print("Warning: decoded_frame contains NaN values")
            
        if not hasattr(self, "reward_model_3d_optimizer"):
            self.reward_model_3d_optimizer = Adam(self.reward_model_3d.parameters(), lr=1e-4)

        reward = torch.tensor(reward).to(self.accelerator.device, dtype=torch.bfloat16)
        reward = reward.unsqueeze(1).detach()
        # print(latent_pred.shape, prompt_embedding.shape)
        # exit(0)
        # print(latent_pred.shape)
        # reward_predicted = torch.zeros_like(reward)
        # for stored_latent, stored_timestep in zip(latents_store, timesteps_store):
        #     reward_predicted += self.reward_model_3d(stored_latent, prompt_embedding, stored_timestep)
        # reward_predicted/=len(timesteps_store)
        reward_predicted = self.reward_model_3d(latent0.detach(), prompt_embedding)  # , timesteps_store[0].detach())
        
        # print("predicted_reward_3d.dtype:", predicted_reward_3d.dtype)
        # print("compress_reward_reshaped.dtype:", compress_reward_reshaped.dtype)
        # print("reward_loss dtype check done.")

        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # print("reward", reward, ", LRM reward", reward_predicted)
        reward_loss = F.mse_loss(reward, reward_predicted)
        self.reward_model_3d_optimizer.zero_grad()
        # 注意这里要保留计算图，否则等会儿要继续用 latent_pred 做 PPO 时会报错
        # reward_loss.backward(retain_graph=True)
        reward_loss.backward()
        # print(compress_reward_reshaped, predicted_reward_3d)
        # print(reward_loss)
        self.reward_model_3d_optimizer.step()
        reward_new = self.reward_model_3d(latent0, prompt_embedding).squeeze(1)
        print("reward_loss, last, new: ", reward_loss, reward_predicted, reward, reward_new)
        
        loss = -reward_new.mean()
        print(reward_new.mean())
        return loss, reward_loss, reward, reward_new

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXImageToVideoPipeline
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
            image=image,
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


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
