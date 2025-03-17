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
from finetune.ddpo_pytorch.ddpo_pytorch.rewards import jpeg_compressibility, compute_temporal_consistency, compute_lpips_reward, simple_MSE_reward
from finetune.ddpo_pytorch.ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob

from ..utils import register
from torch import nn
import wandb
import torch.nn.functional as F
from torch.optim import Adam

class LatentRewardFunction3D(nn.Module):
    """
    用一个简单的 3D Conv + 全连接层做示例。
    你也可以换成 3D Transformer、ViT-3D 等更复杂结构。
    """
    def __init__(self, in_channels=16, hidden_dim=64):
        super().__init__()
        # 简单 3D Conv 例子
        self.conv3d_1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d((2, 2, 2))  # 在 T,H,W 三维都做 pool
        
        # 全连接层，输出1个标量作为奖励
        # 假设后面 flatten 后大小是 some_dim，这里根据你的输入大小实际计算
        self.fc1 = nn.Linear(64 *  (8164), hidden_dim)  # some_dim 需要根据视频帧数/空间分辨率计算
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, video_latent: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        video_latent: [B, C, F, H, W] (比如 [batch_size, 4, frames, 64, 64])
        text_emb: [B, seq_len, emb_dim] (比如 [batch_size, 77, 768])，或根据你的 text encoder 输出而定
        
        返回值： [B, 1] 的标量奖励
        """
        # (B, C, F, H, W) => (B, 32, F, H, W)
        # print(video_latent.shape)
        x = F.relu(self.conv3d_1(video_latent))
        x = F.relu(self.conv3d_2(x))
        x = self.pool(x)  # (B, 64, F/2, H/2, W/2) => flatten => (B, -1)

        x = x.flatten(start_dim=1)  # [B, 64*(F/2)*(H/2)*(W/2)]
        
        # 简单做法：把文本特征与 video flatten 拼接
        # 也可以做 cross-attention 或者 separate encoding
        text_feat = text_emb.mean(dim=1)  # [B, emb_dim], 这里简单做个平均
        x = torch.cat([x, text_feat], dim=1)  # [B, 64*(...)+emb_dim]
        # print(x.shape)

        x = F.relu(self.fc1(x))
        reward = self.fc2(x)  # [B, 1]
        return reward


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
        ret = {"encoded_videos": [], "prompt_embedding": [], "images": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["images"].append(image)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["images"] = torch.stack(ret["images"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
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

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.components.scheduler.config.num_train_timesteps, (batch_size,), device=self.accelerator.device
        )
        
        if not hasattr(self, "reward_model"):
            # self.reward_model = jpeg_compressibility()
            self.reward_model = simple_MSE_reward
            
            self.reward_model_3d = LatentRewardFunction3D(
                in_channels=16,    # 跟你的 VAE latent channel 对齐
                hidden_dim=64     # 可以根据需要调整
            ).to(self.accelerator.device, dtype=torch.bfloat16)

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (image_latents.shape[0], *image_latents.shape[2:])

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        # noise = torch.randn_like(latent)
        noise = torch.ones_like(latent) * 0.35
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

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None if self.state.transformer_config.ofs_embed_dim is None else latent.new_full((1,), fill_value=2.0)
        )
        
        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)
        # latent_img_noisy_2 = latent_img_noisy.detach()
        # prompt_embedding_2 = prompt_embedding.detach()
        # print(latent_img_noisy.shape)
        
        
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        with torch.autocast(device_type="cuda"):
            # print("get in")
            latent_pred, log_prob = ddim_step_with_logprob(
                self=self.components.scheduler,
                model_output=predicted_noise,
                timestep=timesteps,
                sample=latent,
                eta=1.0,  # 和DDPO一样
                num_inference_steps=50
            )
            
        
        latent_pred = latent_pred.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W] -> [B, C, F, H, W]
        # 确保形状匹配
        latent_pred = latent_pred.contiguous()
        
        # print(latent_pred.shape)
        
        # OOM
        with torch.no_grad():
            decoded_frame = self.components.vae.decode(latent_pred / self.components.vae.config.scaling_factor).sample  # [B, C, F, H, W]
            reward = self.reward_model(decoded_frame) 

        # 检查 decoded_frame 是否包含 NaN 值
        if torch.any(torch.isnan(decoded_frame)):
            print("Warning: decoded_frame contains NaN values")
            
        if not hasattr(self, "reward_model_3d_optimizer"):
            self.reward_model_3d_optimizer = Adam(self.reward_model_3d.parameters(), lr=1e-5)

        compress_reward = torch.tensor(reward).to(self.accelerator.device, dtype=torch.bfloat16)
        compress_reward_reshaped = compress_reward.unsqueeze(1).detach()
        predicted_reward_3d = self.reward_model_3d(latent_pred, prompt_embedding)
        
        # print("predicted_reward_3d.dtype:", predicted_reward_3d.dtype)
        # print("compress_reward_reshaped.dtype:", compress_reward_reshaped.dtype)
        # print("reward_loss dtype check done.")

        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        reward_loss = F.mse_loss(predicted_reward_3d, compress_reward_reshaped)
        self.reward_model_3d_optimizer.zero_grad()
        # 注意这里要保留计算图，否则等会儿要继续用 latent_pred 做 PPO 时会报错
        reward_loss.backward(retain_graph=True)
        # print(compress_reward_reshaped, predicted_reward_3d)
        # print(reward_loss)
        self.reward_model_3d_optimizer.step()
        
        reward_new = self.reward_model_3d(latent_pred, prompt_embedding).squeeze(1)
        # print(rewards)
        # rewards = rewards.detach()
        
        # print("rewards_std", rewards.std())
        if not hasattr(self, "global_step"):  # count epochs
            self.global_step = 0
        else:
            self.global_step += 1
            
        # it is NOT DPO
        
        loss = -reward_new.mean()
        # advantages = reward_new
        # if not hasattr(self, "advantages") or self.global_step%30==0:  # store the first sample of epoch
        #     self.advantages = advantages.detach()
        # else:
        #     advantages = self.advantages
        # advantages = torch.clamp(advantages, -5, 5)
        
        # advantages = torch.clamp(
        #     advantages,
        #     -5,
        #     5,
        # )
        # if not hasattr(self, "old_log_probs"):  # store the first sample of epoch
        #     ratio = torch.exp(log_prob - log_prob)
        #     self.old_log_probs = log_prob.detach()
        # elif self.global_step%30==0:
        #     ratio = torch.exp(log_prob - self.old_log_probs)
        #     self.old_log_probs = log_prob.detach()
        # else:
        #     ratio = torch.exp(log_prob - self.old_log_probs)
        
        # print(log_prob, self.old_log_probs)
        # unclipped_loss = -advantages * ratio
        # clipped_loss = -advantages * torch.clamp(
        #     ratio,
        #     1.0 - 0.1,
        #     1.0 + 0.1,
        # )
        
        # loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        # print(loss)
        # if torch.any(torch.isnan(loss)):
        #     print("Warning: loss contains NaN values")

        # alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        # weights = 1 / (1 - alphas_cumprod)
        # while len(weights.shape) < len(latent_pred.shape):
        #     weights = weights.unsqueeze(-1)

        # loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        # loss = loss.mean()
        # exit(0)
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
