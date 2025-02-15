import gradio as gr
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel

pipe = None

def init_model():
    global pipe
    
    
    model_id = "THUDM/CogVideoX-5b-I2V"

    transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
    text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.float16,
    )

    pipe.enable_sequential_cpu_offload()


def run_inference(input_image, prompt):

    # prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."

    video = pipe(image=input_image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]
    
    video_path = "output.mp4"

    export_to_video(video, video_path, fps=8)
    
    return video_path


def build_interface():
    interface = gr.Interface(
        fn=run_inference,
        inputs=[
            gr.Image(type="pil", label="input image"), 
            gr.Textbox(lines=2, label="input text prompt")
        ],
        outputs="video",
        title="CogVideoX-5b-i2v Demo",
        description="input text and image to generate video"
    )
    return interface

if __name__ == "__main__":
    init_model()
    demo = build_interface()
    # launch at http://127.0.0.1:7860
    demo.launch(server_name="0.0.0.0", server_port=7860)
