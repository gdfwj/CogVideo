This repository is forked from CogVideoX and for a test.

### Running the demo

Running original model with gradio interface is at branch original, and the model with modified attention mechanism is at main branch.

#### Setup environment

```shell
git clone --recursive https://github.com/gdfwj/CogVideo.git # get diffusers module
cd CogVideo
pip install -r requirements.txt
```

make sure the python version is between 3.10 and 3.12, which is the same requirement as CogVideoX repository.

I used python=3.11 for the implementation.

To use the modified attention mechanism, you should use the modified diffusers by

```shell
pip uninstall diffusers
cd diffusers
pip install -e .
```

#### Start the demo

To start web service, you could run this command line.

```shell
python gradio_interface.py
```

then it will run on http://localhost:7860/. 

### Causal Attention Mask Modifications

I simply modified the code in 

```python
hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
```

to is_causal=True

the code is at the 2860 line in diffusers/src/diffusers/models/attention_processor.py, in the \_\_call\_\_ of CogVideoAttnProcessor2_0.

How did I find it and the reason why I modified the lib code is demonstrated in the last part of this file.

### Inference and finetune model

to inference, you could run test.py to use the provided example or try the gradio demo.

to finetune model, you need to install listed packages

```
pip install peft
pip install decord
```

I modified the finetune/train_ddp_i2v.sh to CogVideoX-5B-I2V with modified attention and proper datasets at /data/zihao/Disney-VideoGeneration-Dataset. You can try run it directly by

```shell
cd finetune
bash train_ddp_i2v.sh
```

also I wrote a jupyter notebook of finetuning, at **finetune.ipynb**, it may run slower than the bash and sometime get OOM because it will only run on single GPU without accelerate. 

#### Testing finetuned model

The last part of finetune.ipynb is the code of loading the finetuned model, the main difficulty was to find the lora config and add into the checkpoint. 

You can see the difference between output.mp4 and output-finetuned.mp4 (but difference is small because only trained for 10 epochs)

(before testing, you need to restart the kernel to avoid OOM)



### All trying progress

#### Finding Configs and places of transformer and diffusion

```python
transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
print(transformer.config)

import inspect

with open("modules.txt", "w") as f:
    for name, module in transformer.named_modules():
        cls = module.__class__
        file_path = inspect.getfile(cls)
        f.write(f"Module name: {name}, class: {cls.__name__}, file: {file_path}\n")
```

from which I found the main model is in 

site-packages/diffusers/models/transformers/cogvideox_transformer_3d.py

and attention class is in

site-packages/diffusers/models/attention_processor.py and CogVideoXAttnProcessor2_0 is used in cogvideox_transformer_3d.py

verified them by adding print.

other classes in attention_processor.py is not found in the modules list.

#### Editing attention mask in diffusers

There is is_causal parameter in Attention class, but is not used.

also attention_masks is not passed to attention block in CogVideoXBlock

```python
attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
```

the attention mask is not available to modify outside. So I try to manually set attention masks inside Attention class.

I meet OOM when simply adding

```python
hidden_size = hidden_states.size(1)  # hidden_size
causal_mask = torch.triu(torch.ones(hidden_size, hidden_size, device=hidden_states.device, dtype=torch.float16)) * -1e7  # make upper triangular matrix mask to -inf
attention_mask = causal_mask[None, None, :, :]
```

Then I find 

```python
hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
```

in CogVideoXAttenProcessor2_0 class, so I changed the is_causal to True.

Then I rethink the masks by getting the shapes of important variables

Q shape: torch.Size([2, 48, 17776, 64]) 

K shape: torch.Size([2, 48, 17776, 64]) 

V shape: torch.Size([2, 48, 17776, 64])

encoder_hidden, hidden: torch.Size([2, 226, 3072]) torch.Size([2, 17776, 3072])

Then I found some possible bugs (or I didn't understand fully?)

(1) 

```python
if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
```

is called in processor, but the attention mask will only be repeat to [attn.heads, ...] in the function, and if we reshape it as the code said, the last second dimension will always be divided by batch_size

(2)q k v is always at the length of hidden_states, but we input encoder_hidden_states into prepare_attention mask, which will cause dimensional mismatch in attention. 

#### Finding LoRA config

the checkpoints of finetuning the model through provided functions didn't contain adapter_config.json. To test the finetuned model, I have to find the lora config. 

Here I found the code in 248 line of finetune/trainer.py

```python
self.components.transformer.add_adapter(transformer_lora_config)
```

Also find parameters in finetune.schemas

```python
rank: int = 128
lora_alpha: int = 64
target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]
```

so I use generate_lora_config.py to create the config

But peft always try to find the repository but not local files. 

finally I found the demo in inference/cli_demo.py