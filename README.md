This repository is forked from CogVideoX and for a test.

### Running the demo

Running original model with gradio interface is at branch original, and the model with modified attention mechanism is at main branch.

#### Setup environment

```shell
pip install -r requirements.txt
```

make sure the python version is between 3.10 and 3.12, which is the same requirement as CogVideoX repository.

I used python=3.11 for the implementation.

To use the modified attention mechanism, you should use the modified diffusers by

```
pip uninstall diffusers
cd diffusers
pip install .
```

#### Start the demo

To start web service, you could simply run this command line.

```shell
python gradio_interface.py
```

then it will run on http://localhost:7860/. 

### Causal Attention Mask Modifications



### All trying progress

#### Finding Configs and places of transformer and diffusion

```python
transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)
print(transformer.config)

import inspect

with open("modules.txt", "w") as f:
    for name, module in transformer.named_modules():
        # 取得当前module对应的类
        cls = module.__class__
        # 获取该类定义所在的文件路径
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