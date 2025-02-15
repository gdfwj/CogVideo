from peft import LoraConfig
import json
transformer_lora_config = LoraConfig(
                r=128,
                lora_alpha=64,
                init_lora_weights=True,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            )

lora_config_dict = transformer_lora_config.to_dict()

for key, value in lora_config_dict.items():
    if isinstance(value, set):
        lora_config_dict[key] = list(value)

config_path = "adapter_config.json"

with open(config_path, "w") as f:
    json.dump(lora_config_dict, f, indent=4)