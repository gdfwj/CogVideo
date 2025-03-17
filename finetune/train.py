import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args
import wandb
import torch

def main():
    args = Args.parse_args()
    torch.autograd.set_detect_anomaly(True)
    wandb.init(
        project="pandora_distill",  # 你的项目名称
        name="DDPO_finetune",  # 运行的名称
        config=args
    )
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
