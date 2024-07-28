import torch.nn.functional as F
from accelerate import Accelerator
import yaml
import argparse
from tqdm import tqdm
import src.model
import src.data
import torch
import os
import math
from transformers import get_scheduler
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from transformers import LlamaTokenizer, GenerationConfig
from transformers import SoloForCausalLM

import os
from src.data.utils import (
    load_image_to_base64,
    download_image_to_base64,
    load_base64_to_PILImage,
    convert_image_base64_to_patches,
    visualize_patches
)
from tqdm import tqdm
import jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)



    max_epochs = config["training"]["max_epochs"]
    accumulate_grad_batches = config["training"]["accumulate_grad_batches"]
    checkpoint_every_n_steps = config["training"]["checkpoint_every_n_steps"]
    output_dir = config['output_dir']
    
    # load model and data, optimizer
    MODEL_PATH = config["model"]["checkpoint"]
    tokenizer_path = MODEL_PATH

    
    
    model = SoloForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    
    accelerator = Accelerator(gradient_accumulation_steps=accumulate_grad_batches, log_with="wandb", mixed_precision='bf16')    
            

   
    data_module = getattr(
        src.data, config["data"]["data_module"]
    )(config, tokenizer)
    data_loader = data_module.train_dataloader()
    lr = config["training"]["lr"]
    # optimize those parameters that require grad
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"]["weight_decay"],
    )
    
    # add lr scheduler
    num_update_steps_per_epoch = math.ceil(len(data_loader) / accelerator.gradient_accumulation_steps)
    
    # if args.max_train_steps is None:
    max_train_steps = max_epochs * num_update_steps_per_epoch
    num_warmup_steps = 50

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )

    model, optimizer, data, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
    
    
    
    global_bs = config["data"]['batch_size'] * accelerator.num_processes * accumulate_grad_batches
    
    accelerator.init_trackers(project_name=config['logger']['proj_dir'], init_kwargs={"wandb": {"name": f"lr{lr}-gbs{global_bs}"}} )
    model.train()
    
    
    step = 0
    for epoch in range(max_epochs):
        for batch in tqdm(data):
            step += 1
            with accelerator.accumulate(model):
                if batch is None: continue
                # if accelerator.is_main_process:
                    # print(batch)
                outputs = model(**batch)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                accelerator.log({"loss":loss.item(), "perplexity":perplexity.item()})
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if step % checkpoint_every_n_steps == 0:
                    # accelerator.wait_for_everyone()
                    # if accelerator.is_main_process:
                        # print("Saving model")
                    record_step = step // accumulate_grad_batches
                    if os.path.exists(output_dir):
                        pass
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        os.path.join(output_dir, f"{record_step}.pt"),
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        safe_serialization=False
                    )

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(output_dir, f"final.pt"),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=False
    )
  

if __name__ == "__main__":
    main()