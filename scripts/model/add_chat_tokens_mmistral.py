import os
import sys
import math
import argparse
import sentencepiece as spm

from transformers import LlamaTokenizer, MistralConfig
from transformers.convert_slow_tokenizer import import_protobuf
# add the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_multimodal_mistral import (
    MultimodalMistralConfig,
    MultimodalMistralForCausalLM
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="data/ckpts/MultimodalMistral-7B-dproj-T2P1-train_1sSlimpajama+96sCapfusion+stage2-no_sep_loss-32k-lr5e-5-warmup200-bs128-seq32768/hf/model_iter_150", help="Path to the checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="data/ckpts/MultimodalMistral-7B-dproj-T2P1-train_1sSlimpajama+96sCapfusion+stage2-no_sep_loss-32k-lr5e-5-warmup200-bs128-seq32768/hf/model_iter_150_chat_added", help="Path to the output directory")
    args = parser.parse_args()

    # === Tokenizer ===
    new_tokens = [
        "<|im_start|>",
        "<|im_end|>",
    ]
    
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(args.ckpt_dir)
    # NOTE: we don't do this as these new tokens will NOT be added to the underlying setence piece tokenizer!!!
    # tokenizer.add_tokens(new_tokens)
    # tokenizer.add_special_tokens({
    #     "additional_special_tokens": new_tokens
    # })
    tokenizer.save_pretrained(args.output_dir)

    # re-load the tokenizer to add the new tokens to the underlying sentence piece
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(args.output_dir)

    # a hack to modify the underlying sentence piece
    s = spm.SentencePieceProcessor()
    with open(tokenizer.vocab_file, "rb") as f:
        model_pb2 = import_protobuf()
        model = model_pb2.ModelProto.FromString(f.read())

        for token in new_tokens:
            new_token = model_pb2.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            model.pieces.append(new_token)

    with open(tokenizer.vocab_file, 'wb') as f:
        f.write(model.SerializeToString())

    # reload the tokenizer to reflect the changes
    tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(args.output_dir)
    
    # === Config ===
    config = MultimodalMistralConfig.from_pretrained(args.ckpt_dir)
    print(config)

    # === Model ===
    model = MultimodalMistralForCausalLM.from_pretrained(args.ckpt_dir, config=config)
    model.resize_token_embeddings(len(tokenizer))

    # === Update config vocab size ===
    config.vocab_size = len(tokenizer) # len(tokenizer)
    print(config)

    # # === Test Model Output ===
    # vision_patches = torch.randn(32 * 32, 32 * 32 * 3) # (n_patches, 32 * 32 * 3)
    # n_patches = vision_patches.shape[0]
    # prompt = f"This is an image of a dog. <vision>{''.join(['<vpatch>'] * n_patches)}</vision>"
    # inputs = tokenizer(prompt, return_tensors="pt")
    
    # # prep text inputs
    # input_ids = inputs["input_ids"] # (batch_size, seq_len)
    # attention_mask = inputs["attention_mask"] # (batch_size, seq_len)
    
    # # prep vision indices
    # assert input_ids.shape[0] == 1, "Example only work with batch_size=1"
    # vision_patch_indices = torch.where(
    #     input_ids[0] == tokenizer.convert_tokens_to_ids('<vpatch>'),
    #     torch.zeros_like(input_ids[0]),  # ready to be filled
    #     torch.ones_like(input_ids[0]) * -1,
    # ) # (seq_len)
    # # fill patch indices with order using torch scatter
    # vision_patch_indices = vision_patch_indices.scatter_(
    #     dim=0,
    #     index=torch.where(vision_patch_indices == 0)[0],
    #     src=torch.arange(n_patches)
    # )
    # vision_patch_indices = vision_patch_indices.unsqueeze(0) # (batch_size, seq_len)

    # outputs = model(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     vision_patch_indices=vision_patch_indices,
    #     vision_patches=vision_patches
    # )

    # Save others to output_dir
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
