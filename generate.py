import torch
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, GPTConfig
from tokenizer import build_tokenizer

from torch.nn import functional as F


def load_model(model_path, config):
    model = GPT(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def sample_from_logits(logits, temp=1.0):
    # logits = top_k_logits(logits, top_k)
    # Greedy selection: choose the token with highest score (argmax).
    # Preserve the (B, 1) shape so callers expecting a column vector continue to work.
    scaled = logits / (temp if temp != 0 else 1.0)
    next_token = torch.argmax(scaled, dim=-1, keepdim=True)
    return next_token


def generate_sample(model, tokenizer, conditions, max_length):
    model.eval()
    input_ids = tokenizer.generation_encode(conditions)
    # place input on the same device as the model
    device = next(model.parameters()).device
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    len_conditions = len(input_ids[0])

    with torch.no_grad():
        for _ in range(max_length - len_conditions):

            # 1. Forward pass to get logits
            # We pass the current input_ids.
            # The model returns (logits, loss, attn_maps). We only need logits.
            logits, _, _ = model(input_ids)
            
            # 2. Get logits for the last token only
            # logits shape is (Batch, Sequence_Length, Vocab_Size)
            # We want the last step: logits[:, -1, :]
            next_token_logits = logits[:, -1, :]

            # 3. Sample the next token
            next_token = sample_from_logits(next_token_logits)

            # 4. Append the new token to the sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)

            # 5. Check stopping conditions
            # 2 is </s> (EOS), 0 is <pad>
            if next_token.item() == tokenizer.vocab["</s>"] or next_token.item() == tokenizer.vocab["<pad>"]:
                break


    generated_text = tokenizer.decode(input_ids[0][len_conditions:])
    return generated_text


def generate(args):

    data_SCAN = load_dataset("scan", args.data_split, trust_remote_code=True)

    max_len = args.max_len
    tokenizer, vocab_size = build_tokenizer(args, data_SCAN, max_len, args.output_tokenizer_dir)

    mconf = GPTConfig(vocab_size, max_len,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      isconditional=True)

    # Load model and tokenizer
    print("loading model")
    model = load_model(args.ckpt_path, mconf).cuda()
    print('total params:', sum(p.numel() for p in model.parameters()))


    # Sample generation
    test_data = data_SCAN['test']
    correct_count = 0
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for i, data in pbar:
        generated_actions = generate_sample(model, tokenizer, data['commands'], max_len)
        if generated_actions == data['actions']:
            correct_count += 1
        pbar.set_description(f'Accuracy: {correct_count / (i + 1):.4f}')
    print(f'Test accuracy: {correct_count / len(test_data)}')
