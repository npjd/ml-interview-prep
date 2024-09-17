from dataset import BillingualDataset, causal_mask
from config import get_weights_file_path, get_config
from model import build_model

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.backends import mps

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path 

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len):
    sos_token = tokenizer_tgt.token_to_id("[SOS]")
    eos_token = tokenizer_tgt.token_to_id("[EOS]")

    encoder_input = model.encode(encoder_input, encoder_mask)

    decoder_input = torch.tensor([[sos_token]]).to(encoder_input.device)

    while True:
        if decoder_input.size(1) >= max_len:
            break
    
        decoder_mask = causal_mask(decoder_input.size(1)).to(encoder_input.device)
        
        logits = model.decode(decoder_input, encoder_input, encoder_mask, decoder_mask)


def get_all_sentances(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_path'].format(lang))
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentances(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    
    return tokenizer

def get_ds(config): 
    ds = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds, config['tgt_lang'])

    train_size = int(len(ds) * 0.8)
    val_size = len(ds) - train_size

    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_ds = BillingualDataset(train_ds, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], max_len=config['max_len'])
    val_ds = BillingualDataset(val_ds, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], max_len=config['max_len'])


    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)


    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_model(
        d_model=config['d_model'],
        src_vocab_size=vocab_src_len,
        tgt_vocab_size=vocab_tgt_len,
        src_max_seq_len=config['max_len'],
        tgt_max_seq_len=config['max_len'],
        h=config['h'],
        N=config['N'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    )

    return model


def train_model(config):
    # ISSUE WITH MPS
    # device = torch.device('mps' if mps.is_available() else 'cpu')
    device = 'cpu'

    print(f'Using device: {device}')

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataset, _, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    writer = SummaryWriter(config['name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    inital_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Loading model from {model_filename}')
        state = torch.load(model_filename)
        inital_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range (inital_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataset, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, max_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, max_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1 max_len) — will get broadcasted 
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, max_len, max_len)

            label = batch['label'].to(device) # (batch_size, max_len)            

            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask) # (batch_size, max_len, vocab_size)    
            
            # (batch_size * max_len, vocab_size), (batch_size * max_len)
            J = loss_fn(output.view(-1, output.size(-1)), label.view(-1))

            writer.add_scalar('Loss/train', J.item(), global_step)
            writer.flush()

            J.backward()

            optimizer.step()

            optimizer.zero_grad()

            global_step += 1
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)