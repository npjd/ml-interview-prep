import torch
from torch.utils.data import Dataset


class BillingualDataset(Dataset):
    def __init__(self, hf_ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, max_len=512) -> None:
        self.hf_ds = hf_ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len

        self.sos_token = torch.tensor([src_tokenizer.token_to_id('[SOS]')])
        self.eos_token = torch.tensor([src_tokenizer.token_to_id('[EOS]')])
        self.pad_token = torch.tensor([src_tokenizer.token_to_id('[PAD]')])
    
    
    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, index):
        pair = self.hf_ds[index]
        src_text = pair["translation"][self.src_lang]
        tgt_text = pair["translation"][self.tgt_lang]

        enc_input = self.src_tokenizer.encode(src_text).ids
        dec_input = self.tgt_tokenizer.encode(tgt_text).ids

        enc_num_padding = self.max_len - len(enc_input) - 2
        dec_num_padding = self.max_len - len(dec_input) - 1

        if enc_num_padding < 0 or dec_num_padding < 0:
            raise ValueError("Input text is too long")
        
        # add sos and eos to source text
        enc_input_tensor = torch.cat([
            self.sos_token,
            torch.tensor(enc_input),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding)
        ])

        # add sos to the decoder input
        dec_input_tensor = torch.cat([
            self.sos_token,
            torch.tensor(dec_input),
            torch.tensor([self.pad_token] * dec_num_padding)
        ])

        # add eos to the label (what we expect)
        label = torch.cat(
            [
                torch.tensor(dec_input),
                self.eos_token, 
                torch.tensor([self.pad_token] * dec_num_padding)
            ]
        )

        assert enc_input_tensor.size(0) == self.max_len
        assert dec_input_tensor.size(0) == self.max_len
        assert label.size(0) == self.max_len

        return {
            "encoder_input": enc_input_tensor,
            "decoder_input": dec_input_tensor,
            "encoder_mask": (enc_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, max_len),
            "decoder_mask": (dec_input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(dec_input_tensor.size(0)), # (1, max_len, max_len),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):
    mask = torch.ones(1, size, size)
    mask = torch.triu(mask, diagonal=1)
    return mask == 0