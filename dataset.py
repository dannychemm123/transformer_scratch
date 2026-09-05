import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,seq_len,lang_src,lang_tgt)->None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    
    def __getitem__(self,index:Any)->Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len-len(enc_input_tokens)-2
        dec_num_padding_tokens = self.seq_len-len(dec_input_tokens)-1

        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError("Sentence is too long for the seq_len")
        
        # Add SOS and ROS to the source text
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*enc_num_padding_tokens,dtype=torch.int64)
        ])

        # Decoder Input: Add SOS token at the beginning and padding tokens at the end
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype=torch.int64)

        ])

        # Target: Add EOS token at the end and padding tokens at the end
        label = torch.cat([
            torch.tensor(dec_input_tokens,dtype = torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*dec_num_padding_tokens,dtype = torch.int64)

        ])

        assert encoder_input.shape[0] == self.seq_len
        assert decoder_input.shape[0] == self.seq_len
        assert label.shape[0] == self.seq_len

        return  {
            "encoder_input":encoder_input, # (seq_len,)
            "decoder_input":decoder_input, # (seq_len,)
            "encoder_mask": (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            "decoder_mask": (decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]), #(1,1,seq_len) & (1,seq_len,seq_len)
            "label":label, # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text

        
        
        }
def causal_mask(size):
    """Create a mask to prevent the decoder from attending to future positions"""
    mask = torch.triu(torch.ones(1,size,size), diagonal= 1).type(torch.int)
    return mask ==0
    
