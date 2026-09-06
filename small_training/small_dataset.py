from torch.utils.data import Dataset
import torch


class SmallBilingualDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer_src,
        tokenizer_tgt,
        seq_len
    ):

        self.data = data
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len

        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")])

        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")])

        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")])

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        src_text, tgt_text = self.data[idx]
        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        encoder_input = torch.cat([
                self.sos_token,
                torch.tensor(enc_tokens),
                self.eos_token,
                torch.tensor(
                    [self.pad_token.item()] *(self.seq_len- len(enc_tokens)- 2))])

        decoder_input = torch.cat([ self.sos_token, torch.tensor(dec_tokens), torch.tensor([self.pad_token.item()] *
             (
                        self.seq_len
                        - len(dec_tokens)
                        - 1 ))])

        label = torch.cat(
            [
                torch.tensor(dec_tokens),
                self.eos_token,
                torch.tensor(
                    [self.pad_token.item()] *
                    (
                        self.seq_len
                        - len(dec_tokens)
                        - 1
                    )
                )
            ]
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (
                encoder_input != self.pad_token
            ).unsqueeze(0).unsqueeze(0).int(),

            "decoder_mask": (
                (decoder_input != self.pad_token)
                .unsqueeze(0)
                .unsqueeze(0)
                & causal_mask(self.seq_len)
            ).int(),

            "label": label,

            "src_text": src_text,
            "tgt_text": tgt_text
        }


def causal_mask(size):

    mask = torch.triu(
        torch.ones(1, size, size),
        diagonal=1
    ).type(torch.int)

    return mask == 0