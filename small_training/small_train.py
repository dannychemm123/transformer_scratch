import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from small_dataset import SmallBilingualDataset
from model import build_transformer


# =====================================================
# Build tokenizer
# =====================================================

def build_tokenizer(sentences):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        min_frequency=1
    )

    tokenizer.train_from_iterator(sentences, trainer)
    return tokenizer


# =====================================================
# Visualize encoder attention
# =====================================================

def log_encoder_attention(model, writer, dataset, tokenizer_src, device, epoch):
    model.eval()

    sample = dataset[0]
    encoder_input = sample["encoder_input"].unsqueeze(0).to(device)
    encoder_mask = sample["encoder_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        model.encode(encoder_input, encoder_mask)

    # First encoder layer
    attention = model.encoder.layers[0].self_attention_block.attention_scores

    # attention: (batch, heads, seq, seq)
    # First sample, first attention head
    attention = attention[0, 0].cpu().numpy()

    ids = encoder_input[0].cpu().tolist()

    tokens = []
    valid_length = 0

    for token_id in ids:
        token = tokenizer_src.id_to_token(token_id)
        if token == "[PAD]":
            break
        tokens.append(token)
        valid_length += 1

    attention = attention[:valid_length, :valid_length]

    # -------------------------
    # Create heatmap
    # -------------------------

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(attention)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    ax.set_title(f"Encoder Layer 1 - Head 1 - Epoch {epoch}")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    writer.add_figure("Attention/Encoder_Layer1_Head1", fig, global_step=epoch)
    plt.close(fig)

    model.train()


# =====================================================
# Train
# =====================================================

def train_model(config, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------------------------
    # Extract English and Italian sentences
    # ---------------------------------------------

    src_sentences = [src for src, tgt in data]
    tgt_sentences = [tgt for src, tgt in data]

    # ---------------------------------------------
    # Tokenizers
    # ---------------------------------------------

    tokenizer_src = build_tokenizer(src_sentences)
    tokenizer_tgt = build_tokenizer(tgt_sentences)

    print("English vocabulary:", tokenizer_src.get_vocab_size())
    print("Italian vocabulary:", tokenizer_tgt.get_vocab_size())

    # ---------------------------------------------
    # Dataset
    # ---------------------------------------------

    dataset = SmallBilingualDataset(data, tokenizer_src, tokenizer_tgt, config["seq_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # ---------------------------------------------
    # Model
    # ---------------------------------------------

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
        h=config["h"],
        dropout=config["dropout"],
        N=config["N"],
        d_ff=config["d_ff"]
    )

    model = model.to(device)

    # ---------------------------------------------
    # Count parameters
    # ---------------------------------------------

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {parameters:,}")

    # ---------------------------------------------
    # Loss
    # ---------------------------------------------

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1
    )

    # ---------------------------------------------
    # Optimizer
    # ---------------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # ---------------------------------------------
    # TensorBoard
    # ---------------------------------------------

    writer = SummaryWriter(config["experiment_name"])
    global_step = 0

    # =============================================
    # Epoch loop
    # =============================================

    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            # --------------------------------
            # Encoder / Decoder / Projection
            # --------------------------------

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            logits = model.project(decoder_output)  # (batch, seq_len, target_vocab_size)

            # --------------------------------
            # Loss
            # --------------------------------

            loss = loss_fn(
                logits.reshape(-1, tokenizer_tgt.get_vocab_size()),
                label.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # TensorBoard batch loss
            writer.add_scalar("Loss/batch", loss.item(), global_step)
            global_step += 1

        average_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1:03d} | Loss: {average_loss:.4f}")

        # TensorBoard epoch loss
        writer.add_scalar("Loss/epoch", average_loss, epoch)

        # Attention visualization
        log_encoder_attention(model, writer, dataset, tokenizer_src, device, epoch)

        writer.flush()

    writer.close()

    return model, tokenizer_src, tokenizer_tgt