import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import re
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from config import configs
from models import Transformer
from datasets import TranslateDataset
from train import read_data
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction()
import lightning as L
from train import Lightning_model


def load_model_tokenizer(configs):
    """
    This function will load model and tokenizer from pretrained model and tokenizer
    """
    device = torch.device(configs["device"])
    source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
    target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])  

    # Load model Transformer
    model = Transformer(
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        embedding_dim=configs["embedding_dim"],
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        num_layers=configs["n_layers"],
        num_heads=configs["n_heads"],
        dropout=configs["dropout"]
    )

    model = Lightning_model.load_from_checkpoint(
        checkpoint_path="checkpoints/language-epoch=99-val_loss=0.01-val_acc=0.00.ckpt"
    )

    model.eval()
    model.to(device)
    print(f"Done load model on the {device} device")  
    return model, source_tokenizer, target_tokenizer


def translate(model, sentence, source_tokenizer, target_tokenizer, source_max_seq_len=256, 
    target_max_seq_len=256, device=torch.device("cpu")):
    """
    This funciton will translate give a source sentence and return target sentence using beam search
    """
    # Convert source sentence to tensor
    source_tokens = source_tokenizer.encode(sentence)[:source_max_seq_len]
    source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
    # Create source sentence mask
    source_mask = model.make_source_mask(source_tensor, source_tokenizer.pad_token_id).to(device)
    # Feed forward Encoder
    encoder_output = model.model.encoder.forward(source_tensor, source_mask)
    generated = torch.tensor([target_tokenizer.cls_token_id]).unsqueeze(0).to(device)
    for _ in range(target_max_seq_len):
      
        target_mask = model.make_target_mask(generated).to(device)
        pred = model.model.decoder.forward(generated, encoder_output, source_mask, target_mask)
        pred = F.softmax(model.model.final_linear(pred), dim=-1)
        next_token = torch.argmax(pred[:, -1, :], dim=-1)
        next_token = next_token.unsqueeze(0)
        generated = torch.cat((generated, next_token), dim=1)
        if (next_token == target_tokenizer.sep_token_id).all():
            break

    target_tokens = generated[0]
    # Convert target sentence from tokens to string
    target_sentence = target_tokenizer.decode(target_tokens, skip_special_tokens=True)
    return target_sentence