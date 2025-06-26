import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import json
import os

from config import configs
from datasets import TranslateDataset
from models import Transformer
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

def validate_epoch(model, valid_loader, epoch, n_epochs, source_pad_id, target_pad_id, device):
    model.eval()
    total_loss = []
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Validating epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        target_input = target[:, :-1]
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        preds = model(source, target_input, source_mask, target_mask)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])

    valid_loss = sum(total_loss) / len(total_loss)
    return valid_loss, total_loss


def train_epoch(model, train_loader, optim, epoch, n_epochs, source_pad_id, target_pad_id, device):
    model.train()
    total_loss = []
    bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training epoch {epoch+1}/{n_epochs}")
    for i, batch in bar:
        source, target = batch["source_ids"].to(device), batch["target_ids"].to(device)
        target_input = target[:, :-1]
        source_mask, target_mask = model.make_source_mask(source, source_pad_id), model.make_target_mask(target_input)
        preds = model(source, target_input, source_mask, target_mask)
        print(preds.shape, 'preds batch size ===== ')
        optim.zero_grad()
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=target_pad_id)
        loss.backward()
        optim.step()
        total_loss.append(loss.item())
        bar.set_postfix(loss=total_loss[-1])
    
    train_loss = sum(total_loss) / len(total_loss)
    return train_loss, total_loss


def train(model, train_loader, valid_loader, optim, n_epochs, source_pad_id, target_pad_id, device, model_path, early_stopping):
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    best_val_loss = np.Inf
    best_epoch = 1
    count_early_stop = 0
    log = {"train_loss": [], "valid_loss": [], "train_batch_loss": [], "valid_batch_loss": []}
    for epoch in range(n_epochs):
        train_loss, train_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            optim=optim,
            epoch=epoch,
            n_epochs=n_epochs,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )
        valid_loss, valid_losses = validate_epoch(
            model=model,
            valid_loader=valid_loader,
            epoch=epoch,
            n_epochs=n_epochs,
            source_pad_id=source_pad_id,
            target_pad_id=target_pad_id,
            device=device
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_epoch = epoch + 1
            # save model
            torch.save(model.state_dict(), model_path)
            print("---- Detect improment and save the best model ----")
            count_early_stop = 0
        else:
            count_early_stop += 1
            if count_early_stop >= early_stopping:
                print("---- Early stopping ----")
                break

        torch.cuda.empty_cache()

        log["train_loss"].append(train_loss)
        log["valid_loss"].append(valid_loss)
        log["train_batch_loss"].extend(train_losses)
        log["valid_batch_loss"].extend(valid_losses)
        log["best_epoch"] = best_epoch
        log["best_val_loss"] = best_val_loss
        log["last_epoch"] = epoch + 1

        with open(os.path.join(log_dir, "log.json"), "w") as f:
            json.dump(log, f)

        print(f"---- Epoch {epoch+1}/{n_epochs} | Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f} | Best Valid loss: {best_val_loss:.4f} | Best epoch: {best_epoch}")
    
    return log

def read_data(source_file, target_file):
    source_data = open(source_file).read().strip().split("\n")
    target_data = open(target_file).read().strip().split("\n")
    return source_data, target_data

class Lightning_model(L.LightningModule):
    def __init__(self):
        super(Lightning_model, self).__init__()
        self.train_src_data, self.train_trg_data = read_data(configs["train_source_data"], configs["train_target_data"])
        self.valid_src_data, self.valid_trg_data = read_data(configs["valid_source_data"], configs["valid_target_data"])
        self.source_tokenizer = AutoTokenizer.from_pretrained(configs["source_tokenizer"])
        self.target_tokenizer = AutoTokenizer.from_pretrained(configs["target_tokenizer"])
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.model = Transformer(
            source_vocab_size=self.source_tokenizer.vocab_size,
            target_vocab_size=self.target_tokenizer.vocab_size,
            embedding_dim=configs["embedding_dim"],
            source_max_seq_len=configs["source_max_seq_len"],
            target_max_seq_len=configs["target_max_seq_len"],
            num_layers=configs["n_layers"],
            num_heads=configs["n_heads"],
            dropout=configs["dropout"]
        )
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
       
    def make_source_mask(self, source_ids, source_pad_id):
        return (source_ids != source_pad_id).unsqueeze(-2)
    
    def make_target_mask(self, target_ids):
        batch_size, len_target = target_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_target, len_target), device=target_ids.device), diagonal=1)).bool()
        return subsequent_mask
    
    def train_dataloader(self):
        train_dataset = TranslateDataset(
            source_tokenizer=self.source_tokenizer, 
            target_tokenizer=self.target_tokenizer, 
            source_data=self.train_src_data, 
            target_data=self.train_trg_data, 
            source_max_seq_len=configs["source_max_seq_len"],
            target_max_seq_len=configs["target_max_seq_len"],
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=configs["batch_size"],
            shuffle=True
        )
        return train_loader
    
    def val_dataloader(self):
        valid_dataset = TranslateDataset(
            source_tokenizer=self.source_tokenizer, 
            target_tokenizer=self.target_tokenizer, 
            source_data=self.valid_src_data, 
            target_data=self.valid_trg_data, 
            source_max_seq_len=configs["source_max_seq_len"],
            target_max_seq_len=configs["target_max_seq_len"],
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=configs["batch_size"],
            shuffle=False
        )
        return valid_loader
    
    def test_dataloader(self):
        valid_dataset = TranslateDataset(
            source_tokenizer=self.source_tokenizer, 
            target_tokenizer=self.target_tokenizer, 
            source_data=self.valid_src_data, 
            target_data=self.valid_trg_data, 
            source_max_seq_len=configs["source_max_seq_len"],
            target_max_seq_len=configs["target_max_seq_len"],
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=configs["batch_size"],
            shuffle=False
        )
        return valid_loader
    
    def predict_dataloader(self):
        valid_dataset = TranslateDataset(
            source_tokenizer=self.source_tokenizer, 
            target_tokenizer=self.target_tokenizer, 
            source_data=self.valid_src_data, 
            target_data=self.valid_trg_data, 
            source_max_seq_len=configs["source_max_seq_len"],
            target_max_seq_len=configs["target_max_seq_len"],
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=configs["batch_size"],
            shuffle=False
        )
        return valid_loader
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.make_source_mask(src, self.source_tokenizer.pad_token_id), self.make_target_mask(tgt)
        preds = self.model(src, tgt, src_mask, tgt_mask)
        return preds

    def training_step(self, batch, batch_idx):
        source, target = batch["source_ids"], batch["target_ids"]
        target_input = target[:, :-1]
        preds = self(source, target_input)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=self.target_tokenizer.pad_token_id)
        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        source, target = batch["source_ids"], batch["target_ids"]
        target_input = target[:, :-1]
        preds = self(source, target_input)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=self.target_tokenizer.pad_token_id)
        self.validation_step_outputs.append(loss.detach())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        source, target = batch["source_ids"], batch["target_ids"]
        target_input = target[:, :-1]
        preds = self(source, target_input)

        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=self.target_tokenizer.pad_token_id)
        self.test_step_outputs.append(loss.detach())
        # self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # print(f"Test Loss: {loss.item()}")
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        source, target = batch["source_ids"], batch["target_ids"]
        target_input = target[:, :-1]
        preds = self(source, target_input)
        gold = target[:, 1:].contiguous().view(-1)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), gold, ignore_index=self.target_tokenizer.pad_token_id)
        self.test_step_outputs.append(loss.detach())
        # self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        print(f"Predict Loss: {loss.item()}")
        return loss

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        print(f"on_test_epoch_end Loss: {avg_loss}")
        self.test_step_outputs.clear()

    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # optim = torch.optim.Adam(model.parameters(), lr=configs["lr"], betas=(0.9, 0.98), eps=1e-9)
        optimizer = torch.optim.Adam(self.parameters(), lr=configs["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
    }

def train():
# Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val_loss",
        filename="language-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=False
    )

    # Initialize the logger
    # logger = TensorBoardLogger(save_dir="logs", name="language-translation", version="1.0")
    logger = TensorBoardLogger(save_dir="logs", name="test", version="1.0")

    # Initialize the Trainer
    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )

    model = Lightning_model()
    # Train the model
    trainer.fit(model)

    # Test the model
    # trainer.test(model, test_loader)
def test():
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )
    model = Lightning_model.load_from_checkpoint(
        checkpoint_path="checkpoints/language-epoch=99-val_loss=0.01-val_acc=0.00.ckpt"
    )

    # trainer.test(ckpt_path="/mnt/c/Users/daiyu/vscode/transformer/checkpoints/language-epoch=99-val_loss=0.01-val_acc=0.00.ckpt")
    trainer.test(model)

def predict():
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )
    model = Lightning_model.load_from_checkpoint(
        checkpoint_path="checkpoints/language-epoch=99-val_loss=0.01-val_acc=0.00.ckpt"
    )

    # trainer.test(ckpt_path="/mnt/c/Users/daiyu/vscode/transformer/checkpoints/language-epoch=99-val_loss=0.01-val_acc=0.00.ckpt")
    trainer.predict(model)


if __name__ == "__main__":
    # train()
    # test()
    predict()
