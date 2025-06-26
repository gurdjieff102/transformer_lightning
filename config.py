import torch
configs = {
    "train_source_data":"./data_en_cn/en.txt",
    "train_target_data":"./data_en_cn/cn.txt",
    "valid_source_data":"./data_en_cn/en.test.txt",
    "valid_target_data":"./data_en_cn/cn.test.txt",
    "source_tokenizer":"bert-base-uncased",
    "target_tokenizer": "bert-base-chinese",
    "source_max_seq_len":256,
    "target_max_seq_len":256,
    "batch_size":20,
    "embedding_dim": 512,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.1,
    "lr":0.0001,
    "beam_size":3,
    "device":"cuda" if torch.cuda.is_available() else "cpu",
}