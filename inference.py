import torch
import time
from config import configs
from evaluate import load_model_tokenizer, translate

def main():   
    import time
    # Translate a sentence
    sentence = "some localities and departments , however , failed to successfully implement these policy decisions and plans . as a result , corrupt phenomena are not effectively checked ."
    print("--- English input sentence:", sentence)
    print("--- Translating...")
    device = torch.device(configs["device"])
    model, source_tokenizer, target_tokenizer = load_model_tokenizer(configs)
    st = time.time()
    trans_sen = translate(
        model=model, 
        sentence=sentence, 
        source_tokenizer=source_tokenizer, 
        target_tokenizer=target_tokenizer, 
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],      
        device=device
    )
    end = time.time()
    print("--- Sentences translated into Chinese:", trans_sen)
    print(f"--- Time: {end-st} (s)")

if __name__ == "__main__":
    main()