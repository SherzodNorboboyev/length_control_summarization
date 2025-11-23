import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LocalBARTScorer:
    """Lightweight BARTScore-style scorer (HF BART).
    Computes average log-probability of target given source.
    Higher is better (less negative); units are nats/token.
    """
    def __init__(self, device: Optional[str] = None, checkpoint: str = "facebook/bart-large-cnn"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, sources: List[str], targets: List[str], batch_size: int = 4, max_src_len: int = 1024, max_tgt_len: int = 128) -> List[float]:
        assert len(sources) == len(targets), "sources and targets must have the same length"
        scores = []
        for i in range(0, len(sources), batch_size):
            src_batch = sources[i:i+batch_size]
            tgt_batch = targets[i:i+batch_size]

            enc = self.tokenizer(src_batch, max_length=max_src_len, truncation=True, padding=True, return_tensors="pt").to(self.device)
            tgt = self.tokenizer(text_target=tgt_batch, max_length=max_tgt_len, truncation=True, padding=True, return_tensors="pt").to(self.device)

            labels = tgt["labels"] if "labels" in tgt else tgt["input_ids"]
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
            outputs = self.model(**enc, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits  # [B, T, V]

            log_probs = F.log_softmax(logits, dim=-1)
            target_ids = labels.clone()
            pad_id = self.tokenizer.pad_token_id
            mask = (target_ids != pad_id) & (target_ids != -100)
            target_ids[target_ids == -100] = 0
            gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            gathered = gathered * mask
            lengths = mask.sum(dim=-1).clamp(min=1)
            seq_scores = gathered.sum(dim=-1) / lengths
            scores.extend(seq_scores.detach().cpu().tolist())
        return scores