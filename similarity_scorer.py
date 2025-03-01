
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np

from tqdm import tqdm


class SimScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)


    def score(self, sources, targets, batch_size=4):
        score_list = []
        for i in tqdm(range(0, len(sources), batch_size), desc="Scoring"):
            source_list = sources[i: i + batch_size]
            target_list = targets[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_source = self.tokenizer(
                        source_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_target = self.tokenizer(
                        target_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    source_tokens = encoded_source['input_ids'].to(self.device)
                    source_mask = encoded_source['attention_mask'].to(self.device)

                    target_tokens = encoded_target['input_ids'].to(self.device)
                    target_mask = encoded_target['attention_mask']
                    target_len = target_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=source_tokens,
                        attention_mask=source_mask,
                        labels=target_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), target_tokens.view(-1))
                    loss = loss.view(target_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / target_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {source_list}')
                print(f'target: {target_list}')
                exit(0)
        return score_list



