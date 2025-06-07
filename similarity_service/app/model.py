import torch.nn as nn
from transformers import AutoModel

class RegressionModel_v3(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(RegressionModel_v3, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.regression_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        raw_output = self.regression_head(mean_pooled)
        predictions = raw_output * 5
        return predictions
