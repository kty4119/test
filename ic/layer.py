import torch
from torch import nn

class Adapter(nn.Module):
  """Layers used in mapping text embeddings to visual outputs."""

  def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1):
    super().__init__()
    self.num_input_tokens = num_input_tokens
    self.num_output_tokens = num_output_tokens
    self.model = nn.Linear(in_dim, out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    outputs = None
    if isinstance(self.model, nn.ModuleList):
        assert len(self.model) == x.shape[1] == self.num_input_tokens, (len(self.model), x.shape, self.num_input_tokens)
        outputs = []
        for i in range(self.num_input_tokens):
            outputs.append(self.model[i](x[:, i, :]))  # (N, D)
        outputs = torch.stack(outputs, dim=1)  # (N, T, D)
    else:
        outputs = self.model(x)
        
        if outputs.shape[1] != self.num_output_tokens:
            outputs = outputs[:, :self.num_output_tokens, :]
    return outputs  # (N, T, D)