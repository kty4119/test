from typing import Optional
import torch
import numpy as np
from ic import utils

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
  return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def contrastive_acc(logits: torch.Tensor, target: Optional[torch.Tensor] = None, topk=(1,)) -> torch.Tensor:
  """
  Args:
    logits: (N, N) predictions.
    target: (N, num_correct_answers) labels.
  """
  assert len(logits.shape) == 2, logits.shape
  batch_size = logits.shape[0]

  if target is None:
    target = torch.arange(len(logits), device=logits.device)
    return utils.accuracy(logits, target, -1, topk)
  else:
    assert len(target.shape) == 2, target.shape
    with torch.no_grad():
      maxk = max(topk)
      if logits.shape[-1] < maxk:
        print(f"[WARNING] Less than {maxk} predictions available. Using {logits.shape[-1]} for topk.")
      maxk = min(maxk, logits.shape[-1])

      # Take topk along the last dimension.
      _, pred = logits.topk(maxk, -1, True, True)  # (N, topk)
      assert pred.shape == (batch_size, maxk)

      target_expand = target[:, :, None].repeat(1, 1, maxk)  # (N, num_correct_answers, topk)
      pred_expand = pred[:, None, :].repeat(1, target.shape[1], 1)  # (N, num_correct_answers, topk)
      correct = pred_expand.eq(target_expand)  # (N, num_correct_answers, topk)
      correct = torch.any(correct, dim=1)  # (N, topk)

      res = []
      for k in topk:
        any_k_correct = torch.clamp(correct[:, :k].sum(1), max=1)  # (N,)
        correct_k = any_k_correct.float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
      return res
    
def sugar_crepe_acc(pos_score, neg_score) -> torch.Tensor:
  # print("pos_score: ", pos_score)
  # print("neg_score: ", neg_score)
  # print("pos_score shape: ", pos_score.shape) # (50, 50)이므로 대각행렬끼리 비교해야함
  diag_pos_score = torch.diag(pos_score)
  diag_neg_score = torch.diag(neg_score)
  # result = (pos_score > neg_score).float()
  result = (diag_pos_score > diag_neg_score).float()
  
  all_cnt = pos_score.shape[0]
  # print("result: ", result, result.shape)
  # print("all_cnt ",all_cnt)
  # print("score: ", torch.tensor([torch.sum(result).item() / all_cnt]))
  return torch.tensor([torch.sum(result).item() / all_cnt])