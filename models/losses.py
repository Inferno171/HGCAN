"""
models/losses.py                                                     [HGCAN]
Decoupled loss:
  L = L_exist(BCE, pos_weight)             detection
    + lambda_type * L_type(CE, weights)    type, on positives only
    + lambda_dof  * L_dof                  DOF aux, on positives only

Existence and type are weighted independently: existence pos_weight controls the
detection precision/recall tradeoff; type class-weights (sqrt/linear/none + cap,
computed in train.py) balance the 7 joint types WITHOUT NoJoint dragging them.
"""
import torch
import torch.nn.functional as F

from models.constants import dof_targets_from_type


def hgcan_loss(exist_logit, type_logits, rot_logits, trans_logits, labels,
               type_weights=None, exist_pos_weight=None,
               lambda_type=1.0, lambda_dof=0.2, focal_gamma=0.0):
    """labels: (K+1) pair labels [P] (0=NoJoint, 1..7=type).
    focal_gamma>0 switches the TYPE loss from weighted CE to focal loss, which
    down-weights easy (confident) examples so the gradient is dominated by the
    hard/rare joint types. gamma=0 -> ordinary weighted CE (unchanged behaviour)."""
    if exist_logit.numel() == 0:
        z = exist_logit.new_zeros(())
        return z, {"exist": 0.0, "type": 0.0, "dof": 0.0}

    device = exist_logit.device
    exist_target = (labels > 0).float()
    pw = (torch.as_tensor(exist_pos_weight, dtype=torch.float32, device=device)
          if exist_pos_weight is not None else None)
    l_exist = F.binary_cross_entropy_with_logits(exist_logit, exist_target, pos_weight=pw)

    pos = labels > 0
    if pos.any():
        tt = labels[pos] - 1                     # 0..6
        if focal_gamma and focal_gamma > 0:
            # focal loss: (1-p_t)^gamma * CE, with the same class weights applied
            ce = F.cross_entropy(type_logits[pos], tt, weight=type_weights, reduction="none")
            pt = torch.exp(-ce)                                   # p of the true class
            l_type = ((1.0 - pt) ** focal_gamma * ce).mean()
        else:
            l_type = F.cross_entropy(type_logits[pos], tt, weight=type_weights)
        rot_t, trans_t = dof_targets_from_type(tt)
        l_dof = F.cross_entropy(rot_logits[pos], rot_t) + F.cross_entropy(trans_logits[pos], trans_t)
    else:
        l_type = exist_logit.new_zeros(())
        l_dof = exist_logit.new_zeros(())

    total = l_exist + lambda_type * l_type + lambda_dof * l_dof
    return total, {"exist": l_exist.detach().item(),
                   "type": float(l_type.detach()) if torch.is_tensor(l_type) else 0.0,
                   "dof": float(l_dof.detach()) if torch.is_tensor(l_dof) else 0.0}
