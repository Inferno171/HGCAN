"""
models/constants.py                                                  [HGCAN]
(K+1) class order and the joint-type -> DOF signature used by the auxiliary
regularizer. Kept independent of the geometry stack so the model imports without
occwl. The class order MUST match data/build_assembly_dataset.JOINT_TYPES_K1
(canonical Fusion order); a mismatch would misalign DOF targets.
"""
import torch

# index 0 = NoJoint, 1..7 = Fusion joint types (canonical order)
JOINT_TYPES_K1 = [
    "NoJoint",
    "RigidJointType",
    "RevoluteJointType",
    "SliderJointType",
    "CylindricalJointType",
    "PinSlotJointType",
    "PlanarJointType",
    "BallJointType",
]
NUM_CLASSES = len(JOINT_TYPES_K1)  # 8

# DOF signature per type: (n_rot, n_trans)
_DOF = {
    "NoJoint":              (0, 0),
    "RigidJointType":       (0, 0),
    "RevoluteJointType":    (1, 0),
    "SliderJointType":      (0, 1),
    "CylindricalJointType": (1, 1),
    "PinSlotJointType":     (1, 1),
    "PlanarJointType":      (1, 2),
    "BallJointType":        (3, 0),
}
_ROT_MAP = {0: 0, 1: 1, 3: 2}     # rot count -> class idx  ({0,1,3} -> {0,1,2})
N_ROT_CLASSES = 3
N_TRANS_CLASSES = 3               # trans count in {0,1,2} == class idx

# graph relation counts (must match the extractors)
#   entity graph (step_graph): 0 convex / 1 concave / 2 smooth / 3 incidence
NUM_ENTITY_RELATIONS = 4
#   assembly graph (build_assembly_dataset): contact / knn / same-occurrence
NUM_ASM_RELATIONS = 3
#   hole feature width per body node
HOLE_DIM = 6

# precomputed per-(K+1)-class DOF target indices
_ROT_TGT = torch.tensor([_ROT_MAP[_DOF[n][0]] for n in JOINT_TYPES_K1], dtype=torch.long)
_TRANS_TGT = torch.tensor([_DOF[n][1] for n in JOINT_TYPES_K1], dtype=torch.long)


def dof_targets_from_labels(labels: torch.Tensor):
    """(K+1) class labels [P] -> (n_rot_idx [P], n_trans_idx [P]) for the DOF aux loss."""
    return _ROT_TGT.to(labels.device)[labels], _TRANS_TGT.to(labels.device)[labels]


# ---- decoupled two-head setup: 7 joint types (NoJoint handled by existence head) ----
JOINT_TYPES_7 = JOINT_TYPES_K1[1:]      # Rigid..Ball
NUM_JOINT_TYPES = len(JOINT_TYPES_7)    # 7

_ROT_TGT7 = torch.tensor([_ROT_MAP[_DOF[n][0]] for n in JOINT_TYPES_7], dtype=torch.long)
_TRANS_TGT7 = torch.tensor([_DOF[n][1] for n in JOINT_TYPES_7], dtype=torch.long)


def dof_targets_from_type(type_labels: torch.Tensor):
    """type labels in 0..6 (Rigid..Ball) -> (n_rot_idx, n_trans_idx) for DOF aux."""
    return _ROT_TGT7.to(type_labels.device)[type_labels], _TRANS_TGT7.to(type_labels.device)[type_labels]


# ---- hierarchical (DOF-factored) type decoding: Design A ----
# signature (rot_idx, trans_idx) -> candidate type indices (0..6 in JOINT_TYPES_7).
# Only (1,1) is ambiguous (Cylindrical/PinSlot); the type head breaks that tie.
_SIG_TO_TYPES = {}
for _ti, _name in enumerate(JOINT_TYPES_7):
    _nr, _nt = _DOF[_name]
    _SIG_TO_TYPES.setdefault((_ROT_MAP[_nr], _nt), []).append(_ti)


def hierarchical_type_pred(rot_logits, trans_logits, type_logits):
    """Design A: predict (n_rot, n_trans) from the DOF heads, map to type via the
    signature table; where a signature maps to >1 type (Cylindrical/PinSlot) the
    7-way type head breaks the tie; an unmapped (rot,trans) combo falls back to the
    type head. Returns type indices 0..6 (caller adds 1 for the K+1 label space)."""
    ri = rot_logits.argmax(-1)
    ti = trans_logits.argmax(-1)
    P = rot_logits.size(0)
    out = torch.zeros(P, dtype=torch.long, device=rot_logits.device)
    for p in range(P):
        cands = _SIG_TO_TYPES.get((int(ri[p]), int(ti[p])))
        if not cands:
            out[p] = int(type_logits[p].argmax())
        elif len(cands) == 1:
            out[p] = cands[0]
        else:
            out[p] = cands[int(type_logits[p, cands].argmax())]
    return out
