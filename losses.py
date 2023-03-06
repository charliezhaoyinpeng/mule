import torch
import torch.nn as nn
import torch.nn.functional as F


def ava_pose_softmax_func(logits):
    pose_logits = nn.Softmax(dim=1)(logits[:, :13])
    interact_logits = nn.Sigmoid()(logits[:, 13:])
    logits = torch.cat([pose_logits, interact_logits], dim=1)
    logits = torch.clamp(logits, min=0., max=1.)
    return logits


def ava_pose_softmax_criterion(logits, targets):
    logits = ava_pose_softmax_func(logits)
    return F.binary_cross_entropy(logits, targets)


def ava_edl_criterion(B_alpha, B_beta, targets):
    edl_loss = torch.mean(targets * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha)) + (1 - targets) * (
            torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)))
    return edl_loss


def uncertainty_pos(B_alpha, B_beta, num_class=40, tau=1):
    return 2 / (1 + torch.exp(tau * (torch.sum(B_alpha, dim=1) - num_class)))
    # return num_class / (torch.sum(lamb * B_alpha + (1 - lamb) * B_beta, dim=1))


def uncertainty_neg(B_alpha, B_beta, num_class=40, tau=1):
    return 2 / (1 + torch.exp(tau * (num_class - torch.sum(B_beta, dim=1)))) - 1


def uncertainty_posneg(B_alpha, B_beta, num_class=40, lamb=0.5):
    return num_class / (torch.sum(lamb * B_alpha + (1 - lamb) * B_beta, dim=1))


def bc_operator(b_i, b_j):
    return b_i + b_j - b_i * b_j


def cal_b(b):
    ans = 0
    for i in range(len(b)):
        ans = bc_operator(ans, b[i])
    return ans


def belief(B_alpha, B_beta):
    b_actions = (B_alpha - 1) / (B_alpha + B_beta)
    num = b_actions.shape[0]
    ans = []
    for ind in range(num):
        b_ind = b_actions[ind]
        ans.append(1 - cal_b(b_ind))
    return torch.tensor(ans)


def ava_criterion(pose_softmax=False, use_edl=False):
    if use_edl:
        return ava_edl_criterion, belief
    if pose_softmax:
        return ava_pose_softmax_criterion, ava_pose_softmax_func
    return nn.BCEWithLogitsLoss(), nn.Sigmoid()
