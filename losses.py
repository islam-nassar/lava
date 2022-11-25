import torch
import torch.nn as nn
import torch.nn.functional as F
from random import randint

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the batch mean of the loss which aligns with the math definition of kl divergence. No need to divide
    by batch size afterwards.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')   #  Islam - to align with the mat definition


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the mean over all examples.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='mean') / num_classes


# adapted from https://github.com/ShaoTengLiu/Hyperbolic_ZSL/
class MyHingeLoss(torch.nn.Module):

    def __init__(self, margin, dimension):
        super(MyHingeLoss, self).__init__()
        self.M = torch.randn((dimension,dimension), requires_grad=True)
        self.margin = margin

    # TODO the correct implement should set compare_num to a large number
    def forward_val(self, output, target):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        loss = 0
        num_compare = 5
        count = 0
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            for j in range(num_compare):
                if j != i:
                    count += 1
                    t_j = target[j]
                    loss += torch.relu(self.margin - cos(t_label, v_image) + cos(t_j, v_image))
        return loss / count

    def forward(self, output, target):
        if len(output) <= 1:
            return torch.tensor(0.).cuda()  # zero loss if a single image batch is encountered
        loss = 0
        for i in range(len(output)):
            v_image = output[i]
            t_label = target[i]
            j = randint(0, len(output)-1)
            while j == i:
                j = randint(0, len(output)-1)
            t_j = target[j]
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            if not torch.allclose(t_j, t_label):
                loss += torch.relu(self.margin - cos(t_label, v_image) + cos(t_j, v_image))
            else:
                loss += torch.relu(-self.margin - cos(t_label, v_image) + cos(t_j, v_image))  # zero loss if the t_j happens to be the same as the true label t_label
        return loss / len(output)