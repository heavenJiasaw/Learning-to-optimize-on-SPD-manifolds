import torch
import torch.nn as nn
from torch.autograd import Function


class EigLayerF(Function):
    @staticmethod
    def forward(ctx, input):

        n = input.shape[0]
        S = torch.zeros(input.shape, device=input.device)
        U = torch.zeros(input.shape, device=input.device)

        for i in range(n):
            # 使用新的 linalg.eig 替代旧的 eig
            values_complex, vectors_complex = torch.linalg.eig(input[i])
            # 提取实部值
            values = torch.real(values_complex)
            vectors = torch.real(vectors_complex)
            S[i] = torch.diag(values)
            U[i] = vectors

        ctx.save_for_backward(input, S, U)
        return S, U

    @staticmethod
    def backward(ctx, grad_S, grad_U):
        input, S, U = ctx.saved_tensors
        n = input.shape[0]
        dim = input.shape[1]
        grad_input = torch.zeros(input.shape, device=input.device)

        e = torch.eye(dim, device=input.device)
        P_i = torch.matmul(S, torch.ones(dim, dim, device=input.device))

        P = (P_i - P_i.permute(0, 2, 1)) + e
        epo = torch.ones_like(P, device=input.device) * 0.000001
        P = torch.where(P != 0, P, epo)
        P = (1 / P) - e

        g1 = torch.matmul(U.permute(0, 2, 1), grad_U)
        g1 = (g1 + g1.permute(0, 2, 1)) / 2
        g1 = torch.mul(P.permute(0, 2, 1), g1)
        g1 = 2 * torch.matmul(torch.matmul(U, g1), U.permute(0, 2, 1))
        g2 = torch.matmul(torch.matmul(U, torch.mul(grad_S, e)), U.permute(0, 2, 1))
        grad_input = g1 + g2

        return grad_input


class EigLayer(nn.Module):
    def __init__(self):
        super(EigLayer, self).__init__()

    def forward(self, input1):
        return EigLayerF.apply(input1)