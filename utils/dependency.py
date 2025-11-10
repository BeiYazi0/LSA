import torch
import torch.nn.functional as F


def similar_imp(weight):  # one by one or match
    Q = F.normalize(weight, p=2, dim=1)
    # A = F.softmax(Q @ Q.T, dim=1)
    A = 1 - torch.abs(Q @ Q.T)
    score = torch.sum(A, dim=1)
    return score


def norm_imp(weight):
    co = weight.shape[0]
    norm_nuclear = torch.linalg.norm(weight, ord='nuc')
    score = torch.zeros((co,), device=weight.device)

    mask = torch.ones((co, 1), device=weight.device)
    for i in range(co):
        mask[i] = 0.
        score[i] = norm_nuclear - torch.linalg.norm(weight * mask, ord='nuc')
        mask[i] = 1.
    return score


def similar_imp2(weight):  # distance
    # weight = torch.abs(weight)
    mean_weight = torch.mean(weight, dim=0)
    score = torch.sum((weight - mean_weight) ** 2, dim=1)
    return score


def similar_imp3(weight):  # distance
    n = weight.shape[0]
    res = torch.sum(weight ** 2)
    gm = 0
    for i in range(n):
        mean_weight = weight[i].reshape(1, -1)
        score = torch.sum((weight - mean_weight) ** 2)
        if score < res:
            gm = mean_weight
            res = score
    score = torch.sum((weight - gm) ** 2, dim=1)
    return score


def err_imp(weight, inputs, p):  # distance
    sa = weight.T @ weight
    sb = inputs @ inputs.T
    score = sa * sb
    if p == score.shape[0]:
        score = torch.diag(score)
    else:
        s = score.shape[0] // p
        new_score = torch.diag(score)
        for i in range(p):
            start = i * s
            end = (i + 1) * s
            new_score[start:end] = torch.sum(score[start:end, start:end])
        score = new_score / s
    return score


class EmbedNode:
    def __init__(self, module):
        self.module = module
        self.channels = self.module.weight.shape[1]

    def cal_importance(self):
        score = torch.abs(self.module.weight.grad * self.module.weight)
        score = torch.sum(score, dim=0)
        return score

    def prune(self, keep_idxs):
        self.channels = len(keep_idxs)
        self.module.weight = torch.nn.Parameter(self.module.weight.data[:, keep_idxs])


class NormNode:
    def __init__(self, module):
        self.module = module
        self.channels = self.module.weight.shape[0]

    def cal_importance(self):
        score = torch.abs(self.module.weight.grad * self.module.weight)
        return score

    def prune(self, keep_idxs):
        self.channels = len(keep_idxs)
        self.module.weight = torch.nn.Parameter(self.module.weight.data[keep_idxs])


class ShareHeadNode:
    def __init__(self, head_num, nodes):
        self.n_head = head_num
        self.q_node, self.k_node, self.v_node, self.o_node = nodes

        self.q_dim = self.q_node.channels // head_num
        self.v_dim = self.v_node.channels // head_num

        self.num_key_value_groups = self.q_dim // self.v_dim

        self.num_key_value_heads = self.n_head // self.num_key_value_groups
        self.o_channels = self.num_key_value_heads

    def cal_importance(self):
        q_score = self.q_node.cal_importance().reshape(self.num_key_value_heads, self.num_key_value_groups, self.q_dim)

        k_score = self.k_node.cal_importance().reshape(self.num_key_value_heads, self.q_dim)

        v_score = self.v_node.cal_importance().reshape(self.num_key_value_heads, self.q_dim)

        o_score = self.o_node.cal_importance().reshape(self.num_key_value_heads,
                                                                               self.num_key_value_groups, self.q_dim)

        score = torch.sum(q_score, dim=[1, 2]) + torch.sum(k_score, dim=1) \
                + torch.sum(v_score, dim=1) + torch.sum(o_score, dim=[1, 2])
        return score

    def prune(self, idxs):
        t = self.num_key_value_groups * self.q_dim
        q_idxs = torch.tensor([p for i in idxs for p in range(i * t, (i + 1) * t)], device=idxs.device)
        self.q_node.prune(q_idxs)
        self.o_node.prune(q_idxs)

        k_idxs = torch.unique(q_idxs // self.num_key_value_groups)
        self.k_node.prune(k_idxs)
        self.v_node.prune(k_idxs)

        self.num_key_value_heads = len(idxs)
        self.n_head = self.num_key_value_heads * self.num_key_value_groups


class ChannelHeadNode:
    def __init__(self, head_num, nodes):
        self.n_head = head_num
        self.q_node, self.k_node, self.v_node, self.o_node = nodes

        self.q_dim = self.q_node.channels // head_num
        self.v_dim = self.v_node.channels // head_num
        self.o_channels = self.v_dim // 2

        self.num_key_value_groups = self.q_dim // self.v_dim

        self.num_key_value_heads = self.n_head // self.num_key_value_groups

    def cal_importance(self):
        q_score = self.q_node.cal_importance().reshape(self.n_head, self.num_key_value_groups, self.v_dim // 2, 2)

        k_score = self.k_node.cal_importance().reshape(self.n_head, self.v_dim // 2, 2)

        v_score = self.v_node.cal_importance().reshape(self.n_head, self.v_dim // 2, 2)

        o_score = self.o_node.cal_importance().reshape(self.n_head, self.num_key_value_groups, self.v_dim // 2, 2)

        score = torch.sum(q_score, dim=[0, 1, 3]) + torch.sum(k_score, dim=[0, 2]) \
                + torch.sum(v_score, dim=[0, 2]) + torch.sum(o_score, dim=[0, 1, 3])
        return score

    def prune(self, idxs):
        idxs = torch.sort(torch.concat([2 * idxs, 2 * idxs + 1], dim=0))[0]

        t = torch.arange(0, self.n_head * self.v_dim, self.v_dim, dtype=torch.int64, device=idxs.device)
        k_idxs = [i + idxs for i in t]
        k_idxs = torch.concat(k_idxs, dim=0)
        self.k_node.prune(k_idxs)
        self.v_node.prune(k_idxs)

        t = self.num_key_value_groups
        q_idxs = torch.tensor([p for i in k_idxs for p in range(i * t, (i + 1) * t)], device=idxs.device)
        self.q_node.prune(q_idxs)
        self.o_node.prune(q_idxs)

        self.v_dim = len(idxs)
        # self.num_key_value_heads = len(idxs)
        # self.n_head = self.num_key_value_heads * self.num_key_value_groups


class LinearNode:
    def __init__(self, module, ntype, itype):
        self.module = module
        self.ntype = ntype
        self.itype = itype
        if ntype == "in":
            self.channels = self.module.weight.shape[1]
        else:
            self.channels = self.module.weight.shape[0]
        self.o_channels = self.channels

    def cal_importance(self, p=0):
        if self.itype == "err":
            if self.ntype == "in":
                score = err_imp(self.module.weight, self.module.input, self.o_channels if p == 0 else p)
            else:
                score = torch.zeros((self.module.weight.shape[0],), device=self.module.weight.device)
        else:
            score = torch.abs(self.module.weight.grad * self.module.weight)
            if self.ntype == "in":
                score = torch.sum(score, dim=0)
            else:
                score = torch.sum(score, dim=1)

        return score

    def prune(self, keep_idxs):
        keep_idxs = keep_idxs.to(self.module.weight.device)
        self.channels = len(keep_idxs)

        if self.ntype == "in":
            self.module.weight = torch.nn.Parameter(self.module.weight.data[:, keep_idxs])
        else:
            self.module.weight = torch.nn.Parameter(self.module.weight.data[keep_idxs])


class Group:
    def __init__(self, name, nodes, mode=None):
        self.name = name
        self.nodes = nodes
        self.mode = mode
        self.o_channels = self.nodes[0].o_channels

    def cal_importance(self):  # itype
        score = 0
        for node in self.nodes:
            score += node.cal_importance()
        return score

    def prune(self, idxs):
        for node in self.nodes:
            node.prune(idxs)


def create_llama_groups(model, n_heads=32, start=3, end=31, mode="none", itype="none"):
    groups = []
    for i, decoder in enumerate(model.model.layers):
        if i < start or i > end:
            continue
        q_node = LinearNode(decoder.self_attn.q_proj, "out", itype=itype)
        k_node = LinearNode(decoder.self_attn.k_proj, "out", itype=itype)
        v_node = LinearNode(decoder.self_attn.v_proj, "out", itype=itype)
        o_node = LinearNode(decoder.self_attn.o_proj, "in", itype=itype)
        if mode == "channel":
            groups.append(Group(f"head_{i}", [ChannelHeadNode(n_heads, [q_node, k_node, v_node, o_node])], mode=mode))
        else:
            groups.append(Group(f"head_{i}", [ShareHeadNode(n_heads, [q_node, k_node, v_node, o_node])], mode=mode))

        g_node = LinearNode(decoder.mlp.gate_proj, "out", itype=itype)
        u_node = LinearNode(decoder.mlp.up_proj, "out", itype=itype)
        d_node = LinearNode(decoder.mlp.down_proj, "in", itype=itype)
        groups.append(Group(f"mlp_{i}", [g_node, u_node, d_node], mode=mode))

    return groups
