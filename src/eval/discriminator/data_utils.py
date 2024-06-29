import torch

def collate_fn(data):
    batch_src, batch_dst, batch_x_n, batch_label = map(list, zip(*data))

    # (B + 1), B for batch size
    num_nodes_cumsum = torch.cumsum(torch.tensor(
        [0] + [len(x_n) for x_n in batch_x_n]
    ), dim=0)

    batch_size = len(batch_src)
    batch_src_ = []
    batch_dst_ = []
    for i in range(batch_size):
        batch_src_.append(batch_src[i] + num_nodes_cumsum[i])
        batch_dst_.append(batch_dst[i] + num_nodes_cumsum[i])

    batch_src = torch.cat(batch_src_, dim=0)               # (E)
    batch_dst = torch.cat(batch_dst_, dim=0)               # (E)
    batch_edge_index = torch.stack([batch_dst, batch_src]) # (2, E)

    # (V)
    batch_x_n = torch.cat(batch_x_n, dim=0).long()
    # (B, F_G)
    batch_label = torch.tensor(batch_label).unsqueeze(-1)

    return batch_edge_index, batch_x_n, num_nodes_cumsum, batch_label
