from collections import namedtuple

import torch


KNNResult = namedtuple("KNNResult", ["dists", "idx", "knn"])


def _is_oom_error(exc):
    message = str(exc).lower()
    return 'out of memory' in message or 'cuda error: out of memory' in message


def _compute_topk_chunk(query_chunk, reference, k, return_sorted):
    dists = torch.cdist(query_chunk, reference, p=2) ** 2
    return torch.topk(dists, k=k, dim=-1, largest=False, sorted=return_sorted)


def _compute_topk_chunk_cpu(query_chunk, reference, k, return_sorted):
    query_cpu = query_chunk.detach().cpu()
    reference_cpu = reference.detach().cpu()
    topk_dists, topk_idx = _compute_topk_chunk(query_cpu, reference_cpu, k, return_sorted)
    return topk_dists.to(query_chunk.device), topk_idx.to(query_chunk.device)


def _fallback_knn_points(p1, p2, K=1, return_sorted=True, return_nn=False, chunk_size=512, min_chunk_size=32, **kwargs):
    if p1.dim() != 3 or p2.dim() != 3:
        raise ValueError(f"knn_points expects [B, N, C] inputs, got {p1.shape} and {p2.shape}")

    if p1.shape[0] != p2.shape[0]:
        raise ValueError(f"Batch size mismatch in knn_points fallback: {p1.shape} vs {p2.shape}")

    if p1.shape[-1] != p2.shape[-1]:
        raise ValueError(f"Point dimension mismatch in knn_points fallback: {p1.shape} vs {p2.shape}")

    k = min(int(K), int(p2.shape[1]))
    if k <= 0:
        raise ValueError(f"K must be positive, got {K}")

    chunk_size = max(int(chunk_size), 1)
    min_chunk_size = max(min(int(min_chunk_size), chunk_size), 1)
    batch_dists = []
    batch_idx = []

    for b in range(p1.shape[0]):
        query = p1[b]
        reference = p2[b]
        chunk_dists = []
        chunk_idx = []
        start = 0
        adaptive_chunk_size = min(chunk_size, max(query.shape[0], 1))
        while start < query.shape[0]:
            current_chunk_size = min(adaptive_chunk_size, query.shape[0] - start)
            query_chunk = query[start:start + current_chunk_size]
            try:
                topk_dists, topk_idx = _compute_topk_chunk(query_chunk, reference, k, return_sorted)
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                if query_chunk.is_cuda:
                    torch.cuda.empty_cache()
                if current_chunk_size > min_chunk_size:
                    adaptive_chunk_size = max(current_chunk_size // 2, min_chunk_size)
                    continue
                topk_dists, topk_idx = _compute_topk_chunk_cpu(query_chunk, reference, k, return_sorted)
            chunk_dists.append(topk_dists)
            chunk_idx.append(topk_idx)
            start += current_chunk_size
        batch_dists.append(torch.cat(chunk_dists, dim=0))
        batch_idx.append(torch.cat(chunk_idx, dim=0))

    knn_dists = torch.stack(batch_dists, dim=0)
    knn_idx = torch.stack(batch_idx, dim=0)

    knn = None
    if return_nn:
        gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1])
        expanded_p2 = p2.unsqueeze(1).expand(-1, p1.shape[1], -1, -1)
        knn = torch.gather(expanded_p2, dim=2, index=gather_idx)

    return KNNResult(dists=knn_dists, idx=knn_idx, knn=knn)


try:
    from pytorch3d import ops as _pytorch3d_ops
except Exception:
    _pytorch3d_ops = None


def knn_points(*args, **kwargs):
    if _pytorch3d_ops is not None:
        return _pytorch3d_ops.knn_points(*args, **kwargs)
    return _fallback_knn_points(*args, **kwargs)


class _CompatOps:
    @staticmethod
    def knn_points(*args, **kwargs):
        return knn_points(*args, **kwargs)


ops = _CompatOps()
