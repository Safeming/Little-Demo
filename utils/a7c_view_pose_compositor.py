from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from utils.a7c_overlap_set_compositor import dense_overlap_adjacency
from utils.a7c_quotient_compositor import runtime_target_mass
from utils.a7c_ray_context_probe import select_feature_group


def _finite_array(name: str, value, *, dtype=np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def axis_angle_pose_to_rotation_6d(pose) -> np.ndarray:
    values = _finite_array("pose", pose)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("pose must have shape [frames, joints, 3]")
    matrices = Rotation.from_rotvec(values.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(values.shape[:-1] + (3, 3))
    first_two_columns = np.swapaxes(matrices[..., :, :2], -1, -2)
    return first_two_columns.reshape(values.shape[0], -1)


def load_pose_rotation_6d(
    model_dir, frame_ids, joint_indices
) -> np.ndarray:
    root = Path(model_dir)
    joints = tuple(int(value) for value in joint_indices)
    if not joints:
        raise ValueError("joint_indices must not be empty")
    rows = []
    for frame in frame_ids:
        path = root / f"{int(frame):06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            if "pose_body" not in source:
                raise ValueError(f"{path} does not contain pose_body")
            body = _finite_array("pose_body", source["pose_body"]).reshape(-1, 3)
        if any(index < 0 or index >= body.shape[0] for index in joints):
            raise ValueError("joint index is out of range for pose_body")
        rows.append(body[list(joints)])
    return axis_angle_pose_to_rotation_6d(np.asarray(rows, dtype=np.float64))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_manifest_sha256(model_dir, frame_ids, repo_root) -> str:
    root = Path(repo_root).resolve()
    model_root = Path(model_dir).resolve()
    digest = hashlib.sha256()
    for frame in frame_ids:
        path = model_root / f"{int(frame):06d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("pose model path must be below repo_root") from error
        line = f"{_file_sha256(path)}  {relative}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def fit_normalization(values, fit_mask) -> dict[str, np.ndarray]:
    array = _finite_array("normalization values", values)
    mask = np.asarray(fit_mask, dtype=bool).reshape(-1)
    if array.ndim < 2 or array.shape[0] != mask.size or not np.any(mask):
        raise ValueError("normalization requires aligned nonempty fit rows")
    mean = array[mask].mean(axis=0)
    scale = array[mask].std(axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    return {"mean": mean, "scale": scale, "fit_mask": mask.copy()}


def apply_normalization(values, stats) -> np.ndarray:
    array = _finite_array("normalization values", values)
    mean = _finite_array("normalization mean", stats["mean"])
    scale = _finite_array("normalization scale", stats["scale"])
    if array.shape[1:] != mean.shape or mean.shape != scale.shape:
        raise ValueError("normalization statistics do not match values")
    if np.any(scale <= 0.0):
        raise ValueError("normalization scale must be positive")
    return (array - mean) / scale


def pack_camera_block_segments(
    camera_index, block_ids, frame_index, *, frame_stride
) -> list[np.ndarray]:
    cameras = np.asarray(camera_index).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    if cameras.shape != blocks.shape or cameras.shape != frames.shape:
        raise ValueError("camera, block, and frame manifests must align")
    if cameras.size == 0 or int(frame_stride) <= 0:
        raise ValueError("segment manifest and frame_stride must be nonempty")
    segments = []
    for camera, block in sorted(
        {(int(camera), int(block)) for camera, block in zip(cameras, blocks)}
    ):
        selected = np.flatnonzero((cameras == camera) & (blocks == block))
        order = selected[np.argsort(frames[selected], kind="stable")]
        ordered_frames = frames[order]
        if np.unique(ordered_frames).size != ordered_frames.size:
            raise ValueError("camera-block segment contains duplicate frames")
        if ordered_frames.size > 1 and not np.all(
            np.diff(ordered_frames) == int(frame_stride)
        ):
            raise ValueError("camera-block segment is not contiguous")
        segments.append(order.astype(np.int64, copy=False))
    return segments


def build_runtime_inputs(
    *,
    probe,
    feature_names,
    feature_group,
    pose_by_frame,
    camera_index,
    frame_index,
    carrier_ids,
    a5_weight,
    spatial_scale,
    depth_scale,
    edge_log_weight_minimum,
) -> dict:
    names = list(map(str, feature_names))
    requested = list(map(str, feature_group))
    forbidden = {
        "camera_id",
        "camera_index",
        "frame_id",
        "frame_index",
        "subject_id",
        "gaussian_id",
        "image_name",
        "held_block_identity",
    }
    leaked = sorted(forbidden.intersection(requested))
    if leaked:
        raise ValueError(f"forbidden runtime feature names: {leaked}")
    features = _finite_array("probe features", probe["features"], dtype=np.float32)
    cameras = np.asarray(camera_index).reshape(-1)
    frames = np.asarray(frame_index).reshape(-1)
    carriers = np.asarray(carrier_ids, dtype=np.int64).reshape(-1)
    if features.ndim != 3:
        raise ValueError("probe features must have shape [samples, carriers, fields]")
    if cameras.shape != (features.shape[0],) or frames.shape != cameras.shape:
        raise ValueError("runtime sample manifest differs from features")
    if not np.array_equal(np.asarray(probe["camera_index"]).reshape(-1), cameras):
        raise ValueError("probe camera manifest differs")
    if not np.array_equal(np.asarray(probe["frame_index"]).reshape(-1), frames):
        raise ValueError("probe frame manifest differs")
    if not np.array_equal(np.asarray(probe["carrier_ids"]).reshape(-1), carriers):
        raise ValueError("probe carrier manifest differs")
    if carriers.shape != (features.shape[1],):
        raise ValueError("carrier manifest differs from features")
    weights = _finite_array("A5 weight", a5_weight, dtype=np.float32).reshape(-1)
    if weights.shape != carriers.shape:
        raise ValueError("A5 weight must match carriers")

    selected_features = select_feature_group(features, names, requested).astype(
        np.float32, copy=False
    )
    required_fields = (
        "visibility",
        "camera_x_over_z",
        "camera_y_over_z",
        "log_depth",
        "alpha_transmittance_mass",
        "semantic_support_mean",
        "alpha_mean",
    )
    missing = [name for name in required_fields if name not in names]
    if missing:
        raise ValueError(f"missing runtime graph fields: {missing}")
    field = {
        name: features[:, :, names.index(name)].astype(np.float32, copy=False)
        for name in required_fields
    }
    visibility = field["visibility"]
    projected_xy = np.stack(
        (field["camera_x_over_z"], field["camera_y_over_z"]), axis=-1
    )
    adjacency = dense_overlap_adjacency(
        projected_xy=torch.from_numpy(projected_xy),
        log_depth=torch.from_numpy(field["log_depth"]),
        visibility=torch.from_numpy(visibility),
        spatial_scale=float(spatial_scale),
        depth_scale=float(depth_scale),
        edge_log_weight_minimum=float(edge_log_weight_minimum),
    ).numpy()
    mass = runtime_target_mass(
        alpha_transmittance_mass=torch.from_numpy(
            field["alpha_transmittance_mass"]
        ),
        a5_weight=torch.from_numpy(weights),
        semantic_support_mean=torch.from_numpy(field["semantic_support_mean"]),
        alpha_mean=torch.from_numpy(field["alpha_mean"]),
    ).numpy()
    pose_rows = []
    for frame in frames:
        key = int(frame)
        if key not in pose_by_frame:
            raise ValueError(f"pose is missing frame {key}")
        pose_rows.append(_finite_array("pose row", pose_by_frame[key]))
    pose = np.asarray(pose_rows, dtype=np.float32)
    if pose.shape != (features.shape[0], 36):
        raise ValueError("runtime pose must have shape [samples, 36]")
    return {
        "features": selected_features,
        "pose": pose,
        "projected_xy": projected_xy.astype(np.float32, copy=False),
        "log_depth": field["log_depth"],
        "visibility": visibility,
        "adjacency": adjacency.astype(np.float32, copy=False),
        "runtime_mass": mass.astype(np.float32, copy=False),
        "feature_names": np.asarray(requested),
    }


class ViewPoseResidualCompositor(nn.Module):
    def __init__(
        self,
        view_dimension: int,
        view_embedding_dimension: int,
        pose_dimension: int,
        pose_embedding_dimension: int,
        gru_hidden_dimension: int,
        residual_gate_scale: float,
        minimum_gate: float,
        maximum_gate: float,
    ) -> None:
        super().__init__()
        dimensions = (
            view_dimension,
            view_embedding_dimension,
            pose_dimension,
            pose_embedding_dimension,
            gru_hidden_dimension,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if int(view_embedding_dimension) != int(pose_embedding_dimension):
            raise ValueError("view and pose embedding dimensions must match")
        if not 0.0 <= float(minimum_gate) < float(maximum_gate) <= 1.0:
            raise ValueError("gate bounds are invalid")
        if float(residual_gate_scale) <= 0.0:
            raise ValueError("residual gate scale must be positive")
        embedding = int(view_embedding_dimension)
        self.view_encoder = nn.Sequential(
            nn.Linear(int(view_dimension), embedding), nn.SiLU()
        )
        self.graph_encoder = nn.Sequential(
            nn.Linear(4 * embedding, embedding), nn.SiLU()
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(int(pose_dimension), int(pose_embedding_dimension)),
            nn.SiLU(),
            nn.Linear(int(pose_embedding_dimension), embedding),
            nn.SiLU(),
        )
        self.temporal = nn.GRU(
            input_size=3 * embedding,
            hidden_size=int(gru_hidden_dimension),
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )
        self.residual_head = nn.Linear(2 * int(gru_hidden_dimension), 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.residual_gate_scale = float(residual_gate_scale)
        self.minimum_gate = float(minimum_gate)
        self.maximum_gate = float(maximum_gate)

    def forward(
        self,
        view: torch.Tensor,
        pose: torch.Tensor,
        adjacency: torch.Tensor,
        visibility: torch.Tensor,
        base_gates: torch.Tensor,
    ) -> torch.Tensor:
        if view.ndim != 3:
            raise ValueError("view must have shape [frames, carriers, fields]")
        frames, carriers = view.shape[:2]
        if pose.ndim != 2 or pose.shape[0] != frames:
            raise ValueError("pose must have shape [frames, fields]")
        if adjacency.shape != (frames, carriers, carriers):
            raise ValueError("adjacency shape differs from view")
        if visibility.shape != (frames, carriers):
            raise ValueError("visibility shape differs from view")
        if base_gates.shape != (frames, carriers):
            raise ValueError("base gate shape differs from view")
        for name, value in (
            ("view", view),
            ("pose", pose),
            ("adjacency", adjacency),
            ("visibility", visibility),
            ("base_gates", base_gates),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")

        node = self.view_encoder(view)
        message = torch.bmm(adjacency, node)
        visible = visibility.unsqueeze(-1)
        global_context = torch.sum(node * visible, dim=1, keepdim=True) / torch.clamp(
            torch.sum(visible, dim=1, keepdim=True), min=1.0
        )
        global_context = global_context.expand(-1, carriers, -1)
        graph = self.graph_encoder(
            torch.cat((node, message, node - message, global_context), dim=-1)
        )
        pose_embedding = self.pose_encoder(pose).unsqueeze(1).expand(
            -1, carriers, -1
        )
        interaction = graph * pose_embedding
        sequence_input = torch.cat(
            (graph, pose_embedding, interaction), dim=-1
        ).transpose(0, 1)
        sequence, _ = self.temporal(sequence_input)
        residual = self.residual_head(sequence).squeeze(-1).transpose(0, 1)
        return torch.clamp(
            base_gates + self.residual_gate_scale * torch.tanh(residual),
            min=self.minimum_gate,
            max=self.maximum_gate,
        )
