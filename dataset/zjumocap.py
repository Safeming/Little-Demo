import os
from pathlib import Path
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from utils.general_utils import get_body_model_misc_path
from scene.cameras import Camera
from utils.camera_utils import freeview_camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh
from PIL import Image

PARSER_BODY_LABELS = (13, 14, 15, 16, 17)
PARSER_CLOTH_LABELS = (5, 6, 7, 9, 10, 11, 12)
PARSER_VALID_LABELS = tuple(sorted(set(PARSER_BODY_LABELS + PARSER_CLOTH_LABELS)))

class ZJUMoCapDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.root_dir = cfg.root_dir
        self.refine = cfg.refine
        if self.refine:
            self.root_dir = "../../data/refined_ZJUMoCap_arah_format"

        self.subject = cfg.subject
        self.parsing_prior_cfg = cfg.get('parsing_prior', None)
        self.use_parsing_prior = bool(
            self.parsing_prior_cfg
            and (
                self.parsing_prior_cfg.get('enable', False)
                or self.parsing_prior_cfg.get('roi_enable', False)
            )
        )
        self.soft_mask_cfg = cfg.get('soft_mask', None)
        self.use_soft_mask = bool(self.soft_mask_cfg and self.soft_mask_cfg.get('enable', False))
        self.soft_mask_root = ''
        self.soft_mask_dirname = 'alpha'
        self.soft_mask_suffix = '.png'
        self.soft_mask_layout = 'cihp_subject'
        self.soft_mask_background_threshold = 1.0e-3
        self.soft_mask_fg_threshold = 0.5
        self.direct_parser_root = ''
        self.direct_parser_layout = 'cihp_subject'
        self.use_direct_parser_labels = False
        self.compact_mapping_file = ''
        self.compact_mapping = None
        self.skip_empty_samples = False
        self.skip_empty_min_pixels = 64
        if self.parsing_prior_cfg is not None:
            self.direct_parser_root = self.parsing_prior_cfg.get('parser_root', '')
            self.direct_parser_layout = self.parsing_prior_cfg.get('parser_layout', 'cihp_subject')
            self.use_direct_parser_labels = bool(self.parsing_prior_cfg.get('use_direct_parser_labels', False))
            self.compact_mapping_file = self.parsing_prior_cfg.get('compact_mapping_file', '')
            self.compact_mapping = self._load_compact_mapping(self.compact_mapping_file)
            self.skip_empty_samples = bool(self.parsing_prior_cfg.get('skip_empty_samples', False)) and split == 'train'
            self.skip_empty_min_pixels = int(self.parsing_prior_cfg.get('skip_empty_min_pixels', 64))
        if self.soft_mask_cfg is not None:
            self.soft_mask_root = self.soft_mask_cfg.get('root_dir', '')
            self.soft_mask_dirname = self.soft_mask_cfg.get('dirname', 'alpha')
            self.soft_mask_suffix = self.soft_mask_cfg.get('suffix', '.png')
            self.soft_mask_layout = self.soft_mask_cfg.get('layout', 'cihp_subject')
            self.soft_mask_background_threshold = float(self.soft_mask_cfg.get('background_threshold', 1.0e-3))
            self.soft_mask_fg_threshold = float(self.soft_mask_cfg.get('foreground_threshold', 0.5))
        self.train_frames = cfg.train_frames
        self.train_cams = cfg.train_views
        self.val_frames = cfg.val_frames
        self.val_cams = cfg.val_views
        self.white_bg = cfg.white_background
        crop_cfg = cfg.get('person_crop', None)
        self.person_crop_enable = bool(crop_cfg and crop_cfg.get('enable', False))
        self.person_crop_padding_ratio = float(crop_cfg.get('padding_ratio', 0.15)) if crop_cfg is not None else 0.15
        self.person_crop_square = bool(crop_cfg.get('square', True)) if crop_cfg is not None else True
        self.person_crop_min_size = int(crop_cfg.get('min_size', 256)) if crop_cfg is not None else 256
        self.H, self.W = 1024, 1024 # hardcoded original size
        self.h, self.w = cfg.img_hw

        self.faces = np.load(get_body_model_misc_path('faces.npz'))['faces']
        self.skinning_weights = dict(np.load(get_body_model_misc_path('skinning_weights_all.npz')))
        self.posedirs = dict(np.load(get_body_model_misc_path('posedirs_all.npz')))
        self.J_regressor = dict(np.load(get_body_model_misc_path('J_regressors.npz')))

        if split == 'train':
            cam_names = self.train_cams
            frames = self.train_frames
        elif split == 'val':
            cam_names = self.val_cams
            frames = self.val_frames
        elif split == 'test':
            cam_names = self.cfg.test_views[self.cfg.test_mode]
            frames = self.cfg.test_frames[self.cfg.test_mode]
        elif split == 'predict':
            cam_names = self.cfg.predict_views
            frames = self.cfg.predict_frames
        else:
            raise ValueError

        with open(os.path.join(self.root_dir, self.subject, 'cam_params.json'), 'r') as f:
            self.cameras = json.load(f)

        if len(cam_names) == 0:
            cam_names = self.cameras['all_cam_names']
        elif self.refine:
            cam_names = [f'{int(cam_name) - 1:02d}' for cam_name in cam_names]
        else:
            cam_names = [str(cam_name) for cam_name in cam_names]

        start_frame, end_frame, sampling_rate = frames

        subject_dir = os.path.join(self.root_dir, self.subject)
        if split == 'predict':
            predict_seqs = ['gBR_sBM_cAll_d04_mBR1_ch05_view1',
                            'gBR_sBM_cAll_d04_mBR1_ch06_view1',
                            'MPI_Limits-03099-op8_poses_view1',
                            'canonical_pose_view1',]
            predict_seq = self.cfg.get('predict_seq', 0)
            predict_seq = predict_seqs[predict_seq]
            model_files = sorted(glob.glob(os.path.join(subject_dir, predict_seq, '*.npz')))
            self.model_files = model_files
            frames = list(reversed(range(-len(model_files), 0)))
            if end_frame == 0:
                end_frame = len(model_files)
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            model_files = model_files[frame_slice]
            frames = frames[frame_slice]
        else:
            if self.cfg.get('arah_opt', False):
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'opt_models/*.npz')))
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))
            self.model_files = model_files
            frames = list(range(len(model_files)))
            if end_frame == 0:
                end_frame = len(model_files)
            frame_slice = slice(start_frame, end_frame, sampling_rate)
            model_files = model_files[frame_slice]
            frames = frames[frame_slice]

        # add freeview rendering
        if cfg.freeview:
            # with open(os.path.join(self.root_dir, self.subject, 'freeview_cam_params.json'), 'r') as f:
            #     self.cameras = json.load(f)
            model_dict = np.load(model_files[0])
            trans = model_dict['trans'].astype(np.float32)
            self.cameras = freeview_camera(self.cameras[cam_names[0]], trans)
            cam_names = self.cameras['all_cam_names']

        self.data = []
        if split == 'predict' or cfg.freeview:
            for cam_idx, cam_name in enumerate(cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)

                for d_idx, f_idx in enumerate(frames):
                    model_file = model_files[d_idx]
                    # get dummy gt...
                    # img_file = glob.glob(os.path.join(cam_dir, '*.jpg'))[0]
                    img_file = os.path.join(subject_dir, '1', '000000.jpg')
                    # mask_file = glob.glob(os.path.join(cam_dir, '*.png'))[0]
                    mask_file = os.path.join(subject_dir, '1', '000000.png')

                    self.data.append({
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'data_idx': d_idx,
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                    })
        else:
            for cam_idx, cam_name in enumerate(cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)
                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[frame_slice]
                mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[frame_slice]

                for d_idx, f_idx in enumerate(frames):
                    img_file = img_files[d_idx]
                    mask_file = mask_files[d_idx]
                    model_file = model_files[d_idx]

                    self.data.append({
                        'cam_idx': cam_idx,
                        'cam_name': cam_name,
                        'data_idx': d_idx,
                        'frame_idx': f_idx,
                        'img_file': img_file,
                        'mask_file': mask_file,
                        'model_file': model_file,
                    })

        if self.skip_empty_samples:
            original_count = len(self.data)
            filtered = [item for item in self.data if self._sample_has_nonempty_parsing(item['cam_name'], item['frame_idx'])]
            if len(filtered) == 0:
                raise RuntimeError('All training samples were filtered out by parsing_prior.skip_empty_samples; check compact_mapping_file / parser labels.')
            self.data = filtered
            print(f'[ZJUMoCapDataset] parsing-based sample filter kept {len(self.data)}/{original_count} samples (min_pixels={self.skip_empty_min_pixels}).')

        self.frames = frames
        self.model_files_list = model_files

        self.get_metadata()

        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]


    def get_metadata(self):
        data_paths = self.model_files
        data_path = data_paths[0]

        cano_data = self.get_cano_smpl_verts(data_path)
        if self.split != 'train':
            self.metadata = cano_data
            return

        start, end, step = self.train_frames
        frames = list(range(len(data_paths)))
        if end == 0:
            end = len(frames)
        frame_slice = slice(start, end, step)
        frames = frames[frame_slice]

        frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        }
        self.metadata.update(cano_data)
        if self.cfg.train_smpl:
            self.metadata.update(self.get_smpl_data())

    def _get_parsing_prior_paths(self, cam_name, frame_idx):
        if not self.use_parsing_prior:
            return None, None, None, None
        root_dir = self.parsing_prior_cfg.get('root_dir', '')
        if not root_dir:
            return None, None, None, None

        suffix = self.parsing_prior_cfg.get('mask_suffix', '.png')
        frame_name = f'{int(frame_idx):06d}{suffix}'
        cam_name = str(cam_name)
        base_dir = os.path.join(root_dir, self.subject, cam_name)
        body_path = os.path.join(base_dir, self.parsing_prior_cfg.get('body_dirname', 'body'), frame_name)
        cloth_path = os.path.join(base_dir, self.parsing_prior_cfg.get('cloth_dirname', 'cloth'), frame_name)
        valid_path = os.path.join(base_dir, self.parsing_prior_cfg.get('valid_dirname', 'valid'), frame_name)
        uncertain_path = os.path.join(base_dir, self.parsing_prior_cfg.get('uncertain_dirname', 'uncertain'), frame_name)
        if not os.path.exists(body_path):
            body_path = None
        if not os.path.exists(cloth_path):
            cloth_path = None
        if not os.path.exists(valid_path):
            valid_path = None
        if not os.path.exists(uncertain_path):
            uncertain_path = None
        return body_path, cloth_path, valid_path, uncertain_path

    def _crop_array(self, array, crop_box):
        if crop_box is None:
            return array
        x0, y0, x1, y1 = crop_box
        return array[y0:y1, x0:x1]

    def _compute_person_crop_box(self, mask):
        mask_bool = mask > 0
        ys, xs = np.where(mask_bool)
        if xs.size == 0 or ys.size == 0:
            return None

        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        width = x1 - x0
        height = y1 - y0

        if self.person_crop_square:
            side = int(np.ceil(max(width, height) * (1.0 + self.person_crop_padding_ratio)))
            side = max(side, self.person_crop_min_size)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            x0 = int(np.floor(cx - side * 0.5))
            y0 = int(np.floor(cy - side * 0.5))
            x1 = x0 + side
            y1 = y0 + side
        else:
            pad_x = int(np.ceil(width * self.person_crop_padding_ratio))
            pad_y = int(np.ceil(height * self.person_crop_padding_ratio))
            x0 -= pad_x
            x1 += pad_x
            y0 -= pad_y
            y1 += pad_y
            crop_w = max(x1 - x0, self.person_crop_min_size)
            crop_h = max(y1 - y0, self.person_crop_min_size)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            x0 = int(np.floor(cx - crop_w * 0.5))
            y0 = int(np.floor(cy - crop_h * 0.5))
            x1 = x0 + crop_w
            y1 = y0 + crop_h

        if x0 < 0:
            x1 -= x0
            x0 = 0
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if x1 > self.W:
            x0 -= (x1 - self.W)
            x1 = self.W
        if y1 > self.H:
            y0 -= (y1 - self.H)
            y1 = self.H

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.W, x1)
        y1 = min(self.H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _load_parsing_prior_mask(self, path, K, dist, crop_box=None):
        if path is None:
            return None
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask = cv2.undistort(mask, K, dist, None)
        mask = self._crop_array(mask, crop_box)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def _resolve_direct_parser_mask_path(self, cam_name, frame_idx):
        if not self.direct_parser_root:
            return None
        root = Path(self.direct_parser_root)
        frame_name = f'{int(frame_idx):06d}.png'
        if self.direct_parser_layout == 'cihp_subject':
            return root / self.subject / 'mask_cihp' / f'Camera_B{int(cam_name)}' / frame_name
        if self.direct_parser_layout == 'flat_png':
            return root / frame_name
        raise ValueError(f'Unsupported parser layout: {self.direct_parser_layout}')

    def _load_direct_parser_index_mask(self, cam_name, frame_idx, K, dist, crop_box=None):
        path = self._resolve_direct_parser_mask_path(cam_name, frame_idx)
        if path is None or not path.exists():
            return None
        with Image.open(path) as img:
            mask = np.array(img)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = cv2.undistort(mask, K, dist, None)
        mask = self._crop_array(mask, crop_box)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.int32)

    def _resolve_soft_mask_path(self, cam_name, frame_idx):
        if not self.use_soft_mask or not self.soft_mask_root:
            return None
        root = Path(self.soft_mask_root)
        frame_name = f'{frame_idx:06d}{self.soft_mask_suffix}'
        if self.soft_mask_layout == 'cihp_subject':
            return root / self.subject / self.soft_mask_dirname / f'Camera_B{int(cam_name)}' / frame_name
        return root / self.subject / f'Camera_B{int(cam_name)}' / frame_name

    def _load_soft_mask(self, cam_name, frame_idx, K, dist, crop_box=None):
        path = self._resolve_soft_mask_path(cam_name, frame_idx)
        if path is None or not path.exists():
            return None
        mask = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask = cv2.undistort(mask, K, dist, None)
        mask = self._crop_array(mask, crop_box)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


    def _load_compact_mapping(self, path):
        if not path:
            return None
        mapping_path = Path(path)
        if not mapping_path.exists():
            print(f'[ZJUMoCapDataset] compact mapping file not found: {mapping_path}')
            return None
        with open(mapping_path, 'r') as f:
            data = json.load(f)
        raw_groups = data.get('groups', {})
        groups = {}
        active_labels = set()
        for name, labels in raw_groups.items():
            label_ids = tuple(sorted({int(label) for label in labels}))
            if not label_ids:
                continue
            groups[name] = label_ids
            active_labels.update(label_ids)
        if not groups:
            return None
        return {
            'path': str(mapping_path),
            'groups': groups,
            'active_labels': tuple(sorted(active_labels)),
            'ignore_labels': tuple(sorted({int(label) for label in data.get('ignore_labels', [])})),
            'class_names': tuple(data.get('class_names', tuple(groups.keys()))),
        }

    def _camera_intrinsics_distortion(self, cam_name):
        cam = self.cameras[str(cam_name)]
        K = np.array(cam['K'], dtype=np.float32)
        dist = np.array(cam['D'], dtype=np.float32)
        return K, dist

    def _sample_has_nonempty_parsing(self, cam_name, frame_idx):
        if not self.use_parsing_prior:
            return True
        min_pixels = max(int(self.skip_empty_min_pixels), 1)
        active_labels = self.compact_mapping['active_labels'] if self.compact_mapping is not None else PARSER_VALID_LABELS
        if self.use_direct_parser_labels and self.direct_parser_root:
            K, dist = self._camera_intrinsics_distortion(cam_name)
            parser_index_mask = self._load_direct_parser_index_mask(cam_name, frame_idx, K, dist, crop_box=None)
            if parser_index_mask is None:
                return False
            return int(np.isin(parser_index_mask, np.asarray(active_labels)).sum()) >= min_pixels

        body_path, cloth_path, valid_path, _ = self._get_parsing_prior_paths(cam_name, frame_idx)
        if valid_path is not None:
            valid_mask = cv2.imread(valid_path, cv2.IMREAD_GRAYSCALE)
            if valid_mask is not None:
                return int((valid_mask > 0).sum()) >= min_pixels
        total = 0
        for p in (body_path, cloth_path):
            if p is None:
                continue
            mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                total += int((mask > 0).sum())
        return total >= min_pixels


    def get_cano_smpl_verts(self, data_path):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''
        # compute scale from SMPL body
        model_dict = np.load(data_path)
        gender = 'neutral'

        # 3D models and points
        minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            'gender': gender,
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v,
            'cano_mesh': cano_mesh,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def get_smpl_data(self):
        # load all smpl fitting of the training sequence
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_files_list)):
            model_dict = np.load(model_file)

            if idx == 0:
                smpl_data['betas'] = model_dict['betas'].astype(np.float32)

            smpl_data['frames'].append(frame)
            smpl_data['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            smpl_data['pose_body'].append(model_dict['pose_body'].astype(np.float32))
            smpl_data['pose_hand'].append(model_dict['pose_hand'].astype(np.float32))
            smpl_data['trans'].append(model_dict['trans'].astype(np.float32))

        return smpl_data

    def __len__(self):
        return len(self.data)

    def getitem(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]
        cam_idx = data_dict['cam_idx']
        cam_name = data_dict['cam_name']
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']
        parsing_body_file, parsing_cloth_file, parsing_valid_file, parsing_uncertain_file = self._get_parsing_prior_paths(cam_name, frame_idx)

        K = np.array(self.cameras[cam_name]['K'], dtype=np.float32).copy()
        dist = np.array(self.cameras[cam_name]['D'], dtype=np.float32).ravel()
        R = np.array(self.cameras[cam_name]['R'], np.float32)
        T = np.array(self.cameras[cam_name]['T'], np.float32)

        # note that in ZJUMoCap the camera center does not align perfectly
        # here we try to offset it by modifying the extrinsic...
        M = np.eye(3)
        M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        K[0, 2] = self.W / 2
        K[1, 2] = self.H / 2
        R = M @ R
        T = M @ T

        R = np.transpose(R)
        T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

        if self.refine:
            hard_mask = cv2.imread(mask_file)
            hard_mask = hard_mask.sum(-1)
            hard_mask[hard_mask != 0] = 100
            hard_mask = hard_mask.astype(np.uint8)
        else:
            hard_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        K_undistort = K.copy()
        image = cv2.undistort(image, K_undistort, dist, None)
        hard_mask = cv2.undistort(hard_mask, K_undistort, dist, None)

        crop_box = None
        crop_width = self.W
        crop_height = self.H
        if self.person_crop_enable:
            crop_box = self._compute_person_crop_box(hard_mask)
            if crop_box is not None:
                image = self._crop_array(image, crop_box)
                hard_mask = self._crop_array(hard_mask, crop_box)
                x0, y0, x1, y1 = crop_box
                K[0, 2] -= x0
                K[1, 2] -= y0
                crop_width = x1 - x0
                crop_height = y1 - y0

        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        hard_mask = cv2.resize(hard_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        hard_mask = (hard_mask != 0).astype(np.float32)
        soft_mask = self._load_soft_mask(cam_name, frame_idx, K_undistort, dist, crop_box=crop_box)
        if soft_mask is None:
            mask = hard_mask
            bg_mask = mask <= 0.5
        else:
            mask = soft_mask.astype(np.float32)
            bg_mask = mask <= self.soft_mask_background_threshold

        image[bg_mask] = 255. if self.white_bg else 0.
        image = image / 255.

        parsing_body_mask = None
        parsing_cloth_mask = None
        parsing_valid_mask = None
        parsing_uncertain_mask = None
        parsing_compact_masks = None
        parsing_compact_class_names = None
        parsing_parser_mask = None
        if self.use_parsing_prior:
            fg = (mask > self.soft_mask_fg_threshold).astype(np.float32)
            parser_index_mask = None
            uncertain_prior = self._load_parsing_prior_mask(parsing_uncertain_file, K_undistort, dist, crop_box=crop_box)
            if self.use_direct_parser_labels:
                parser_index_mask = self._load_direct_parser_index_mask(cam_name, frame_idx, K_undistort, dist, crop_box=crop_box)
            if parser_index_mask is not None:
                body_prior = np.isin(parser_index_mask, np.asarray(PARSER_BODY_LABELS)).astype(np.float32) * fg
                cloth_prior = np.isin(parser_index_mask, np.asarray(PARSER_CLOTH_LABELS)).astype(np.float32) * fg
                valid_labels = (
                    self.compact_mapping['active_labels']
                    if self.compact_mapping is not None
                    else PARSER_VALID_LABELS
                )
                parsing_valid = np.isin(parser_index_mask, np.asarray(valid_labels)).astype(np.float32) * fg
                parsing_parser_mask = torch.from_numpy(parser_index_mask.astype(np.float32)).unsqueeze(0).float()
                if self.compact_mapping is not None:
                    compact_masks = []
                    compact_names = []
                    for class_name in self.compact_mapping['class_names']:
                        label_ids = self.compact_mapping['groups'].get(class_name, ())
                        if len(label_ids) == 0:
                            continue
                        compact_mask = np.isin(parser_index_mask, np.asarray(label_ids)).astype(np.float32) * fg
                        compact_masks.append(compact_mask)
                        compact_names.append(class_name)
                    if compact_masks:
                        parsing_compact_masks = torch.from_numpy(np.stack(compact_masks, axis=0)).float()
                        parsing_compact_class_names = tuple(compact_names)
            else:
                body_prior = self._load_parsing_prior_mask(parsing_body_file, K_undistort, dist, crop_box=crop_box)
                cloth_prior = self._load_parsing_prior_mask(parsing_cloth_file, K_undistort, dist, crop_box=crop_box)
                valid_prior = self._load_parsing_prior_mask(parsing_valid_file, K_undistort, dist, crop_box=crop_box)
                if body_prior is not None or cloth_prior is not None or valid_prior is not None:
                    if body_prior is None:
                        body_prior = np.zeros((self.h, self.w), dtype=np.float32)
                    if cloth_prior is None:
                        cloth_prior = np.zeros((self.h, self.w), dtype=np.float32)
                    body_prior = np.clip(body_prior, 0.0, 1.0) * fg
                    cloth_prior = np.clip(cloth_prior, 0.0, 1.0) * fg
                    if valid_prior is None:
                        parsing_valid = np.clip(body_prior + cloth_prior, 0.0, 1.0)
                    else:
                        parsing_valid = np.clip(valid_prior, 0.0, 1.0) * fg
                else:
                    body_prior = None
                    cloth_prior = None
                    parsing_valid = None
            if uncertain_prior is not None:
                uncertain_prior = np.clip(uncertain_prior, 0.0, 1.0) * fg
            if body_prior is not None and cloth_prior is not None and parsing_valid is not None:
                if uncertain_prior is not None:
                    body_prior = body_prior * (1.0 - uncertain_prior)
                    cloth_prior = cloth_prior * (1.0 - uncertain_prior)
                    parsing_valid = parsing_valid * (1.0 - uncertain_prior)
                    parsing_uncertain_mask = torch.from_numpy(uncertain_prior).unsqueeze(0).float()
                    if parsing_compact_masks is not None:
                        parsing_compact_masks = parsing_compact_masks * (1.0 - parsing_uncertain_mask)
                body_prior = body_prior * parsing_valid
                cloth_prior = cloth_prior * parsing_valid
                parsing_body_mask = torch.from_numpy(body_prior).unsqueeze(0).float()
                parsing_cloth_mask = torch.from_numpy(cloth_prior).unsqueeze(0).float()
                parsing_valid_mask = torch.from_numpy(parsing_valid).unsqueeze(0).float()
                if parsing_compact_masks is not None:
                    parsing_compact_masks = parsing_compact_masks * parsing_valid_mask

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        hard_mask = torch.from_numpy(hard_mask).unsqueeze(0).float()
        soft_mask_tensor = None if soft_mask is None else torch.from_numpy(soft_mask.astype(np.float32)).unsqueeze(0).float()

        # update camera parameters
        K[0, :] *= self.w / crop_width
        K[1, :] *= self.h / crop_height

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # Compute posed SMPL body
        minimal_shape = self.metadata['minimal_shape']
        gender = self.metadata['gender']

        model_dict = np.load(model_file)
        n_smpl_points = minimal_shape.shape[0]
        trans = model_dict['trans'].astype(np.float32)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose_body = model_dict['pose_body'].astype(np.float32)
        pose_hand = model_dict['pose_hand'].astype(np.float32)
        # Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)
        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])  # 24 x 9, root rotation is set to identity
        pose_rot_full = pose_mat_full.reshape([-1, 9])  # 24 x 9, including root rotation

        # Minimally clothed shape
        posedir = self.posedirs[gender]
        Jtr = self.metadata['Jtr']

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)
        minimal_shape_centered = minimal_shape - center
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v = self.metadata['bone_transforms_02v']
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)
        bone_transforms = bone_transforms.astype(np.float32)
        bone_transforms[:, :3, 3] += trans  # add global offset
        posed_joints = np.matmul(bone_transforms[:, :3, :3], Jtr[..., np.newaxis]).squeeze(-1) + bone_transforms[:, :3, 3]
        posed_joints = posed_joints.astype(np.float32)

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            hard_mask=hard_mask,
            soft_mask=soft_mask_tensor,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            parsing_body_mask=parsing_body_mask,
            parsing_cloth_mask=parsing_cloth_mask,
            parsing_valid_mask=parsing_valid_mask,
            parsing_uncertain_mask=parsing_uncertain_mask,
            parsing_compact_masks=parsing_compact_masks,
            parsing_compact_class_names=parsing_compact_class_names,
            parsing_parser_mask=parsing_parser_mask,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
            posed_joints=torch.from_numpy(posed_joints).float(),
        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    def readPointCloud(self,):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            ply_path = os.path.join(self.root_dir, self.subject, 'cano_smpl.ply')
            try:
                pcd = fetchPly(ply_path)
            except:
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd
