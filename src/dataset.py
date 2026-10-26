from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info

from .features import build_section_features, make_triplet
from .transforms import augment_image_label

# The physical split math lives in src.splitting. Names used only by callers
# (notebooks, tests) are re-exported with noqa so `from src.dataset import ...`
# keeps working.
from .splitting import (  # noqa: F401
    AXIS_INDEX,
    count_labels_in_split,
    load_or_create_split,
    make_spatial_split,
    range_for_index,
    save_split,
    split_class_summary,
    split_volume,
    subset_ranges,
)


class SeismicSliceDataset(Dataset):
    def __init__(
        self,
        seismic: np.ndarray,
        labels_1_6: np.ndarray,
        split: dict,
        normalization: dict,
        *,
        mode: str,
        orientations: list[str],
        patch_size: int,
        random_crop: bool,
        feature_mode: str = "neighbor_triplet",
        crop_strategy: str | None = None,
        augmentation: dict | None = None,
        samples_per_epoch: int | None = None,
        random_sample_when_limited: bool = True,
        require_all_orientations: bool = False,
        balance_orientations: bool = False,
        class_aware_sampling: bool = False,
        class_aware_fraction: float = 0.5,
        global_context_fraction: float = 0.0,
        seed: int = 42,
    ):
        if seismic.shape != labels_1_6.shape:
            raise ValueError(f"Seismic and label shape mismatch: {seismic.shape} vs {labels_1_6.shape}")
        self.seismic = seismic
        self.labels = labels_1_6
        self.split = split
        self.normalization = normalization
        self.mode = mode
        self.orientations = orientations
        self.patch_size = int(patch_size)
        self.random_crop = bool(random_crop and mode == "train")
        self.crop_strategy = crop_strategy or ("random_crop" if self.random_crop else "center_crop")
        if self.mode != "train" and self.crop_strategy == "random_crop":
            self.crop_strategy = "center_crop"
        self.feature_mode = feature_mode
        self.augmentation = augmentation or {}
        self.samples_per_epoch = samples_per_epoch
        self.random_sample_when_limited = bool(random_sample_when_limited)
        self.balance_orientations = bool(balance_orientations)
        self.class_aware_sampling = bool(class_aware_sampling and mode == "train")
        self.class_aware_fraction = float(class_aware_fraction)
        if not 0.0 <= self.class_aware_fraction <= 1.0:
            raise ValueError("class_aware_fraction must be between 0 and 1")
        self.global_context_fraction = float(global_context_fraction) if mode == "train" else 0.0
        if not 0.0 <= self.global_context_fraction <= 1.0:
            raise ValueError("global_context_fraction must be between 0 and 1")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._rng_worker_id: int | None = None
        self.indices = self._build_indices()
        self.indices_by_orientation = {
            orientation: [entry for entry in self.indices if entry[0] == orientation]
            for orientation in self.orientations
        }
        self.indices_by_orientation = {
            orientation: entries for orientation, entries in self.indices_by_orientation.items() if entries
        }
        if self.global_context_fraction and not self.indices_by_orientation.get("inline"):
            raise ValueError("Global context training requires eligible inline sections")
        self._class_indices: dict[int, dict[str, list[tuple[str, int]]]] = {}
        if require_all_orientations:
            present = set(self.indices_by_orientation)
            missing = sorted(set(self.orientations) - present)
            if missing:
                raise ValueError(
                    f"Leakage-free {self.mode} dataset cannot include orientations {missing}. "
                    f"split={self.split}, patch_size={self.patch_size}."
                )
        self.eval_indices: list[tuple[str, int]] | None = None
        if self.mode != "train" and self.samples_per_epoch is not None:
            count = int(self.samples_per_epoch)
            if count <= 0:
                raise ValueError("val_samples_per_epoch must be positive.")
            if self.random_sample_when_limited:
                rng = np.random.default_rng(self.seed)
                if self.balance_orientations:
                    available = list(self.indices_by_orientation)
                    self.eval_indices = []
                    for i in range(count):
                        candidates = self.indices_by_orientation[available[i % len(available)]]
                        self.eval_indices.append(candidates[int(rng.integers(len(candidates)))])
                else:
                    self.eval_indices = [self.indices[int(rng.integers(len(self.indices)))] for _ in range(count)]
            else:
                self.eval_indices = [self.indices[i % len(self.indices)] for i in range(count)]
        if self.class_aware_sampling:
            for class_id in range(6):
                self._indices_with_class(class_id)
            self.available_classes = [class_id for class_id, pools in self._class_indices.items() if any(pools.values())]
            missing = sorted(set(range(6)) - set(self.available_classes))
            if missing:
                raise ValueError(f"Classes {[class_id + 1 for class_id in missing]} have no eligible training sections")

    def _indices_with_class(self, class_id: int) -> dict[str, list[tuple[str, int]]]:
        if class_id in self._class_indices:
            return self._class_indices[class_id]
        ranges = subset_ranges(self.split, self.mode)
        lateral_range = max(ranges, key=lambda pair: pair[1] - pair[0])
        split_axis = int(self.split["axis_index"])
        present_by_axis = {
            axis: np.zeros(self.labels.shape[axis], dtype=bool) for axis in AXIS_INDEX.values()
        }
        for start, end in ranges:
            for position in range(int(start), int(end), 8):
                stop = min(int(end), position + 8)
                slicer = [slice(None)] * self.labels.ndim
                slicer[split_axis] = slice(position, stop)
                mask = np.asarray(self.labels[tuple(slicer)]) == class_id + 1
                for axis, present in present_by_axis.items():
                    partial = np.any(mask, axis=tuple(dim for dim in range(3) if dim != axis))
                    if axis == split_axis:
                        present[position:stop] = partial
                    elif position < lateral_range[1] and stop > lateral_range[0]:
                        present |= partial
        pools = {}
        for orientation, entries in self.indices_by_orientation.items():
            axis = AXIS_INDEX[orientation]
            present = present_by_axis[axis]
            pools[orientation] = [entry for entry in entries if present[entry[1]]]
        self._class_indices[class_id] = pools
        return pools

    def _get_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else -1
        if self._rng_worker_id != worker_id:
            # Dataset objects are copied to workers with identical RNG state unless reseeded here.
            self.rng = np.random.default_rng(self.seed + 10_007 * (worker_id + 1))
            self._rng_worker_id = worker_id
        return self.rng

    def _build_indices(self) -> list[tuple[str, int]]:
        indices = []
        ranges = subset_ranges(self.split, self.mode)
        holdout_axis = self.split["axis"]
        for orientation in self.orientations:
            if orientation != holdout_axis and self.crop_strategy != "section_resize":
                if max(end - start for start, end in ranges) < self.patch_size:
                    continue
            n = self.seismic.shape[AXIS_INDEX[orientation]]
            if orientation == holdout_axis:
                # Neighbor-triplet inputs must remain fully inside their physical slab.
                for lo, hi in ranges:
                    indices.extend((orientation, idx) for idx in range(max(0, lo + 1), min(n, hi - 1)))
                continue
            else:
                start, end = 1, n - 1
            start = max(0, start)
            end = min(n, end)
            for idx in range(start, end):
                indices.append((orientation, idx))
        if not indices:
            raise ValueError("No dataset indices were created; check split and volume shape.")
        return indices

    def __len__(self) -> int:
        return int(self.samples_per_epoch or len(self.indices))

    def _slice_triplet(self, orientation: str, idx: int) -> tuple[np.ndarray, np.ndarray]:
        if orientation == "inline":
            image = make_triplet(self.seismic, orientation, idx, bounds=range_for_index(self.split, self.mode, idx))
            label = self.labels[idx, :, :]
        elif orientation == "crossline":
            image = make_triplet(self.seismic, orientation, idx)
            label = self.labels[:, idx, :]
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        image = build_section_features(image, self.normalization, self.feature_mode)
        return image, np.asarray(label, dtype=np.int64) - 1

    def class_counts_for_weights(self) -> np.ndarray:
        subset = "train" if self.mode == "train" else "val"
        return count_labels_in_split(self.labels, self.split, subset)

    def _resize_to_patch(self, image: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ps = self.patch_size
        image_t = torch.from_numpy(image[None]).float()
        label_t = torch.from_numpy(label[None, None].astype(np.float32))
        image_r = F.interpolate(image_t, size=(ps, ps), mode="bilinear", align_corners=False).squeeze(0).numpy()
        label_r = F.interpolate(label_t, size=(ps, ps), mode="nearest").squeeze(0).squeeze(0).numpy().astype(np.int64)
        return image_r, label_r

    def _lateral_range(self, orientation: str) -> tuple[int, int] | None:
        holdout_axis = self.split["axis"]
        if orientation == holdout_axis:
            return None
        start, end = max(subset_ranges(self.split, self.mode), key=lambda pair: pair[1] - pair[0])
        return int(start), int(end)

    def _crop_or_pad(
        self,
        image: np.ndarray,
        label: np.ndarray,
        orientation: str,
        lateral_range: tuple[int, int] | None = None,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_strategy == "section_resize":
            if lateral_range is not None:
                start, end = lateral_range
                if end <= start:
                    raise ValueError(f"Empty {self.mode} lateral range: {lateral_range}")
                image = image[:, start:end, :]
                label = label[start:end, :]
            return self._resize_to_patch(image, label)
        h, w = label.shape
        ps = self.patch_size
        pad_h = max(0, ps - h)
        pad_w = max(0, ps - w)
        if pad_h or pad_w:
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="edge")
            h, w = label.shape
        y_min, y_max = 0, h
        if lateral_range is not None:
            y_min = max(0, int(lateral_range[0]))
            y_max = min(h, int(lateral_range[1]))
            if y_max - y_min < ps:
                raise ValueError(
                    f"Leakage-free crop is impossible: orientation={orientation}, mode={self.mode}, "
                    f"allowed lateral range=({y_min}, {y_max}), patch_size={ps}."
                )
        target_positions = None
        if target_class is not None:
            target_positions = np.argwhere(label[y_min:y_max, :] == target_class)
        if target_positions is not None and len(target_positions):
            target_y, target_x = target_positions[int(self.rng.integers(len(target_positions)))]
            target_y += y_min
            if self.crop_strategy == "random_crop":
                y = int(self.rng.integers(max(y_min, target_y - ps + 1), min(y_max - ps, target_y) + 1))
                x = int(self.rng.integers(max(0, target_x - ps + 1), min(w - ps, target_x) + 1))
            else:
                y = int(np.clip(target_y - ps // 2, y_min, y_max - ps))
                x = int(np.clip(target_x - ps // 2, 0, w - ps))
        elif self.crop_strategy == "random_crop":
            y = int(self.rng.integers(y_min, y_max - ps + 1))
            x = int(self.rng.integers(0, w - ps + 1))
        else:
            y = y_min + max(0, (y_max - y_min - ps) // 2)
            x = max(0, (w - ps) // 2)
        return image[:, y : y + ps, x : x + ps], label[y : y + ps, x : x + ps]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int]:
        rng = self._get_rng()
        target_class = None
        global_context = self.global_context_fraction > 0 and rng.random() < self.global_context_fraction
        if global_context:
            candidates = self.indices_by_orientation["inline"]
            orientation, idx = candidates[int(rng.integers(len(candidates)))]
        elif self.class_aware_sampling and rng.random() < self.class_aware_fraction:
            target_class = self.available_classes[int(rng.integers(len(self.available_classes)))]
            pools = self._class_indices[target_class]
            orientations = [orientation for orientation, entries in pools.items() if entries]
            orientation = orientations[int(rng.integers(len(orientations)))]
            candidates = pools[orientation]
            orientation, idx = candidates[int(rng.integers(len(candidates)))]
        elif self.eval_indices is not None:
            orientation, idx = self.eval_indices[item]
        elif self.mode == "train" and self.samples_per_epoch and self.random_sample_when_limited:
            if self.balance_orientations:
                available = list(self.indices_by_orientation)
                chosen_orientation = available[int(rng.integers(0, len(available)))]
                candidates = self.indices_by_orientation[chosen_orientation]
                orientation, idx = candidates[int(rng.integers(0, len(candidates)))]
            else:
                orientation, idx = self.indices[int(rng.integers(0, len(self.indices)))]
        else:
            orientation, idx = self.indices[item % len(self.indices)]
        image, label = self._slice_triplet(orientation, idx)
        if global_context:
            image, label = self._resize_to_patch(image, label)
        else:
            image, label = self._crop_or_pad(image, label, orientation, self._lateral_range(orientation), target_class)
        if self.mode == "train":
            image, label = augment_image_label(image, label, self.augmentation, rng)
        label_min = int(np.min(label))
        label_max = int(np.max(label))
        if label_min < 0 or label_max >= 6:
            raise ValueError(
                f"Target mask must be in 0..5 after conversion, got min={label_min}, max={label_max}, "
                f"mode={self.mode}, orientation={orientation}, index={idx}"
            )
        return {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": torch.from_numpy(label.copy()).long(),
            "orientation": orientation,
            "index": idx,
        }

    def example_for_class(self, label_1_6: int) -> dict[str, torch.Tensor | str | int] | None:
        """Return one diagnostic patch containing the requested class, outside validation scoring."""
        class_id = int(label_1_6) - 1
        if not 0 <= class_id < 6:
            raise ValueError("label_1_6 must be in 1..6")
        pools = self._indices_with_class(class_id)
        for entries in pools.values():
            for orientation, idx in entries:
                image, label = self._slice_triplet(orientation, idx)
                image, label = self._crop_or_pad(image, label, orientation, self._lateral_range(orientation), class_id)
                if np.any(label == class_id):
                    return {
                        "image": torch.from_numpy(image.copy()).float(),
                        "mask": torch.from_numpy(label.copy()).long(),
                        "orientation": orientation,
                        "index": idx,
                    }
        return None


class FullInlineValidationDataset(Dataset):
    """Complete inline sections inside validation or labeled-test blocks."""

    full_inline = True

    def __init__(
        self,
        seismic: np.ndarray,
        labels_1_6: np.ndarray,
        split: dict,
        normalization: dict,
        *,
        feature_mode: str = "neighbor_triplet",
        max_sections: int | None = 8,
        subset: str = "val",
        diagnostic_patch_size: int = 224,
    ):
        if seismic.shape != labels_1_6.shape:
            raise ValueError("Seismic and label volumes must have the same shape")
        if split["axis"] != "inline":
            raise ValueError("Full inline validation requires an inline spatial split")
        if subset not in {"val", "test"} or subset not in split:
            raise ValueError(f"Unknown labeled evaluation subset: {subset}")
        if subset == "test" and max_sections is not None:
            raise ValueError("Labeled test must evaluate all eligible full inlines (max_sections=None)")
        if max_sections is not None and max_sections < 1:
            raise ValueError("max_sections must be positive")
        self.seismic = seismic
        self.labels_1_6 = labels_1_6
        self.split = split
        self.subset = subset
        self.normalization = normalization
        self.feature_mode = feature_mode
        self.diagnostic_patch_size = int(diagnostic_patch_size)
        if self.diagnostic_patch_size < 1:
            raise ValueError("diagnostic_patch_size must be positive")
        candidates = [idx for start, end in subset_ranges(split, subset) for idx in range(start, end)]
        if not candidates:
            raise ValueError(f"{subset} slab is too small for leakage-free triplets")
        counts = {
            idx: np.bincount(np.asarray(labels_1_6[idx]).astype(np.int64).ravel() - 1, minlength=6)[:6]
            for idx in candidates
        }
        present = set().union(*(set(np.flatnonzero(counts[idx])) for idx in candidates))
        missing = sorted(set(range(6)) - present)
        if missing and subset == "val":
            raise ValueError(f"{subset} inlines do not contain classes {[class_id + 1 for class_id in missing]}")
        if max_sections is None:
            self.indices = [("inline", idx) for idx in candidates]
            self.class_support = np.sum([counts[idx] for idx in candidates], axis=0).astype(np.int64).tolist()
            return
        max_class_counts = np.maximum(np.max(np.stack(list(counts.values())), axis=0), 1)
        selected: list[int] = []
        uncovered = set(range(6))
        while uncovered:
            best = max(
                (idx for idx in candidates if idx not in selected),
                key=lambda idx: (
                    sum(counts[idx][class_id] > 0 for class_id in uncovered),
                    sum(counts[idx][class_id] / max_class_counts[class_id] for class_id in uncovered),
                ),
            )
            selected.append(best)
            uncovered -= set(np.flatnonzero(counts[best]))
            if len(selected) > max_sections:
                raise ValueError("max_sections is too small to cover all validation classes")
        while len(selected) < min(max_sections, len(candidates)):
            selected.append(max(
                (idx for idx in candidates if idx not in selected),
                key=lambda idx: (min(abs(idx - chosen) for chosen in selected), -idx),
            ))
        self.indices = [("inline", idx) for idx in sorted(selected)]
        self.class_support = np.sum([counts[idx] for _, idx in self.indices], axis=0).astype(np.int64).tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int]:
        _, idx = self.indices[item]
        image = build_section_features(
            make_triplet(self.seismic, "inline", idx, bounds=range_for_index(self.split, self.subset, idx)),
            self.normalization, self.feature_mode,
        )
        mask = np.asarray(self.labels_1_6[idx], dtype=np.int64) - 1
        return {
            "image": torch.from_numpy(image.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).long(),
            "orientation": "inline",
            "index": idx,
        }

    def example_for_class(self, label_1_6: int) -> dict[str, torch.Tensor | str | int] | None:
        for item, (_, idx) in enumerate(self.indices):
            if np.any(self.labels_1_6[idx] == label_1_6):
                section = self[item]
                mask = section["mask"].numpy()
                positions = np.argwhere(mask == label_1_6 - 1)
                center_y, center_x = positions[len(positions) // 2]
                h, w = mask.shape
                size_y = min(self.diagnostic_patch_size, h)
                size_x = min(self.diagnostic_patch_size, w)
                y = int(np.clip(center_y - size_y // 2, 0, h - size_y))
                x = int(np.clip(center_x - size_x // 2, 0, w - size_x))
                section["image"] = section["image"][:, y:y + size_y, x:x + size_x]
                section["mask"] = section["mask"][y:y + size_y, x:x + size_x]
                return section
        return None
