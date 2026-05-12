from torch.utils.data import Dataset, Sampler
import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from decord import VideoReader
from torchvision import transforms
import numpy as np
from PIL import Image
import torch

try:
    import av as _av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


class BucketSampler(Sampler):
    """将同一分辨率桶的样本组成 batch，确保同 batch 内 tensor 形状一致。

    与 DataLoader 的 batch_sampler 参数配合使用：
        DataLoader(dataset, batch_sampler=BucketSampler(...), ...)
    每次 __iter__ 产生一个 List[int]（一个 batch 的索引列表）。

    DDP 支持：在 Trainer 中设置 use_distributed_sampler=False，由本类在
    __iter__ / __len__ 中自行按 rank 分片，无需 PL 替换 sampler。
    """

    def __init__(self, sample_index, batch_size, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 按 (bucket_h, bucket_w) 分组
        groups: dict = {}
        for i, entry in enumerate(sample_index):
            key = (entry.get("bucket_h", 0), entry.get("bucket_w", 0))
            groups.setdefault(key, []).append(i)
        self.groups = groups

        # 打印各桶信息
        total_samples = sum(len(v) for v in groups.values())
        for (bh, bw), idxs in sorted(self.groups.items()):
            n_batch = len(idxs) // batch_size if drop_last else (len(idxs) + batch_size - 1) // batch_size
            skip_note = " ⚠ all dropped (too few for batch_size)" if n_batch == 0 else ""
            print(f"BucketSampler: bucket ({bh}x{bw}), samples={len(idxs)}, batches_per_epoch={n_batch}{skip_note}", flush=True)
        print(f"BucketSampler: total samples={total_samples}, total batches/epoch={len(self)}", flush=True)

    @staticmethod
    def _get_dist_info():
        """返回 (rank, world_size)；非 DDP 环境返回 (0, 1)。"""
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        return 0, 1

    def _make_batches(self) -> list:
        """构建本 epoch 的全量 batch 列表（所有 rank 使用同一 seed 保证一致性）。"""
        batches = []
        for indices in self.groups.values():
            idx = list(indices)
            if self.shuffle:
                random.shuffle(idx)
            end = (len(idx) // self.batch_size) * self.batch_size
            for start in range(0, end, self.batch_size):
                batches.append(idx[start: start + self.batch_size])
            if not self.drop_last and end < len(idx):
                batches.append(idx[end:])
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._make_batches()
        rank, world_size = self._get_dist_info()
        # 每个 rank 取出不重叠的子集：rank0 取 0,W,2W,...；rank1 取 1,W+1,...
        yield from batches[rank::world_size]

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (len(indices) + self.batch_size - 1) // self.batch_size
        _, world_size = self._get_dist_info()
        return total // world_size


class CustomDataset(Dataset):
    def __init__(
        self,
        video_root,
        video_root2,
        first_root,
        dataset_roots=None,
        cache_index_path=None,
        height=512,
        width=512,
        sample_n_frames=49,
        is_one2three=False,
        training_len=-1,
        caption_ext=".txt",  # 文本文件后缀
        use_bucket_training=False,  # 是否开启分桶训练
        bucket_align=32,            # 分桶对齐粒度（像素）
        max_long_side=0,            # 最大长边限制（像素）；0 表示不限制
        index_num_workers=8,        # 建索引时的并行线程数
        skip_first_clip=False,      # 若视频可切>=2段，丢弃第0段，只保留后续段
        use_tail_as_ref=False,      # 无外部 reference 图时，用视频尾帧作为 reference
        use_random_as_ref=False,    # 无外部 reference 图时，用 out_indices 范围内随机一帧作为 reference
        ref_drop_prob=0.5,          # reference frame 随机 drop 概率（0=不drop，0.5=50%）
    ):
        self.training_len = training_len
        self.is_one2three = is_one2three

        self.video_root = video_root
        self.video_root2 = video_root2
        self.caption_ext = caption_ext
        self.dataset_roots = [] if dataset_roots is None else list(dataset_roots)
        self.cache_index_path = cache_index_path

        self.height = height
        self.width = width
        self.use_bucket_training = use_bucket_training
        self.bucket_align = bucket_align
        self.max_long_side = max_long_side
        self.index_num_workers = index_num_workers
        self.skip_first_clip = skip_first_clip
        self.use_tail_as_ref = use_tail_as_ref
        self.use_random_as_ref = use_random_as_ref
        self.ref_drop_prob = ref_drop_prob

        # 非分桶模式：固定全局 transform
        self.train_video_transforms = transforms.Compose(
            [
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        # 分桶模式：按 (h, w) 缓存 transform，避免重复构建
        self._transform_cache: dict[tuple, transforms.Compose] = {}

        self.sample_n_frames = sample_n_frames

        if len(self.dataset_roots) > 0:
            self.mode = "split_single_video"
            self.first_root = None
            self.first_paths = []
            self.use_first = True
            self.sample_index = self._build_or_load_index()
            if len(self.sample_index) == 0:
                raise RuntimeError("No valid clip pairs found from dataset_roots.")
            print(
                f"CustomDataset(split_single_video): roots={len(self.dataset_roots)}, "
                f"pairs={len(self.sample_index)}, sample_n_frames={self.sample_n_frames}, "
                f"use_bucket_training={self.use_bucket_training}"
            )
        else:
            self.mode = "paired_roots"
            # --- 可选首帧：目录存在且有 >=1 个可读图片才开启 ---
            img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            if first_root and os.path.isdir(first_root):
                first_list_all = sorted(os.listdir(first_root))
                first_list = [x for x in first_list_all if x.lower().endswith(img_exts)]
                if len(first_list) > 0:
                    self.first_root = first_root
                    self.first_paths = [os.path.join(first_root, x) for x in first_list]
                    self.use_first = True
                else:
                    self.first_root = None
                    self.first_paths = []
                    self.use_first = False
            else:
                self.first_root = None
                self.first_paths = []
                self.use_first = False

            print(
                f"CustomDataset(paired_roots): video_root: {video_root}, "
                f"video_root2: {video_root2}, first_root: {self.first_root or ''}"
            )

            video_exts = (".mp4", ".avi", ".mov", ".mkv")

            # 视频列表（只收视频后缀）
            video_list = sorted(
                [x for x in os.listdir(self.video_root) if x.lower().endswith(video_exts)]
            )
            video_list2 = sorted(
                [x for x in os.listdir(self.video_root2) if x.lower().endswith(video_exts)]
            )

            self.video_paths = [os.path.join(self.video_root, v) for v in video_list]
            self.video_paths2 = [os.path.join(self.video_root2, v) for v in video_list2]

            self.len_videos = len(self.video_paths)
            self.len_videos2 = len(self.video_paths2)
            self.len_firsts = len(self.first_paths)

            # 两路视频必须一一对应
            assert self.len_videos == self.len_videos2, "mismatch in first videos and third videos"

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _get_transform(self, h: int, w: int) -> transforms.Compose:
        """按 (h, w) 缓存 transform，分桶模式下每个桶只构建一次。"""
        key = (h, w)
        if key not in self._transform_cache:
            self._transform_cache[key] = transforms.Compose(
                [
                    transforms.Resize((h, w)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        return self._transform_cache[key]

    def _snap_to_bucket(self, native_h: int, native_w: int) -> tuple[int, int]:
        """将原始分辨率按比例缩放（max_long_side 限制）后，向下对齐到 bucket_align 的整数倍。"""
        h, w = native_h, native_w
        if self.max_long_side > 0:
            long_side = max(h, w)
            if long_side > self.max_long_side:
                scale = self.max_long_side / long_side
                h = int(h * scale)
                w = int(w * scale)
        bh = max((h // self.bucket_align) * self.bucket_align, self.bucket_align)
        bw = max((w // self.bucket_align) * self.bucket_align, self.bucket_align)
        return bh, bw

    # ------------------------------------------------------------------
    # Index building / caching
    # ------------------------------------------------------------------

    def _build_or_load_index(self):
        if self.cache_index_path and os.path.isfile(self.cache_index_path):
            try:
                with open(self.cache_index_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                meta_ok = (
                    payload.get("sample_n_frames") == self.sample_n_frames
                    and payload.get("use_bucket_training") == self.use_bucket_training
                    and payload.get("bucket_align", 0) == (self.bucket_align if self.use_bucket_training else 0)
                    and payload.get("max_long_side", 0) == self.max_long_side
                    and payload.get("skip_first_clip", False) == self.skip_first_clip
                )
                if meta_ok:
                    cached_entries = payload.get("entries", [])
                    if len(cached_entries) > 0:
                        print(
                            f"CustomDataset: loaded cached index from {self.cache_index_path}, entries={len(cached_entries)}",
                            flush=True,
                        )
                        return cached_entries
            except Exception:
                # 损坏/空缓存时自动重建，避免启动失败。
                pass

        backend = "av" if _AV_AVAILABLE else "decord"
        print(f"CustomDataset: building dataset index (workers={self.index_num_workers}, backend={backend}) ...", flush=True)
        entries = self._build_index_from_roots(index_num_workers=self.index_num_workers)
        if self.cache_index_path:
            cache_dir = os.path.dirname(self.cache_index_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            payload = {
                "sample_n_frames": self.sample_n_frames,
                "use_bucket_training": self.use_bucket_training,
                "bucket_align": self.bucket_align if self.use_bucket_training else 0,
                "max_long_side": self.max_long_side,
                "skip_first_clip": self.skip_first_clip,
                "entries": entries,
            }
            # 原子写入：先写临时文件再 rename，防止中断留下损坏的缓存
            tmp_path = self.cache_index_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, self.cache_index_path)
            print(
                f"CustomDataset: saved index cache to {self.cache_index_path}, entries={len(entries)}",
                flush=True,
            )
        return entries

    @staticmethod
    def _get_video_meta(video_path: str) -> tuple:
        """从视频容器头快速读取 (total_frames, native_h, native_w)，不解码任何帧。

        优先用 av（只读 moov box，速度极快）；失败时回退 decord（需解码首帧，较慢）。
        返回 (-1, -1, -1) 表示读取失败。
        """
        if _AV_AVAILABLE:
            try:
                with _av.open(video_path) as container:
                    stream = container.streams.video[0]
                    native_h = stream.height
                    native_w = stream.width
                    total_frames = stream.frames
                    # 部分 MP4 不存储帧数，从时长和帧率估算
                    if total_frames == 0 and stream.duration and stream.time_base and stream.average_rate:
                        total_frames = int(
                            float(stream.duration * stream.time_base) * float(stream.average_rate)
                        )
                if native_h > 0 and native_w > 0 and total_frames > 0:
                    return total_frames, native_h, native_w
            except Exception:
                pass
        # 回退：decord（需解码首帧）
        try:
            vr = VideoReader(video_path)
            total_frames = len(vr)
            frame0 = vr[0].asnumpy()
            return total_frames, int(frame0.shape[0]), int(frame0.shape[1])
        except Exception:
            return -1, -1, -1

    def _build_index_from_roots(self, index_num_workers: int = 8):
        """扫描所有 dataset_roots，构建 sample_index。

        使用 ThreadPoolExecutor 并行读取视频元数据，配合 av 避免解码帧，
        大幅加速大规模数据集（70K+ 视频）的首次索引构建。
        """
        entries = []
        video_exts = (".mp4", ".avi", ".mov", ".mkv")
        txt_exts = (".txt",)
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

        def _group_key(stem: str):
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                return parts[0]
            return stem

        def _collect_triplets(root: str):
            """收集 root 下所有 (video_path, prompt_path, first_path) 三元组，不做 IO。"""
            triplets = []
            # 模式 A: 子目录，每目录一组 mp4+txt(+可选 png)
            for name in sorted(os.listdir(root)):
                folder = os.path.join(root, name)
                if not os.path.isdir(folder):
                    continue
                files = sorted(os.listdir(folder))
                mp4s = [x for x in files if x.lower().endswith(video_exts)]
                txts = [x for x in files if x.lower().endswith(txt_exts)]
                imgs = [x for x in files if x.lower().endswith(img_exts)]
                if not mp4s or not txts:
                    continue
                triplets.append((
                    os.path.join(folder, mp4s[0]),
                    os.path.join(folder, txts[0]),
                    os.path.join(folder, imgs[0]) if imgs else None,
                ))

            # 模式 B: 平铺文件，按 stem 配对
            files = sorted(os.listdir(root))
            mp4s = [x for x in files if x.lower().endswith(video_exts)]
            txts = [x for x in files if x.lower().endswith(txt_exts)]
            imgs = [x for x in files if x.lower().endswith(img_exts)]
            txt_by_stem = {os.path.splitext(x)[0]: x for x in txts}
            img_by_stem = {os.path.splitext(x)[0]: x for x in imgs}

            mode_b_found = False
            for v in mp4s:
                stem = os.path.splitext(v)[0]
                t = txt_by_stem.get(stem)
                if t is None:
                    continue
                i = img_by_stem.get(stem)
                triplets.append((
                    os.path.join(root, v),
                    os.path.join(root, t),
                    os.path.join(root, i) if i else None,
                ))
                mode_b_found = True

            # 模式 B-fallback: 按去尾号分组匹配（仅当模式 B 未匹配到任何视频）
            if not mode_b_found and not triplets:
                txt_group: dict = {}
                img_group: dict = {}
                for t in txts:
                    txt_group.setdefault(_group_key(os.path.splitext(t)[0]), []).append(t)
                for i in imgs:
                    img_group.setdefault(_group_key(os.path.splitext(i)[0]), []).append(i)
                for v in mp4s:
                    key = _group_key(os.path.splitext(v)[0])
                    t_list = txt_group.get(key, [])
                    if not t_list:
                        continue
                    i_list = img_group.get(key, [])
                    triplets.append((
                        os.path.join(root, v),
                        os.path.join(root, sorted(t_list)[0]),
                        os.path.join(root, sorted(i_list)[0]) if i_list else None,
                    ))
            return triplets

        def _process_triplet(args):
            """读取单个视频元数据并生成 clip entries（在线程池中执行）。"""
            video_path, prompt_path, first_path, use_bucket, height, width = args
            total_frames, native_h, native_w = CustomDataset._get_video_meta(video_path)
            if total_frames < 0:
                return [], "LOAD_FAIL", os.path.basename(video_path), 0, 0, 0, 0, 0

            if use_bucket:
                # _snap_to_bucket 需要 self，这里直接内联计算
                bh = max((native_h // 32) * 32, 32)  # bucket_align 固定为 32 在此处
                bw = max((native_w // 32) * 32, 32)
            else:
                bh, bw = height, width

            half = total_frames // 2
            usable_half = min(half, total_frames - half)
            max_pairs = usable_half // self.sample_n_frames
            if max_pairs <= 0:
                return [], "TOO_SHORT", os.path.basename(video_path), total_frames, native_h, native_w, bh, bw

            clip_entries = []
            for clip_idx in range(max_pairs):
                clip_entries.append({
                    "video_path": video_path,
                    "prompt_path": prompt_path,
                    "first_path": first_path,
                    "in_start": clip_idx * self.sample_n_frames,
                    "out_start": half + clip_idx * self.sample_n_frames,
                    "native_h": native_h,
                    "native_w": native_w,
                    "bucket_h": bh,
                    "bucket_w": bw,
                    "clip_idx": clip_idx,
                    "num_clips": max_pairs,
                })
            return clip_entries, "OK", os.path.basename(video_path), total_frames, native_h, native_w, bh, bw

        # 注意：_snap_to_bucket 需要 self.max_long_side 参与缩放，上面 _process_triplet 里
        # 内联了 bucket_align=32 的 floor 逻辑。当 max_long_side > 0 时需要在外部先缩放。
        # 这里把缩放后的结果作为 "effective_h/w" 传入，或直接复用 self._snap_to_bucket。
        # 为保持正确性，改用 wrapper 把 self._snap_to_bucket 绑定进去。
        snap = self._snap_to_bucket  # 闭包捕获，支持 max_long_side

        def _process_triplet_with_snap(args):
            video_path, prompt_path, first_path = args
            total_frames, native_h, native_w = CustomDataset._get_video_meta(video_path)
            if total_frames < 0:
                return [], "LOAD_FAIL", os.path.basename(video_path), 0, 0, 0, 0, 0

            if self.use_bucket_training:
                bh, bw = snap(native_h, native_w)
            else:
                bh, bw = self.height, self.width

            half = total_frames // 2
            usable_half = min(half, total_frames - half)
            max_pairs = usable_half // self.sample_n_frames
            if max_pairs <= 0:
                return [], "TOO_SHORT", os.path.basename(video_path), total_frames, native_h, native_w, bh, bw

            # skip_first_clip=True 且可切 >=2 段时，从第 1 段开始（丢弃第 0 段）
            clip_start = 1 if (self.skip_first_clip and max_pairs > 1) else 0
            num_clips = max_pairs - clip_start
            clip_entries = [
                {
                    "video_path": video_path,
                    "prompt_path": prompt_path,
                    "first_path": first_path,
                    "in_start": clip_idx * self.sample_n_frames,
                    "out_start": half + clip_idx * self.sample_n_frames,
                    "native_h": native_h,
                    "native_w": native_w,
                    "bucket_h": bh,
                    "bucket_w": bw,
                    "clip_idx": clip_idx - clip_start,
                    "num_clips": num_clips,
                }
                for clip_idx in range(clip_start, max_pairs)
            ]
            return clip_entries, "OK", os.path.basename(video_path), total_frames, native_h, native_w, bh, bw

        for root in self.dataset_roots:
            if not root or not os.path.isdir(root):
                continue
            root_entries_before = len(entries)
            t0 = time.time()
            print(f"CustomDataset: scanning root {root} ...", flush=True)

            triplets = _collect_triplets(root)
            n_ok = n_short = n_fail = 0
            bucket_summary: dict = {}  # {(bh,bw): count}

            with ThreadPoolExecutor(max_workers=index_num_workers) as pool:
                futures = {pool.submit(_process_triplet_with_snap, t): t for t in triplets}
                for future in as_completed(futures):
                    clip_entries, status, name, total_f, nh, nw, bh, bw = future.result()
                    if status == "OK":
                        entries.extend(clip_entries)
                        n_ok += 1
                        bucket_summary[(bh, bw)] = bucket_summary.get((bh, bw), 0) + len(clip_entries)
                    elif status == "TOO_SHORT":
                        n_short += 1
                    else:
                        n_fail += 1

            # 打印每个桶的样本数汇总（不逐视频打印）
            for (bh, bw), cnt in sorted(bucket_summary.items()):
                print(f"  bucket ({bh}x{bw}): {cnt} clips", flush=True)
            if n_short:
                print(f"  skipped {n_short} videos (too short for {self.sample_n_frames} frames)", flush=True)
            if n_fail:
                print(f"  skipped {n_fail} videos (load failed)", flush=True)

            new_entries = len(entries) - root_entries_before
            print(
                f"CustomDataset: root done, videos_ok={n_ok}, new_entries={new_entries}, "
                f"total_entries={len(entries)}, elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

        # as_completed 的返回顺序非确定，会导致不同 rank 的 sample_index 顺序不一致，
        # 进而让 BucketSampler 的 batches[rank::world_size] 分片出现重叠。
        entries.sort(key=lambda e: (e["video_path"], e["clip_idx"]))
        return entries

    def __len__(self):
        if self.mode == "split_single_video":
            if self.training_len != -1:
                return self.training_len
            return len(self.sample_index)

        if self.training_len != -1:
            return self.training_len
        if self.use_first:
            # 仅当真的有首帧时才把 len_firsts 纳入长度
            return min(self.len_videos, self.len_videos2, self.len_firsts)
        else:
            return min(self.len_videos, self.len_videos2)

    def _caption_path_for(self, video2_path: str):
        stem, _ = os.path.splitext(video2_path)
        return stem + self.caption_ext

    def _load_caption(self, video2_path: str) -> str:
        cap_path = self._caption_path_for(video2_path)
        if os.path.exists(cap_path) and os.path.isfile(cap_path):
            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return ""
        return ""

    def __getitem__(self, index):
        if self.mode == "split_single_video":
            if len(self.sample_index) <= 0:
                raise RuntimeError("No valid samples in sample_index.")

            index = index % len(self.sample_index)
            meta = self.sample_index[index]

            # 分桶模式：每条 entry 有自己的桶尺寸；非分桶模式：用全局 height/width
            bucket_h = meta.get("bucket_h", self.height)
            bucket_w = meta.get("bucket_w", self.width)
            transform = self._get_transform(bucket_h, bucket_w)

            video_reader = VideoReader(meta["video_path"])
            in_indices = meta["in_start"] + np.arange(self.sample_n_frames)
            out_indices = meta["out_start"] + np.arange(self.sample_n_frames)

            video = video_reader.get_batch(in_indices).asnumpy()  # F, H, W, C
            video = [Image.fromarray(frame) for frame in video]
            pixel_values = [transform(frame) for frame in video]
            pixel_values = torch.stack(pixel_values)  # F, C, H, W

            video2 = video_reader.get_batch(out_indices).asnumpy()  # F, H, W, C
            video2 = [Image.fromarray(frame) for frame in video2]
            pixel_values2 = [transform(frame) for frame in video2]
            pixel_values2 = torch.stack(pixel_values2)  # F, C, H, W

            # reference frame：
            #   - 最高优先级：ref_drop_prob 概率 per-sample 丢弃 ref（=1 → 纯文本驱动）
            #   - 否则按优先级回退：first_path PNG → use_random_as_ref → use_tail_as_ref
            # 注意：dataset 层 drop 会让同一 batch 内出现"有/无 first_frames"混合，
            # 默认 collate 会 KeyError。推理 batch_size=1 安全；训练若 batch>1 且
            # 0<ref_drop_prob<1，请保留 trainer 的 batch 级 drop 并把 dataset 设 0。
            first_frame = None
            drop_ref = self.ref_drop_prob > 0 and random.random() < self.ref_drop_prob
            if not drop_ref:
                if meta.get("first_path") is not None:
                    try:
                        first_frame = Image.open(meta["first_path"]).convert("RGB")
                        first_frame = transform(first_frame)
                    except Exception:
                        first_frame = None
                if first_frame is None and self.use_random_as_ref:
                    # 从 out_indices 范围内随机取一帧作为 reference key frame
                    rand_idx = int(out_indices[random.randint(0, len(out_indices) - 1)])
                    try:
                        rand_frame = video_reader.get_batch([rand_idx]).asnumpy()[0]  # H, W, C
                        first_frame = transform(Image.fromarray(rand_frame))
                    except Exception:
                        first_frame = None
                if first_frame is None and self.use_tail_as_ref:
                    # 取 out_indices 的最后一帧（编辑视频末帧）作为 reference
                    tail_idx = int(out_indices[-1])
                    try:
                        tail = video_reader.get_batch([tail_idx]).asnumpy()[0]  # H, W, C
                        first_frame = transform(Image.fromarray(tail))
                    except Exception:
                        first_frame = None

            try:
                with open(meta["prompt_path"], "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            except Exception:
                prompt = ""

            sample = {
                "pixel_values": pixel_values.permute(1, 0, 2, 3),   # C, F, H, W
                "pixel_values2": pixel_values2.permute(1, 0, 2, 3),  # C, F, H, W
                "prompts": prompt,
                "video_path": meta["video_path"],
                "clip_idx": int(meta.get("clip_idx", 0)),
                "num_clips": int(meta.get("num_clips", 1)),
            }
            if first_frame is not None:
                sample["first_frames"] = first_frame
            return sample

        # ---- paired_roots 模式（不支持分桶，仍使用固定分辨率） ----

        # 根据是否启用首帧决定取模长度，避免越界
        if self.use_first:
            min_len = min(self.len_videos, self.len_videos2, self.len_firsts)
        else:
            min_len = min(self.len_videos, self.len_videos2)

        # 如果真的没有数据，给出清晰报错（避免除以 0 或空取模）
        if min_len <= 0:
            raise RuntimeError(
                f"No valid samples: "
                f"len_videos={self.len_videos}, len_videos2={self.len_videos2}, len_firsts={self.len_firsts}."
            )

        index = index % min_len

        # video A
        video_path = self.video_paths[index]
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)

        # video B (GT)
        video_path2 = self.video_paths2[index]
        video_reader2 = VideoReader(video_path2)
        video_length2 = len(video_reader2)

        # 可选的首帧
        first_frame = None
        if self.use_first:
            # 这里安全：index < len_firsts
            first_frame_path = self.first_paths[index]
            first_frame = Image.open(first_frame_path).convert("RGB")
            first_frame = self.train_video_transforms(first_frame)

        assert video_length == video_length2, "video lengths do not match"
        assert self.sample_n_frames <= video_length, "sample_n_frames > video length"

        # 采样帧（stride=2），并避免 randint 上界为负
        stride = 1
        available = video_length - (self.sample_n_frames - 1) * stride
        available = max(available, 1)
        if available <= 4:
            start_index = 0
        else:
            start_index = np.random.randint(0, available - 3)
        frame_indices = start_index + np.arange(self.sample_n_frames) * stride

        # 读取视频 A
        video = video_reader.get_batch(frame_indices).asnumpy()  # F, H, W, C
        video = [Image.fromarray(frame) for frame in video]
        pixel_values = [self.train_video_transforms(frame) for frame in video]
        pixel_values = torch.stack(pixel_values)  # F, C, H, W

        # 读取视频 B (GT)
        video2 = video_reader2.get_batch(frame_indices).asnumpy()
        video2 = [Image.fromarray(frame) for frame in video2]
        pixel_values2 = [self.train_video_transforms(frame) for frame in video2]
        pixel_values2 = torch.stack(pixel_values2)  # F, C, H, W

        # 文本：优先读取 video_root2 同名 .txt；没有则返回空字符串
        prompt = self._load_caption(video_path2)

        sample = {
            "pixel_values": pixel_values.permute(1, 0, 2, 3),       # C, F, H, W
            "pixel_values2": pixel_values2.permute(1, 0, 2, 3),     # C, F, H, W
            "prompts": prompt,
        }
        if self.use_first:
            sample["first_frames"] = first_frame  # C, H, W

        return sample
