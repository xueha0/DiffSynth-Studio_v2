from .operators import *
import os, torch, json, pandas


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        stat_path=None,
        action_type=None,
        num_samples=None,
        sample_indices=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.num_samples = num_samples
        self.sample_indices = None if sample_indices is None else list(sample_indices)
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        self.stat = {}
        self.load_stats(stat_path, action_type=action_type)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        resize_mode="fit",
    ):
        return RouteByType(operator_map=[
            (dict, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode)),
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
        resize_mode="fit",
    ):
        video_operator = RouteByExtensionName(operator_map=[
            (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode) >> ToList()),
            (("gif",), LoadGIF(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode),
            )),
            (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                num_frames, time_division_factor, time_division_remainder,
                frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor, resize_mode),
            )),
        ])
        return RouteByType(operator_map=[
            (dict, ToAbsolutePath(base_path) >> video_operator),
            (str, ToAbsolutePath(base_path) >> video_operator),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> video_operator)),
        ]) >> ToVideoTensor()
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            self.search_for_cached_data_files(self.base_path)
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        self._filter_metadata_keys()
        self._apply_sample_selection()
        base_size = len(self.cached_data) if self.load_from_cache else len(self.data)
        print(f"Dataset size: {base_size}, repeat: {self.repeat}, total: {len(self)}")

    def _filter_metadata_keys(self):
        if self.load_from_cache or not self.data_file_keys:
            return
        allowed_keys = set(self.data_file_keys)
        for item in self.data:
            if not isinstance(item, dict):
                continue
            for key in list(item.keys()):
                if key in allowed_keys:
                    continue
                if self._looks_like_file_ref(item[key]):
                    item.pop(key, None)

    def _looks_like_file_ref(self, value):
        if isinstance(value, dict):
            if "data" not in value:
                return False
            value = value.get("data")
        if isinstance(value, (list, tuple)):
            return any(self._looks_like_file_ref(v) for v in value)
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str):
            return False
        path = value.strip()
        if not path:
            return False
        if "/" in path or "\\" in path:
            return True
        root, ext = os.path.splitext(path)
        if ext and root and not any(ch.isspace() for ch in path):
            return True
        return False

    def _apply_sample_selection(self):
        base_size = len(self.cached_data) if self.load_from_cache else len(self.data)
        indices = self.sample_indices
        if indices is None:
            if self.num_samples is None:
                return
            num_samples = int(self.num_samples)
            if num_samples <= 0 or num_samples >= base_size:
                return
            indices = list(range(num_samples))
        else:
            indices = [int(i) for i in indices]

        invalid = [idx for idx in indices if idx < 0 or idx >= base_size]
        if invalid:
            raise IndexError(f"Sample indices out of range: {invalid} (dataset size: {base_size})")

        self.sample_indices = indices
        if self.load_from_cache:
            self.cached_data = [self.cached_data[idx] for idx in indices]
        else:
            self.data = [self.data[idx] for idx in indices]

    def load_stats(self, stat_path, action_type=None):
        if not stat_path:
            return
        if stat_path.lower().endswith(".jsonl"):
            raise ValueError("Per-episode stats are not supported; use stat.json instead.")
        with open(stat_path, "r") as f:
            stats = json.load(f)
        print(f"Loaded stats from {stat_path} (type={type(stats).__name__})")
        if action_type:
            if action_type in stats:
                self.stat = {action_type: stats[action_type]}
            elif isinstance(stats, dict) and "min" in stats and "max" in stats:
                self.stat = {action_type: stats}
            else:
                raise KeyError(f"Missing stats for action type: {action_type}")
        else:
            self.stat = stats

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            sample_index = data_id % len(self.data)
            data = self.data[sample_index].copy()
            data.setdefault("__sample_id__", sample_index)
            for key in self.data_file_keys:
                if key in self.special_operator_map:
                    source = data[key] if key in data else data
                    source = self._wrap_frame_range_metadata(data, source)
                    data[key] = self.special_operator_map[key](source)
                elif key in data:
                    source = self._wrap_frame_range_metadata(data, data[key])
                    data[key] = self.main_data_operator(source)
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True

    def _wrap_frame_range_metadata(self, data, payload):
        if not isinstance(data, dict):
            return payload
        start_frame = data.get("start_frame")
        end_frame = data.get("end_frame")
        pad_to_frames = data.get("pad_to_frames")
        pad_mode = data.get("pad_mode")

        def wrap_item(item):
            if isinstance(item, str):
                wrapped = {"data": item}
                if start_frame is not None:
                    wrapped["start_frame"] = start_frame
                if end_frame is not None:
                    wrapped["end_frame"] = end_frame
                if pad_to_frames is not None:
                    wrapped["pad_to_frames"] = pad_to_frames
                if pad_mode is not None:
                    wrapped["pad_mode"] = pad_mode
                return wrapped
            if isinstance(item, dict):
                merged = item.copy()
                if merged.get("start_frame") is None and start_frame is not None:
                    merged["start_frame"] = start_frame
                if merged.get("end_frame") is None and end_frame is not None:
                    merged["end_frame"] = end_frame
                if merged.get("pad_to_frames") is None and pad_to_frames is not None:
                    merged["pad_to_frames"] = pad_to_frames
                if merged.get("pad_mode") is None and pad_mode is not None:
                    merged["pad_mode"] = pad_mode
                return merged
            return item

        if isinstance(payload, (list, tuple)):
            return [wrap_item(item) for item in payload]
        return wrap_item(payload)
