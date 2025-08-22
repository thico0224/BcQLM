import os
import io
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TXT_EXTS = {".txt"}


def _is_image(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in IMG_EXTS


def _is_text(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in TXT_EXTS


def _read_text(fp: str, encoding: str = "utf-8") -> str:
    with io.open(fp, "r", encoding=encoding, errors="ignore") as f:
        return f.read().strip()


class PairedFolderDataset(Dataset):


    def __init__(
        self,
        image_folder: str,
        text_folder: str,
        image_transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        max_text_length: Optional[int] = 77,
        encoding: str = "utf-8",
        return_stem: bool = False,
    ) -> None:
        super().__init__()
        self.image_folder = os.path.abspath(image_folder)
        self.text_folder = os.path.abspath(text_folder)
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.encoding = encoding
        self.return_stem = return_stem

        if not os.path.isdir(self.image_folder):
            raise FileNotFoundError(f"image_folder not found: {self.image_folder}")
        if not os.path.isdir(self.text_folder):
            raise FileNotFoundError(f"text_folder not found: {self.text_folder}")

        # index by stem
        img_map: Dict[str, str] = {}
        for fname in os.listdir(self.image_folder):
            if _is_image(fname):
                stem = os.path.splitext(fname)[0]
                img_map[stem] = os.path.join(self.image_folder, fname)

        txt_map: Dict[str, str] = {}
        for fname in os.listdir(self.text_folder):
            if _is_text(fname):
                stem = os.path.splitext(fname)[0]
                txt_map[stem] = os.path.join(self.text_folder, fname)

        # intersection (paired samples)
        common_stems = sorted(set(img_map.keys()) & set(txt_map.keys()))
        if len(common_stems) == 0:
            raise RuntimeError(
                f"No paired samples found.\n"
                f"image_folder={self.image_folder}\ntext_folder={self.text_folder}\n"
                f"Tip: file basenames must match, e.g., 0001.jpg ↔ 0001.txt"
            )

        self.samples: List[Tuple[str, str, str]] = []
        for stem in common_stems:
            self.samples.append((stem, img_map[stem], txt_map[stem]))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, index: int) -> Dict:
        stem, img_fp, txt_fp = self.samples[index]

        # image
        image = self._load_image(img_fp)
        if self.image_transform is not None:
            pixel_values = self.image_transform(image)
        else:
            # 延迟到外部 transform；也可以抛错提醒
            pixel_values = image

        # text
        raw_text = _read_text(txt_fp, encoding=self.encoding)
        item: Dict = {"pixel_values": pixel_values, "raw_text": raw_text}

        # tokenize if provided
        if self.tokenizer is not None:
            tok_kwargs = {}
            if self.max_text_length is not None:
                tok_kwargs.update(
                    dict(
                        max_length=self.max_text_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                )
            encoded = self.tokenizer(raw_text, **tok_kwargs)
            # squeeze batch dim if present
            for k, v in encoded.items():
                try:
                    item[k] = v.squeeze(0)
                except Exception:
                    item[k] = v

        if self.return_stem:
            item["stem"] = stem

        return item

    # convenience constructor
    @classmethod
    def from_folders(
        cls,
        image_folder: str,
        text_folder: str,
        image_transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
        **kwargs,
    ) -> "PairedFolderDataset":

        return cls(
            image_folder=image_folder,
            text_folder=text_folder,
            image_transform=image_transform,
            tokenizer=tokenizer,
            **kwargs,
        )