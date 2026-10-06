"""ZIP_LZMA với từ điển lớn (mặc định zipfile chỉ preset 6 / 8MB): chunk của cùng doc lặp lại ở nhiều câu → từ điển 512MB nén được.
Header LZMA ghi props (dict size) nên mọi bộ giải nén chuẩn (Python zipfile, 7z) đều đọc được.
    python scripts/zip_lzma.py in.zip out.zip   (hoặc import write_zip)"""
import lzma
import struct
import sys
import zipfile

DICT = 512 << 20


def _init(self):
    f = {"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": DICT}
    props = lzma._encode_filter_properties(f)
    self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[f])
    return struct.pack("<BBH", 9, 4, len(props)) + props


zipfile.LZMACompressor._init = _init


def write_zip(out, data: bytes, name="submission.json"):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_LZMA) as z:
        z.writestr(name, data)


if __name__ == "__main__":
    d = zipfile.ZipFile(sys.argv[1]).read("submission.json")
    write_zip(sys.argv[2], d)
    assert zipfile.ZipFile(sys.argv[2]).read("submission.json") == d
    print("ok", len(d) >> 20, "MB raw")
