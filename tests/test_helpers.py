"""Tests for biocracker.helpers module."""

import io
import zipfile
from pathlib import Path
from types import TracebackType

from pytest import MonkeyPatch

from biocracker.helpers import (
    _collapse_singleton,
    _guess_filename_from_url,
    _resolve_return_path,
    _slug_url,
    download_and_prepare,
    get_biocracker_cache_dir,
)


class _FakeHTTPResponse:
    """A fake HTTP response object for testing purposes."""

    def __init__(self, data: bytes, chunk: int = 1024) -> None:
        """
        Initialize the fake HTTP response.

        :param data: bytes, the content to be read
        :param chunk: int, the chunk size for reading
        """
        self._buf = io.BytesIO(data)
        self._chunk = chunk
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n: int = -1) -> bytes:
        """
        Read up to n bytes from the response.

        :param n: int, number of bytes to read. If -1, read all
        :return: bytes, the read content
        """
        if n == -1:
            return self._buf.read()

        return self._buf.read(n)

    def __enter__(self) -> "_FakeHTTPResponse":
        """
        Enter the runtime context related to this object.

        :return: self
        """
        return self

    def __exit__(self, exc_type: type, exc: Exception, tb: TracebackType) -> bool:
        """
        Exit the runtime context related to this object.

        :param exc_type: type, the exception type
        :param exc: Exception, the exception instance
        :param tb: TracebackType, the traceback object
        :return: bool, False to propagate exceptions
        """
        return False


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    """
    Create a ZIP archive in memory containing the specified files.

    :returns: a bytes object of a ZIP containing those files
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)

    return bio.getvalue()


def test_get_cache_dir_custom_path_creates_marker(tmp_path: Path) -> None:
    """
    Test that providing a custom cache directory creates the directory and marker file.

    :param tmp_path: Path, a temporary directory provided by pytest
    """
    custom = tmp_path / "mycache"
    path = get_biocracker_cache_dir(custom)
    assert path == custom
    assert path.exists()
    assert (path / ".biocracker_cache_marker").exists()


# def test_get_cache_dir_auto_linux_uses_xdg(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
#     """
#     Test that on Linux, the cache directory is created under XDG_CACHE_HOME.

#     :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
#     :param tmp_path: Path, a temporary directory provided by pytest
#     """
#     # Pretend we're on Linux
#     monkeypatch.setattr("platform.system", lambda: "Linux")
#     xdg = tmp_path / "xdg"
#     monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
#     path = get_biocracker_cache_dir()
#     assert path == xdg / "biocracker"
#     assert (path / ".biocracker_cache_marker").exists()


# def test_get_cache_dir_auto_macos(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
#     """
#     Test that on macOS, the cache directory is created under ~/Library/Caches.

#     :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
#     :param tmp_path: Path, a temporary directory provided by pytest
#     """
#     monkeypatch.setattr("platform.system", lambda: "Darwin")
#     monkeypatch.setenv("HOME", str(tmp_path))  # control home
#     path = get_biocracker_cache_dir()
#     assert path == tmp_path / "Library" / "Caches" / "biocracker"
#     assert (path / ".biocracker_cache_marker").exists()


# def test_get_cache_dir_auto_windows(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
#     """
#     Test that on Windows, the cache directory is created under LOCALAPPDATA.

#     :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
#     :param tmp_path: Path, a temporary directory provided by pytest
#     """
#     monkeypatch.setattr("platform.system", lambda: "Windows")
#     monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
#     path = get_biocracker_cache_dir()
#     assert path == tmp_path / "AppData" / "Local" / "biocracker"
#     assert (path / ".biocracker_cache_marker").exists()


def test_slug_and_guess_filename() -> None:
    """
    Test _guess_filename_from_url and _slug_url functions.
    """
    url = "https://example.com/path/to/file-model-v1.zip"
    assert _guess_filename_from_url(url) == "file-model-v1.zip"
    slug1 = _slug_url(url)
    slug2 = _slug_url(url)
    assert slug1 == slug2  # deterministic
    assert slug1.startswith("file-model-v1.zip-")
    assert len(slug1.split("-")[-1]) == 16  # 16 hex chars


def test_download_non_archive_idempotent(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """
    Test that downloading a non-archive file is idempotent and uses caching.

    :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
    :param tmp_path: Path, a temporary directory provided by pytest
    """
    # Fake small payload
    data = b"hello-world"
    calls = {"count": 0}

    def fake_urlopen(req):
        calls["count"] += 1
        return _FakeHTTPResponse(data)

    monkeypatch.setattr("biocracker.helpers.urlopen", fake_urlopen)

    # First call downloads
    out = download_and_prepare(
        "https://example.com/payload.bin",
        cache_dir=tmp_path,
    )
    assert out.is_file()
    assert out.read_bytes() == data
    assert calls["count"] == 1

    # Second call should hit READY and not re-download
    def fail_urlopen(_):
        raise AssertionError("Should not be called on idempotent run")

    monkeypatch.setattr("biocracker.helpers.urlopen", fail_urlopen)
    out2 = download_and_prepare(
        "https://example.com/payload.bin",
        cache_dir=tmp_path,
    )
    assert out2 == out


def test_download_zip_single_file_returns_inner_file(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """
    Test downloading a ZIP archive with a single file returns that file directly.

    :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
    :param tmp_path: Path, a temporary directory provided by pytest
    """
    zip_bytes = _make_zip_bytes({"model.pt": b"MODEL-DATA"})
    monkeypatch.setattr("biocracker.helpers.urlopen", lambda req: _FakeHTTPResponse(zip_bytes))

    url = "https://host/models/model.zip"
    out = download_and_prepare(url, cache_dir=tmp_path)
    # Should return the single inner file
    assert out.is_file()
    assert out.name == "model.pt"
    assert out.read_bytes() == b"MODEL-DATA"
    # Ensure the container zip was removed (only extracted contents remain)
    item_dir = next((tmp_path / "downloads").rglob("model.zip-*"))
    assert not (item_dir / "model.zip").exists()


def test_download_zip_multi_file_returns_dir(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """
    Test downloading a ZIP archive with multiple files returns the extraction directory.

    :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
    :param tmp_path: Path, a temporary directory provided by pytest
    """
    zip_bytes = _make_zip_bytes(
        {
            "dir/a.txt": b"A",
            "dir/b.txt": b"B",
        }
    )
    monkeypatch.setattr("biocracker.helpers.urlopen", lambda req: _FakeHTTPResponse(zip_bytes))

    url = "https://host/archive/data.zip"
    out = download_and_prepare(url, cache_dir=tmp_path)
    # Should return a directory (multiple files)
    assert out.is_dir()
    files = sorted(p.name for p in out.rglob("*.txt"))
    assert files == ["a.txt", "b.txt"]


def test_resume_incomplete_extraction(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """
    Test resuming an incomplete extraction of a ZIP file.

    :param tmp_path: Path, a temporary directory provided by pytest
    :param monkeypatch: MonkeyPatch, pytest fixture for monkeypatching
    """
    # Prepare item_dir matching the slug rule
    url = "https://host/pkg/single.zip"
    from biocracker.helpers import _slug_url, get_biocracker_cache_dir

    base = get_biocracker_cache_dir(tmp_path)
    downloads_root = base / "downloads"
    slug = _slug_url(url)
    item_dir = downloads_root / slug
    item_dir.mkdir(parents=True, exist_ok=True)

    # Create a zip file with a single file under the guessed name
    zip_path = item_dir / "single.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("inner.bin", b"BIN")

    # Monkeypatch urlopen to explode if called (shouldn't download again)
    monkeypatch.setattr(
        "biocracker.helpers.urlopen", lambda req: (_ for _ in ()).throw(AssertionError("No re-download"))
    )

    out = download_and_prepare(url, cache_dir=tmp_path)
    assert out.is_file()
    assert out.name == "inner.bin"
    assert out.read_bytes() == b"BIN"
    assert (item_dir / ".READY").exists()


def test_collapse_singleton_and_resolve(tmp_path: Path) -> None:
    """
    Test collapsing a singleton directory structure and resolving the return path.

    :param tmp_path: Path, a temporary directory provided by pytest
    """
    # Build nested single-dir tree with one file
    root = tmp_path / "root"
    d1 = root / "A"
    d2 = d1 / "B"
    d2.mkdir(parents=True)
    f = d2 / "only.txt"
    f.write_text("x")

    collapsed = _collapse_singleton(root)
    # root has single child dir A -> collapse to A (since root itself is not "extracted")
    # but our function descends only when path.is_dir() and looks at children;
    # since root contains only dir A -> _collapse_singleton(root) collapses to A, then to B
    assert collapsed == d2  # fully collapsed to the deepest singleton dir

    # Now test resolve: make structure like item_dir/extracted/...
    item_dir = tmp_path / "item"
    extracted = item_dir / "extracted"
    (extracted / "A" / "B").mkdir(parents=True)
    inner = extracted / "A" / "B" / "only.txt"
    inner.write_text("y")

    ret = _resolve_return_path(item_dir)
    assert ret.is_file()
    assert ret.read_text() == "y"
