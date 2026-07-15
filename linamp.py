import os
import sys
import locale
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
import json
import tempfile
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor

try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Gst', '1.0')
except ImportError as e:
    raise RuntimeError("GTK4 or GStreamer not installed. Please install them to run this application.") from e
from gi.repository import Gtk  # noqa: E402
from gi.repository import Gio, GLib, Gst, Gdk, Pango, GdkPixbuf  # noqa: E402

os.environ['LC_ALL'] = 'C'
os.environ['LANG'] = 'C'
os.environ['GTK_DISABLE_SETLOCALE'] = '1'

# Evo renderer (new clean visualizer)
try:
    from evo_visualizer import EvoRenderer, EvoConfig
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False
    EvoRenderer = None


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    locale.setlocale(locale.LC_ALL, 'C')
except locale.Error:
    pass

os.environ['GTK_CSD'] = '1'

def _get_default_screen_size():
    """Return (width, height) of the default display's bounding box."""
    display = Gdk.Display.get_default()
    if display and hasattr(display, "get_geometry"):
        geometry = display.get_geometry()
        if geometry and geometry.width > 0 and geometry.height > 0:
            return geometry.width, geometry.height
    return None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

if not NUMPY_AVAILABLE and any(name in sys.argv[0].lower() for name in ['visualizer', 'projectm']):
    sys.exit(1)

try:
    from metadata_fixer import MetadataFixer
    METADATA_FIXER_AVAILABLE = True
except ImportError:
    MetadataFixer = None
    METADATA_FIXER_AVAILABLE = False

COVER_ART_DEPENDENCIES = {}

try:
    import requests
    from PIL import Image, ImageEnhance
    from io import BytesIO
    REQUESTS_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    PIL_AVAILABLE = False
    requests = None

COVER_ART_DEPENDENCIES["requests"] = {
    "available": REQUESTS_AVAILABLE,
    "version": getattr(requests, "__version__", None) if requests else None,
}

COVER_ART_DEPENDENCIES["pillow"] = {
    "available": PIL_AVAILABLE,
    "version": getattr(Image, "__version__", None)
    if PIL_AVAILABLE else None,
}

try:
    import mutagen
    COVER_ART_DEPENDENCIES["mutagen"] = {
        "available": True,
        "version": getattr(
            mutagen,
            "__version__",
            getattr(mutagen, "version_string", "unknown")
        ),
    }
except Exception as e:
    COVER_ART_DEPENDENCIES["mutagen"] = {
        "available": False,
        "version": None,
        "error": str(e),
    }

for module_name in ("Gtk", "Gdk", "GdkPixbuf"):
    try:
        module = __import__(
            "gi.repository",
            fromlist=[module_name]
        ).__dict__[module_name]
        version = None
        if module_name == "Gtk":
            version = (
                f"{module.get_major_version()}."
                f"{module.get_minor_version()}."
                f"{module.get_micro_version()}"
            )
        COVER_ART_DEPENDENCIES[module_name.lower()] = {
            "available": True,
            "version": version or "available",
        }
    except Exception as e:
        COVER_ART_DEPENDENCIES[module_name.lower()] = {
            "available": False,
            "version": None,
            "error": str(e),
        }

COVER_DOWNLOAD_AVAILABLE = REQUESTS_AVAILABLE and PIL_AVAILABLE
COVER_ART_AVAILABLE = (
    COVER_DOWNLOAD_AVAILABLE
    and COVER_ART_DEPENDENCIES["gtk"]["available"]
    and COVER_ART_DEPENDENCIES["gdkpixbuf"]["available"]
)

DEFAULT_VOLUME = 0.7
BEAT_DETECTION_INTERVAL = 50
BEAT_HISTORY_SIZE = 10
UPDATE_DISPLAY_INTERVAL = 100
AUTO_SAVE_DELAY = 2000
PERIODIC_SAVE_INTERVAL = 30000
AUDIO_EXTENSIONS = {'.mp3', '.mp4', '.flac', '.ogg', '.wav', '.m4a', '.wma', '.aac', '.opus'}
COVER_ART_PATTERNS = [
    "cover.jpg", "cover.jpeg", "cover.png", "cover.webp", "cover.gif", "cover.bmp",
    "folder.jpg", "folder.jpeg", "folder.png", "folder.webp", "folder.gif", "folder.bmp",
    "album.jpg", "album.jpeg", "album.png", "album.webp", "album.gif", "album.bmp",
    "art.jpg", "art.jpeg", "art.png", "art.webp", "art.gif", "art.bmp",
    "front.jpg", "front.jpeg", "front.png", "front.webp", "front.gif", "front.bmp",
    ".cover.jpg", ".cover.jpeg", ".cover.png", ".cover.webp", ".cover.gif", ".cover.bmp"
]

SETTINGS_FILE = Path(__file__).parent / "linamp_settings.json"
COVER_CACHE_DIR = Path.home() / ".cache" / "linamp" / "covers"

def validate_cover_art_cache():
    try:
        COVER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        test_file = COVER_CACHE_DIR / ".test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return True
        except PermissionError:
            return False
    except (OSError, IOError):
        return False

COVER_CACHE_VALID = validate_cover_art_cache()

@dataclass
class CoverArt:
    path: str = ""
    source: str = "embedded"
    width: int = 0
    height: int = 0
    format: str = "unknown"
    bytes_data: Optional[bytes] = None

    def __post_init__(self):
        if not COVER_CACHE_VALID:
            return
        if self.bytes_data is None and self.path:
            try:
                with open(self.path, 'rb') as f:
                    self.bytes_data = f.read()
            except (OSError, IOError):
                self.bytes_data = b''

    @staticmethod
    def get_cache_key(audio_path: str) -> str:
        try:
            stat = os.stat(audio_path)
            file_data = f"{audio_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(file_data.encode()).hexdigest()[:16]
        except (OSError, IOError):
            return hashlib.md5(audio_path.encode()).hexdigest()[:16]

    @staticmethod
    def normalize_format(format_str: str) -> str:
        if not format_str:
            return "jpg"
        format_str = format_str.lower().strip()
        if format_str.startswith('image/'):
            format_str = format_str[6:]
        format_map = {
            'jpeg': 'jpg',
            'jpg': 'jpg',
            'png': 'png',
            'webp': 'webp',
            'gif': 'gif',
            'bmp': 'bmp',
            'ico': 'ico',
            'tiff': 'tiff',
            'tif': 'tiff'
        }
        return format_map.get(format_str, 'jpg')

    def get_cache_path(self, audio_path: str) -> Optional[Path]:
        if not COVER_CACHE_VALID:
            return None
        try:
            key = self.get_cache_key(audio_path)
            ext = self.normalize_format(self.format)
            cache_path = COVER_CACHE_DIR / f"{key}.{ext}"
            return cache_path
        except (OSError, IOError):
            return None

    def save_to_cache(self, audio_path: str) -> Optional[str]:
        if not self.bytes_data:
            return None
        if not COVER_CACHE_VALID:
            return None
        try:
            cache_path = self.get_cache_path(audio_path)
            if not cache_path:
                return None
            with open(cache_path, 'wb') as f:
                f.write(self.bytes_data)
            return str(cache_path)
        except (OSError, IOError):
            return None

    @staticmethod
    def get_cache_path_for_audio(audio_path: str) -> Optional[Path]:
        if not COVER_CACHE_VALID:
            return None
        try:
            key = CoverArt.get_cache_key(audio_path)
            cache_files = list(COVER_CACHE_DIR.glob(f"{key}.*"))
            if cache_files:
                return cache_files[0] if cache_files else None
        except (OSError, IOError):
            return None

    @staticmethod
    def get_cached_cover_path(audio_path: str) -> Optional[Path]:
        return CoverArt.get_cache_path_for_audio(audio_path)

    @classmethod
    def load_from_cache(cls, audio_path: str) -> Optional['CoverArt']:
        if not COVER_CACHE_VALID:
            return None
        try:
            cache_path = cls.get_cache_path_for_audio(audio_path)
            if cache_path and cache_path.exists():
                with open(cache_path, 'rb') as f:
                    data = f.read()
                ext = cache_path.suffix.lower().lstrip('.')
                return cls(
                    path=str(cache_path),
                    source="cache",
                    bytes_data=data,
                    format=ext
                )
        except (OSError, IOError):
            logger.debug("Failed to load cover art from cache", exc_info=True)
        return None

def _tag_key_matches(key, prefix):
    if isinstance(key, str):
        return key.startswith(prefix)

    if isinstance(key, tuple):
        return any(
            isinstance(item, str) and item.startswith(prefix)
            for item in key
        )

    return str(key).startswith(prefix)


def _extract_cover_art_from_file(file_path: str) -> Optional[CoverArt]:
    if not COVER_ART_DEPENDENCIES.get('mutagen', {}).get('available', False):
        return None
    from mutagen import File
    audio = File(file_path)
    if audio is None:
        return None

    def _make_cover(data, fmt='jpg'):
        return CoverArt(
            path="",
            source="embedded",
            bytes_data=data,
            format=CoverArt.normalize_format(fmt)
        )

    def _try_prefix_tags():
        if not hasattr(audio, 'tags') or not audio.tags:
            return None
        for tag_name in ['APIC:', 'APIC', 'cover', 'Cover', 'albumart', 'coverart']:
            if tag_name.endswith(':'):
                for key in audio.tags:
                    if _tag_key_matches(key, tag_name):
                        tag = audio.tags[key]
                        if hasattr(tag, 'data'):
                            return _make_cover(tag.data, getattr(tag, 'mime', 'jpg'))
                        elif hasattr(tag, 'value'):
                            return _make_cover(tag.value)
            else:
                if tag_name in audio.tags:
                    tag = audio.tags[tag_name]
                    if hasattr(tag, 'data'):
                        return _make_cover(tag.data, getattr(tag, 'mime', 'jpg'))
                    elif hasattr(tag, 'value'):
                        return _make_cover(tag.value)
        return None

    def _try_covr():
        if not (hasattr(audio, 'tags') and audio.tags and 'covr' in audio.tags):
            return None
        covr_data = audio.tags['covr']
        if isinstance(covr_data, list) and covr_data:
            cover_image = covr_data[0]
            if hasattr(cover_image, 'data'):
                return _make_cover(cover_image.data, getattr(cover_image, 'mime', 'jpg'))
            elif isinstance(cover_image, bytes):
                return _make_cover(cover_image)
            return None

    def _try_pictures():
            if hasattr(audio, 'pictures') and audio.pictures:
                picture = audio.pictures[0]
                if hasattr(picture, 'data') and picture.data:
                    return _make_cover(picture.data, getattr(picture, 'mime', 'jpg'))
            return None

    def _try_alternatives():
            methods = [
                ('artwork', 'artwork'),
                ('cover', 'cover'),
                ('coverart', 'coverart'),
                ('albumart', 'albumart'),
                ('covers', 'covers'),
                ('pictures', 'pictures'),
                ('covr', 'covr')
            ]
            for attr_name, _ in methods:
                if hasattr(audio, attr_name) and getattr(audio, attr_name):
                    cover_data = getattr(audio, attr_name)
                    if isinstance(cover_data, list) and cover_data:
                        item = cover_data[0]
                        if hasattr(item, 'data'):
                            return _make_cover(item.data, getattr(item, 'mime', 'jpg'))
                        elif isinstance(item, bytes):
                            return _make_cover(item)
                    elif isinstance(cover_data, bytes):
                        return _make_cover(cover_data)
            return None

    return (
        _try_prefix_tags()
        or _try_covr()
        or _try_pictures()
        or _try_alternatives()
    )


def _find_cover_in_folder(audio_path: str) -> Optional[CoverArt]:
    audio_dir = os.path.dirname(audio_path)
    if not audio_dir:
        return None
    files_in_dir = []
    all_files = os.listdir(audio_dir)
    for f in all_files:
        file_path = os.path.join(audio_dir, f)
        if os.path.isfile(file_path):
            files_in_dir.append(f.lower())
    for pattern in COVER_ART_PATTERNS:
        pattern_lower = pattern.lower()
        if pattern_lower in files_in_dir:
            for actual_file in os.listdir(audio_dir):
                if actual_file.lower() == pattern_lower:
                    cover_path = os.path.join(audio_dir, actual_file)
                    with open(cover_path, 'rb') as f:
                        data = f.read()
                    ext = pattern.split('.')[-1]
                    return CoverArt(
                        path=cover_path,
                        source="file",
                        bytes_data=data,
                        format=ext
                    )
    return None

def extract_cover_art(audio_path: str, allow_download: bool = False) -> Tuple[Optional[CoverArt], Optional[str]]:
    if not os.path.exists(audio_path):
        return None, None
    cached = CoverArt.load_from_cache(audio_path)
    if cached:
        return cached, cached.path if cached.path else None
    cover = _extract_cover_art_from_file(audio_path)
    if cover:
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover, cache_path
    cover = _find_cover_in_folder(audio_path)
    if cover:
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover, cache_path
    return None, None

class CoverArtDownloader:
    def __init__(self):
        if not COVER_DOWNLOAD_AVAILABLE:
            raise ImportError("requests and PIL are required for cover downloading")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        self.timeout = (3, 5)

    def get_metadata(self, file_path: str) -> Tuple[str, str]:
        try:
            if not COVER_ART_DEPENDENCIES.get('mutagen', {}).get('available'):
                return '', ''
            audio_path = Path(file_path)
            if audio_path.suffix.lower() == '.mp3':
                from mutagen.mp3 import MP3
                from mutagen.id3 import ID3
                audio = MP3(file_path, ID3=ID3)
                artist = audio.tags.get('TPE1', [''])[0] if audio.tags else ''
                album = audio.tags.get('TALB', [''])[0] if audio.tags else ''
            elif audio_path.suffix.lower() == '.flac':
                from mutagen.flac import FLAC
                audio = FLAC(file_path)
                artist = audio.get('artist', [''])[0]
                album = audio.get('album', [''])[0]
            elif audio_path.suffix.lower() in ['.m4a', '.mp4']:
                from mutagen.mp4 import MP4
                audio = MP4(file_path)
                artist = audio.get('\xa9ART', [''])[0] if audio.get('\xa9ART') else ''
                album = audio.get('\xa9alb', [''])[0] if audio.get('\xa9alb') else ''
            else:
                return '', ''
            return artist.strip(), album.strip()
        except Exception as e:
            logger.debug("get_metadata failed for %s: %s", audio_path, e, exc_info=True)
            return '', ''

    def search_cover(self, artist: str, album: str) -> Optional[str]:
        if not artist or not album:
            return None
        query = f"{artist} {album} album cover"
        sources = [
            self._search_deezer,
            self._search_lastfm,
        ]
        for source_func in sources:
            try:
                url = source_func(query, artist, album)
                if url:
                    return url
                time.sleep(0.5)
            except Exception as e:
                logger.debug("Cover search source failed: %s", e, exc_info=True)
                continue
        return None

    def _search_deezer(self, query: str, artist: str, album: str) -> Optional[str]:
        try:
            from urllib.parse import quote
            search_url = f"https://api.deezer.com/search/album?q={quote(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and len(data['data']) > 0:
                    album_data = data['data'][0]
                    cover_url = album_data.get('cover_xl') or album_data.get('cover_big') or album_data.get('cover_medium')
                    if cover_url:
                        return cover_url
        except Exception:
            return None

    def _search_google_images(self, query: str, artist: str, album: str) -> Optional[str]:
        try:
            from urllib.parse import quote
            import re
            search_url = f"https://www.google.com/search?tbm=isch&q={quote(query)}"
            response = self.session.get(search_url, timeout=self.timeout)
            if response.status_code == 200:
                pattern = r'\["(https://[^"]+?\.(?:jpg|jpeg|png|webp))",\d+,\d+\]'
                matches = re.findall(pattern, response.text)
                if matches:
                    for url in matches[:3]:
                        if self._is_likely_album_cover(url):
                            return url
        except Exception as e:
            logger.debug("Deezer cover search failed: %s", e, exc_info=True)
            return None

    def _search_lastfm(self, query: str, artist: str, album: str) -> Optional[str]:
        try:
            from urllib.parse import quote
            search_url = f"https://www.last.fm/music/{quote(artist)}/{quote(album)}"
            response = self.session.get(search_url, timeout=self.timeout)
            if response.status_code == 200:
                import re
                pattern = r'https://lastfm\.freetls\.fastly\.net/i/u/[^"]+\.(?:jpg|jpeg|png)'
                matches = re.findall(pattern, response.text)
                if matches:
                    return sorted(matches, key=len, reverse=True)[0]
        except Exception as e:
            logger.debug("Last.fm cover search failed: %s", e, exc_info=True)
            return None

    def _is_likely_album_cover(self, url: str) -> bool:
        exclude_patterns = ['icon', 'logo', 'avatar', 'thumb', 'small', 'tiny']
        url_lower = url.lower()
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        return True

    def download_cover(self, url: str) -> Optional[bytes]:
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            img = Image.open(BytesIO(response.content))
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            img.thumbnail((500, 500), Image.Resampling.LANCZOS)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)
            output = BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()
        except Exception as e:
            logger.debug("Cover download failed: %s", e, exc_info=True)
            return None

_downloader = None

def get_cover_downloader() -> Optional[CoverArtDownloader]:
    global _downloader
    if _downloader is None and COVER_DOWNLOAD_AVAILABLE:
        try:
            _downloader = CoverArtDownloader()
        except Exception as e:
            logger.warning("Could not initialize cover downloader: %s", e)
    return _downloader

def download_cover_for_audio(audio_path: str) -> Optional[CoverArt]:
    downloader = get_cover_downloader()
    if not downloader:
        return None
    try:
        artist, album = downloader.get_metadata(audio_path)
        if not artist or not album:
            return None
        cover_url = downloader.search_cover(artist, album)
        if not cover_url:
            return None
        cover_data = downloader.download_cover(cover_url)
        if not cover_data:
            return None
        cover = CoverArt(
            path="",
            source="downloaded",
            bytes_data=cover_data,
            format="jpg"
        )
        cache_path = cover.save_to_cache(audio_path)
        if cache_path:
            cover.path = cache_path
        return cover
    except Exception as e:
        logger.debug("download_cover_for_audio failed: %s", e, exc_info=True)
        return None

@dataclass
class PlaylistItem:
    path: str
    title: str = ""
    duration: int = 0
    cover_art_path: str = ""
    id: str = ""

    def __post_init__(self):
        self.path = os.path.abspath(os.path.expanduser(str(self.path)))
        if not self.title:
            self.title = os.path.basename(self.path)
        if not self.id:
            self.id = hashlib.md5(f"{self.path}:{self.title}".encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'title': self.title,
            'duration': self.duration,
            'cover_art_path': self.cover_art_path,
            'id': self.id
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlaylistItem':
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        item_data = data.copy()
        if 'filename' in item_data and 'path' not in item_data:
            item_data['path'] = item_data.pop('filename')
        if 'path' not in item_data:
            raise ValueError("Playlist item must contain a 'path' or 'filename' field")
        return cls(**item_data)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def get_display_name(self) -> str:
        return self.title if self.title else os.path.basename(self.path)

    def has_cover_art(self) -> bool:
        if self.cover_art_path and os.path.exists(self.cover_art_path):
            return True
        return False

class CoverArtWidget(Gtk.Box):

    def __init__(self, player, size: int = 150):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.size = size
        self.current_cover_path = None
        self.loading = False
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CoverArtWidget")
        self.set_size_request(size, size)
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.image = Gtk.Image()
        self.image.set_size_request(size, size)
        self.image.set_halign(Gtk.Align.CENTER)
        self.image.set_valign(Gtk.Align.CENTER)
        self.image.set_icon_size(Gtk.IconSize.LARGE)
        self.append(self.image)
        self.show_placeholder()

    def show_placeholder(self):
        self.image.set_from_icon_name("folder-music")
        self.image.set_pixel_size(self.size // 2)
        self.loading = False

    def _create_texture_from_bytes(self, bytes_data: bytes) -> Optional[Gdk.Texture]:
        """Create a Gdk.Texture from raw image bytes, scaled to self.size."""
        loader = GdkPixbuf.PixbufLoader()
        loader.write(bytes_data)
        loader.close()
        pixbuf = loader.get_pixbuf()
        if not pixbuf:
            return None
        width = pixbuf.get_width()
        height = pixbuf.get_height()
        if width > height:
            scale = self.size / width
        else:
            scale = self.size / height
        new_width = int(width * scale)
        new_height = int(height * scale)
        scaled = pixbuf.scale_simple(
            new_width, new_height, GdkPixbuf.InterpType.BILINEAR
        )
        data = scaled.save_to_bufferv("png", [], [])
        if not data:
            return None
        return Gdk.Texture.new_from_bytes(GLib.Bytes.new(data))

    def load_cover(self, cover_art, audio_path: str = ""):
        if not cover_art:
            self.show_placeholder()
            return
        try:
            if cover_art.path and os.path.exists(cover_art.path):
                try:
                    texture = Gdk.Texture.new_from_filename(cover_art.path)
                    self.image.set_from_paintable(texture)
                    self.image.set_size_request(self.size, self.size)
                    self.image.set_halign(Gtk.Align.FILL)
                    self.image.set_valign(Gtk.Align.FILL)
                    self.current_cover_path = cover_art.path
                except GLib.GError:
                    if cover_art.bytes_data:
                        texture = self._create_texture_from_bytes(cover_art.bytes_data)
                        if texture:
                            self.image.set_from_paintable(texture)
                            self.current_cover_path = audio_path
                        else:
                            self.show_placeholder()
                    else:
                        self.show_placeholder()
            elif cover_art.bytes_data:
                texture = self._create_texture_from_bytes(cover_art.bytes_data)
                if texture:
                    self.image.set_from_paintable(texture)
                    self.current_cover_path = audio_path
        except Exception:
            self.show_placeholder()

    def update_settings(self):
        settings = self.player.settings
        if hasattr(settings, 'cover_art_shape'):
            self._apply_shape()

    def _apply_shape(self):
        settings = self.player.settings
        if not hasattr(settings, 'cover_art_shape'):
            return
        shape = settings.cover_art_shape.lower()
        for css_class in ['cover-square', 'cover-circle', 'cover-rounded']:
            self.remove_css_class(css_class)
        if shape == 'circle':
            self.add_css_class('cover-circle')
        elif shape == 'rounded':
            self.add_css_class('cover-rounded')
        elif shape == 'square':
            self.add_css_class('cover-square')
        self.queue_draw()

    def set_size(self, size: int):
        if size < 50:
            size = 50
        elif size > 800:
            size = 800
        self.size = size
        self.set_size_request(size, size)
        self.image.set_size_request(size, size)

    def load_cover_from_path(self, cover_path):
        if not cover_path or not os.path.exists(cover_path):
            self.show_placeholder()
            return
        try:
            texture = Gdk.Texture.new_from_filename(str(cover_path))
            self.image.set_from_paintable(texture)
            self.image.set_size_request(self.size, self.size)
            self.image.set_halign(Gtk.Align.FILL)
            self.image.set_valign(Gtk.Align.FILL)
            self.current_cover_path = str(cover_path)
        except Exception:
            self.show_placeholder()

    def clear(self):
        self.current_cover_path = None
        self.show_placeholder()

    def cleanup(self):
        try:
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None
        except Exception:
            pass
@dataclass
class PlayerSettings:
    auto_play_next: bool = True
    shuffle_mode: bool = False
    repeat_mode: str = "none"
    beat_aware_enabled: bool = True
    beat_threshold: float = 0.1
    auto_download_covers: bool = False
    volume: float = 1.0
    position: float = 0.0
    window_size: Tuple[int, int] = (600, 450)
    window_position: Tuple[int, int] = (100, 100)
    equalizer_settings: List[float] = field(default_factory=lambda: [0.0] * 10)
    last_played_track: str = ""
    last_played_position: float = 0.0
    show_cover_art: bool = True
    cover_art_size: int = 650
    cover_art_shape: str = "square"

    def __post_init__(self):
        if self.equalizer_settings is None:
            self.equalizer_settings = [0.0] * 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            'auto_play_next': self.auto_play_next,
            'shuffle_mode': self.shuffle_mode,
            'repeat_mode': self.repeat_mode,
            'beat_aware_enabled': self.beat_aware_enabled,
            'beat_threshold': self.beat_threshold,
            'auto_download_covers': self.auto_download_covers,
            'volume': self.volume,
            'position': self.position,
            'window_size': self.window_size,
            'window_position': self.window_position,
            'equalizer_settings': self.equalizer_settings,
            'last_played_track': self.last_played_track,
            'last_played_position': self.last_played_position,
            'show_cover_art': self.show_cover_art,
            'cover_art_size': self.cover_art_size,
            'cover_art_shape': self.cover_art_shape
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlayerSettings':
        return cls(
            auto_play_next=data.get('auto_play_next', True),
            shuffle_mode=data.get('shuffle_mode', False),
            repeat_mode=data.get('repeat_mode', 'none'),
            beat_aware_enabled=data.get('beat_aware_enabled', True),
            beat_threshold=data.get('beat_threshold', 0.1),
            auto_download_covers=data.get('auto_download_covers', False),
            volume=data.get('volume', 1.0),
            position=data.get('position', 0.0),
            window_size=tuple(data.get('window_size', (400, 300))),
            window_position=tuple(data.get('window_position', (100, 100))),
            equalizer_settings=data.get('equalizer_settings', None),
            last_played_track=data.get('last_played_track', ''),
            last_played_position=data.get('last_played_position', 0.0),
            show_cover_art=data.get('show_cover_art', True),
            cover_art_size=data.get('cover_art_size', 650),
            cover_art_shape=data.get('cover_art_shape', 'square')
        )

class EqualizerTab(Gtk.Box):
    GRID_COLUMNS = 4
    EQ_PRESETS_PATH = Path(__file__).parent / "configs" / "eq_presets.json"

    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("eq-tab")
        self.bands = 10
        self.band_scales = []
        self.eq_presets = self._load_eq_presets()
        self.eq_presets_map = {p["name"]: p["values"] for p in self.eq_presets}
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        self.append(main_box)
        header_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(header_section)
        header_label = Gtk.Label(label="10-Band Equalizer")
        header_label.add_css_class("section-header")
        header_label.set_halign(Gtk.Align.CENTER)
        header_section.append(header_label)
        bands_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.append(bands_container)
        freq_labels_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        freq_labels_row.set_halign(Gtk.Align.CENTER)
        bands_container.append(freq_labels_row)
        freqs = ["60", "170", "310", "600", "1k", "3k", "6k", "12k", "14k", "16k"]
        for freq in freqs:
            freq_label = Gtk.Label(label=freq)
            freq_label.add_css_class("eq-label")
            freq_label.set_size_request(35, -1)
            freq_label.set_halign(Gtk.Align.CENTER)
            freq_labels_row.append(freq_label)
        sliders_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sliders_row.set_halign(Gtk.Align.CENTER)
        bands_container.append(sliders_row)
        for i in range(self.bands):
            band_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            band_container.add_css_class("eq-band")
            band_container.set_size_request(50, 180)
            value_label = Gtk.Label(label="0 dB")
            value_label.add_css_class("eq-value-label")
            value_label.set_size_request(40, -1)
            value_label.set_halign(Gtk.Align.CENTER)
            band_container.append(value_label)
            lower, upper = -12, 12
            value = max(lower, min(upper, 0))
            adjustment = Gtk.Adjustment(
                value=value, lower=lower, upper=upper, step_increment=0.1,
                page_increment=1, page_size=0
            )
            scale = Gtk.Scale(orientation=Gtk.Orientation.VERTICAL, adjustment=adjustment)
            scale.set_inverted(True)
            scale.set_draw_value(False)
            scale.set_size_request(40, 140)
            scale.set_margin_top(4)
            scale.set_margin_bottom(4)
            scale.add_css_class("eq-scale")
            scale.connect("value-changed", self.on_band_changed, i, value_label)
            scale.add_mark(0, Gtk.PositionType.LEFT, "0")
            scale.add_mark(-12, Gtk.PositionType.LEFT, "-12")
            scale.add_mark(12, Gtk.PositionType.LEFT, "+12")
            band_container.append(scale)
            sliders_row.append(band_container)
            self.band_scales.append((scale, value_label))
        presets_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.append(presets_section)
        presets_header = Gtk.Label(label="Presets")
        presets_header.add_css_class("section-header")
        presets_header.set_halign(Gtk.Align.START)
        presets_section.append(presets_header)
        presets_grid = Gtk.Grid()
        presets_grid.set_column_spacing(8)
        presets_grid.set_row_spacing(8)
        presets_grid.set_halign(Gtk.Align.CENTER)
        presets_section.append(presets_grid)
        presets = []
        for idx, preset in enumerate(self.eq_presets):
            col = idx % self.GRID_COLUMNS
            row = idx // self.GRID_COLUMNS
            presets.append((preset["name"], col, row))
        for preset_name, col, row in presets:
            btn = self._create_preset_button(preset_name)
            btn.connect("clicked", self.on_preset_clicked, preset_name)
            presets_grid.attach(btn, col, row, 1, 1)
        reset_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        reset_container.set_halign(Gtk.Align.CENTER)
        reset_container.set_margin_top(12)
        presets_section.append(reset_container)
        reset_btn = self._create_preset_button("Reset to Flat")
        reset_btn.connect("clicked", self.on_reset_clicked)
        reset_container.append(reset_btn)
        self.load_saved_settings()

    def _load_eq_presets(self):
        defaults = [
            {"name": "Flat", "values": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"name": "Pop", "values": [4, 3, 2, 1, 0, -1, -2, -2, -2, -2]},
            {"name": "Rock", "values": [6, 4, 2, 0, -2, -3, -3, -3, -3, -3]},
            {"name": "Jazz", "values": [3, 2, 1, 1, 0, -1, -2, -2, -2, -2]},
            {"name": "Classical", "values": [4, 3, 2, 1, 0, -1, -2, -3, -4, -5]},
            {"name": "Electronic", "values": [5, 4, 3, 1, 0, 1, 2, 3, 4, 5]},
            {"name": "Hip-Hop", "values": [5, 4, 2, 0, -1, 1, 3, 4, 5, 5]},
            {"name": "Metal", "values": [7, 6, 5, 3, 1, -1, -2, -3, -3, -3]},
            {"name": "Acoustic", "values": [-2, -1, 0, 2, 4, 4, 3, 2, 1, 0]},
            {"name": "Vocal", "values": [-1, 0, 2, 3, 4, 3, 2, 1, 0, -1]},
            {"name": "Bass Boost", "values": [8, 7, 6, 4, 2, 0, -1, -2, -3, -4]},
            {"name": "Treble Boost", "values": [-4, -3, -2, 0, 2, 4, 6, 7, 8, 8]}
        ]
        try:
            with open(self.EQ_PRESETS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = []
            seen = set()
            for item in data:
                name = item.get("name", "")
                values = item.get("values", [])
                if len(values) != 10 or not name or name in seen:
                    continue
                seen.add(name)
                loaded.append({"name": name, "values": [float(v) for v in values]})
            return loaded if loaded else defaults
        except (FileNotFoundError, json.JSONDecodeError):
            return defaults

    def load_saved_settings(self):
        if hasattr(self.player, 'settings') and self.player.settings.equalizer_settings:
            try:
                for i, value in enumerate(self.player.settings.equalizer_settings[:10]):
                    if i < len(self.band_scales):
                        scale, value_label = self.band_scales[i]
                        clamped_value = max(-12, min(12, value))
                        scale.set_value(clamped_value)
                        value_label.set_text(f"{clamped_value:+.1f} dB")
            except Exception:
                pass

    def _create_preset_button(self, text):
        button = Gtk.Button(label=text)
        button.add_css_class("eq-preset-btn")
        button.set_size_request(80, 28)
        return button

    def on_band_changed(self, scale, band, value_label):
        if hasattr(self.player, 'equalizer'):
            value = scale.get_value()
            clamped_value = max(-12, min(12, value))
            if clamped_value != value:
                scale.set_value(clamped_value)
                value = clamped_value
            self.player.equalizer.set_property(f'band{band}', value)
            value_label.set_text(f"{value:+.1f} dB")
            if not hasattr(self, '_eq_save_timer_id'):
                self._eq_save_timer_id = None
            if self._eq_save_timer_id is not None:
                try:
                    GLib.source_remove(self._eq_save_timer_id)
                except Exception:
                    pass
            def _debounced_save():
                self._eq_save_timer_id = None
                if hasattr(self.player, 'auto_save_settings'):
                    self.player.auto_save_settings()
                return False
            self._eq_save_timer_id = GLib.timeout_add(1000, _debounced_save)

    def on_reset_clicked(self, button):
        for scale, value_label in self.band_scales:
            scale.set_value(0)
        if hasattr(self.player, 'auto_save_settings'):
            self.player.auto_save_settings()

    def on_preset_clicked(self, button, preset_name):
        if preset_name in self.eq_presets_map:
            values = self.eq_presets_map[preset_name]
            for i, (scale, value_label) in enumerate(self.band_scales):
                scale.set_value(values[i])
            if hasattr(self.player, 'auto_save_settings'):
                self.player.auto_save_settings()

class PlaylistTab(Gtk.Box):
    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.add_css_class("playlist-tab")
        self.playlist_store = Gtk.StringList()
        self.filtered_store = None
        self.filter_model = None
        self._search_debounce_id = None
        self._playlist_by_id = {}
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        self.append(main_box)
        search_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(search_section)
        search_header = Gtk.Label(label="Search Playlist")
        search_header.add_css_class("section-header")
        search_header.set_halign(Gtk.Align.START)
        search_section.append(search_header)
        search_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        search_section.append(search_container)
        self.search_entry = Gtk.Entry()
        self.search_entry.set_placeholder_text("Type to filter playlist...")
        self.search_entry.set_hexpand(True)
        self.search_entry.add_css_class("search-entry")
        self.search_entry.connect("changed", self.on_search_changed)
        search_container.append(self.search_entry)
        clear_search_btn = self._create_modern_button("Clear", "edit-clear-symbolic")
        clear_search_btn.connect("clicked", self.on_clear_search)
        search_container.append(clear_search_btn)
        toolbar_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.append(toolbar_section)
        toolbar_header = Gtk.Label(label="Playlist Actions")
        toolbar_header.add_css_class("section-header")
        toolbar_header.set_halign(Gtk.Align.START)
        toolbar_section.append(toolbar_header)
        main_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        main_toolbar.set_homogeneous(True)
        toolbar_section.append(main_toolbar)
        add_btn = self._create_modern_button("Add Files", "list-add-symbolic")
        add_btn.connect("clicked", self.on_add_files)
        main_toolbar.append(add_btn)
        add_folder_btn = self._create_modern_button("Add Folder", "folder-open-symbolic")
        add_folder_btn.connect("clicked", self.on_add_folder)
        main_toolbar.append(add_folder_btn)
        remove_btn = self._create_modern_button("Remove", "list-remove-symbolic")
        remove_btn.connect("clicked", self.on_remove)
        main_toolbar.append(remove_btn)
        clear_btn = self._create_modern_button("Clear All", "edit-delete-symbolic")
        clear_btn.connect("clicked", self.on_clear)
        main_toolbar.append(clear_btn)
        shuffle_btn = self._create_modern_button("Shuffle", "media-playlist-shuffle-symbolic")
        shuffle_btn.connect("clicked", self.on_shuffle)
        main_toolbar.append(shuffle_btn)
        secondary_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        secondary_toolbar.set_homogeneous(True)
        toolbar_section.append(secondary_toolbar)
        sort_dropdown = Gtk.DropDown.new_from_strings(["Default Order", "By Title", "By Path", "By Duration"])
        sort_dropdown.add_css_class("sort-dropdown")
        sort_dropdown.connect("notify::selected", self.on_sort_changed)
        secondary_toolbar.append(sort_dropdown)
        self.sort_dropdown = sort_dropdown
        import_btn = self._create_modern_button("Import", "document-open-symbolic")
        import_btn.connect("clicked", self.on_import_m3u)
        secondary_toolbar.append(import_btn)
        export_btn = self._create_modern_button("Export", "document-save-symbolic")
        export_btn.connect("clicked", self.on_export_m3u)
        secondary_toolbar.append(export_btn)
        remove_dups_btn = self._create_modern_button("Remove Dups", "view-refresh-symbolic")
        remove_dups_btn.connect("clicked", self.on_remove_duplicates)
        secondary_toolbar.append(remove_dups_btn)
        if METADATA_FIXER_AVAILABLE:
            fix_tags_btn = self._create_modern_button("Fix Tags", "document-edit-symbolic")
            fix_tags_btn.connect("clicked", self.on_fix_tags)
            secondary_toolbar.append(fix_tags_btn)
            self.fix_tags_btn = fix_tags_btn
        else:
            self.fix_tags_btn = None
        stats_container = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        stats_container.set_halign(Gtk.Align.END)
        stats_container.set_margin_top(8)
        toolbar_section.append(stats_container)
        self.stats_label = Gtk.Label(label="0 tracks")
        self.stats_label.add_css_class("stats-label")
        stats_container.append(self.stats_label)
        playlist_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        playlist_container.set_margin_top(12)
        main_box.append(playlist_container)
        playlist_header = Gtk.Label(label="Track List")
        playlist_header.add_css_class("section-header")
        playlist_header.set_halign(Gtk.Align.START)
        playlist_container.append(playlist_header)
        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC
        )
        scrolled.add_css_class("playlist-scrolled")
        scrolled.set_hexpand(True)
        scrolled.set_vexpand(True)
        playlist_container.append(scrolled)
        self.sort_model = Gtk.SortListModel(model=self.playlist_store)
        self.filter_model = Gtk.FilterListModel(model=self.sort_model)
        self.selection_model = Gtk.SingleSelection(model=self.filter_model)
        self.column_view = Gtk.ColumnView(model=self.selection_model)
        self.column_view.add_css_class("playlist-view")
        self.column_view.set_hexpand(True)
        self.column_view.set_vexpand(True)
        self.column_view.set_size_request(-1, 180)
        factory = Gtk.SignalListItemFactory()
        factory.connect('setup', self._on_factory_setup)
        factory.connect('bind', self._on_factory_bind)
        column = Gtk.ColumnViewColumn(title="Tracks", factory=factory)
        self.column_view.append_column(column)
        scrolled.set_child(self.column_view)
        self.selection_model.connect("selection-changed", self.on_selection_changed)
        self.column_view.connect("activate", self.on_row_activated)

    def _create_modern_button(self, text, icon_name=None):
        button = Gtk.Button()
        if icon_name:
            content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            icon = Gtk.Image.new_from_icon_name(icon_name)
            icon.set_pixel_size(14)
            content_box.append(icon)
            label = Gtk.Label(label=text)
            content_box.append(label)
            button.set_child(content_box)
        else:
            button.set_label(text)
        return button

    def on_add_files(self, button):
        try:
            dialog = Gtk.FileDialog(title="Add Files")
            dialog.set_modal(True)
            audio_filter = Gtk.FileFilter()
            audio_filter.set_name("Audio files")
            audio_filter.add_mime_type("audio/*")
            filters = Gio.ListStore.new(Gtk.FileFilter)
            filters.append(audio_filter)
            dialog.set_filters(filters)
            dialog.open_multiple(self.player, None, self._on_files_selected)
        except Exception as e:
            self.player.set_status_message(f"Failed to open file picker: {e}")

    def _on_files_selected(self, dialog, result):
        try:
            files = dialog.open_multiple_finish(result)
            file_paths = [file.get_path() for file in files if file.get_path() is not None]
            if file_paths:
                self.player.add_to_playlist(file_paths)
            else:
                self.player.set_status_message("No files were selected.")
        except GLib.Error as e:
            self.player.set_status_message(f"File selection error: {e.message if hasattr(e, 'message') else str(e)}")
        except Exception as e:
            self.player.set_status_message(f"Could not add files: {e}")

    def on_add_folder(self, button):
        try:
            dialog = Gtk.FileDialog(title="Add Music Folder")
            dialog.set_modal(True)
            dialog.select_folder(self.player, None, self._on_folder_selected)
        except Exception as e:
            self.player.set_status_message(f"Failed to open folder picker: {e}")

    def _on_folder_selected(self, dialog, result):
        try:
            folder = dialog.select_folder_finish(result)
            if folder:
                folder_path = folder.get_path()
                if folder_path:
                    self.player.add_folder_to_playlist(folder_path)
                else:
                    self.player.set_status_message("Folder selection did not return a valid path.")
        except Exception as e:
            self.player.set_status_message(f"Could not add folder: {e}")

    def on_remove(self, button):
        position = self.selection_model.get_selected()
        if position != Gtk.INVALID_LIST_POSITION:
            item = self.selection_model.get_item(position)
            if item is not None:
                stored_string = item.get_string()
                if '|' in stored_string:
                    track_title, track_id = stored_string.split('|', 1)
                else:
                    track_title = stored_string
                    track_id = None
                    for playlist_item in self.player.playlist:
                        if playlist_item.title == track_title or playlist_item.path.endswith(track_title):
                            track_id = playlist_item.id
                            break
                    if track_id is None:
                        track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                for i, playlist_item in enumerate(self.player.playlist):
                    if getattr(playlist_item, 'id', None) == track_id:
                        self.player.playlist.pop(i)
                        break
                self.player._update_playlist_display()
                self.update_statistics()

    def on_clear(self, button):
        while self.playlist_store.get_n_items() > 0:
            self.playlist_store.remove(0)
        self.player.playlist.clear()
        self.update_statistics()

    def _on_factory_setup(self, factory, list_item):
        label = Gtk.Label()
        label.set_halign(Gtk.Align.START)
        list_item.set_child(label)

    def _on_factory_bind(self, factory, list_item):
        label = list_item.get_child()
        item = list_item.get_item()
        if item is not None:
            text = item.get_string() if hasattr(item, 'get_string') else str(item)
            if '|' in text:
                text = text.split('|', 1)[0]
            label.set_text(text)

    def on_selection_changed(self, selection, position, n_items):
        selected_pos = selection.get_selected()
        if selected_pos != Gtk.INVALID_LIST_POSITION:
            item = self.selection_model.get_item(selected_pos)
            if item is not None:
                stored_string = item.get_string()
                if '|' in stored_string:
                    track_title, track_id = stored_string.split('|', 1)
                else:
                    track_title = stored_string
                    track_id = None
                    for playlist_item in self.player.playlist:
                        if playlist_item.title == track_title or playlist_item.path.endswith(track_title):
                            track_id = playlist_item.id
                            break
                    if track_id is None:
                        track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                for i, playlist_item in enumerate(self.player.playlist):
                    if playlist_item.id == track_id:
                        self.player.current_track = i
                        break

    def on_search_changed(self, entry):
        search_text = (entry.get_text() or "").strip().lower()
        if self._search_debounce_id is not None:
            try:
                GLib.source_remove(self._search_debounce_id)
            except Exception:
                pass
            self._search_debounce_id = None

        def apply_filter():
            try:
                if not search_text:
                    self.filter_model.set_filter(None)
                else:
                    try:
                        filter_obj = Gtk.CustomFilter.new(lambda item: self._filter_func(item, search_text))
                    except Exception:
                        filter_obj = Gtk.CustomFilter()
                        try:
                            filter_obj.set_filter_func(lambda item: self._filter_func(item, search_text))
                        except Exception:
                            filter_obj.set_filter(lambda item: self._filter_func(item, search_text))
                    self.filter_model.set_filter(filter_obj)
            finally:
                self._search_debounce_id = None
            return False
        self._search_debounce_id = GLib.timeout_add(120, apply_filter)

    def _filter_func(self, item, search_text):
        if item is None:
            return False
        raw = item.get_string() if hasattr(item, "get_string") else str(item)
        title, track_id = (raw.split("|", 1) + [""])[:2] if "|" in raw else (raw, "")
        terms = [t for t in search_text.split() if t]
        if not terms:
            return True
        haystacks = [title.lower()]
        if track_id:
            pitem = self._playlist_by_id.get(track_id)
            if pitem is not None:
                haystacks.append((pitem.path or "").lower())
        for term in terms:
            if not any(term in h for h in haystacks):
                return False
        return True

    def on_clear_search(self, button):
        self.search_entry.set_text("")
        self.filter_model.set_filter(None)

    def on_shuffle(self, button):
        if self.player.playlist:
            import random
            random.shuffle(self.player.playlist)
            self.player._update_playlist_display()
            self.player.save_playlist()

    def on_sort_changed(self, dropdown, pspec):
        selected = dropdown.get_selected()
        if selected == 0:
            pass
        elif selected == 1:
            self.player.playlist.sort(key=lambda x: x.title.lower())
        elif selected == 2:
            self.player.playlist.sort(key=lambda x: x.path.lower())
        elif selected == 3:
            self.player.playlist.sort(key=lambda x: x.duration)
        if selected != 0:
            self.player._update_playlist_display()
            self.player.save_playlist()

    def update_statistics(self):
        total_tracks = len(self.player.playlist)
        try:
            by_id = {}
            for item in self.player.playlist:
                if isinstance(item, PlaylistItem) and item.id:
                    by_id[item.id] = item
                elif isinstance(item, dict):
                    tid = item.get("id")
                    if tid:
                        by_id[tid] = PlaylistItem.from_dict(item)
            self._playlist_by_id = by_id
        except Exception:
            self._playlist_by_id = {}
        total_duration = 0
        for item in self.player.playlist:
            try:
                dur = item.duration if hasattr(item, "duration") else item.get("duration", 0)
                if dur and dur > 0:
                    total_duration += int(dur)
            except Exception:
                continue
        if total_duration > 0:
            hours = total_duration // 3600
            minutes = (total_duration % 3600) // 60
            if hours > 0:
                duration_text = f"{hours}h {minutes}m"
            else:
                duration_text = f"{minutes}m"
            stats_text = f"{total_tracks} tracks • {duration_text}"
        else:
            stats_text = f"{total_tracks} tracks"
        self.stats_label.set_text(stats_text)

    def on_import_m3u(self, button):
        dialog = Gtk.FileDialog(title="Import M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        m3u_filter.add_pattern("*.M3U")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.open(self.player, None, self._on_m3u_import_selected)

    def _on_m3u_import_selected(self, dialog, result):
        try:
            file = dialog.open_finish(result)
            if file:
                filepath = file.get_path()
                if filepath:
                    self._import_m3u_file(filepath)
        except GLib.Error:
            pass

    def _import_m3u_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            imported_files = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not os.path.isabs(line):
                    m3u_dir = os.path.dirname(filepath)
                    line = os.path.join(m3u_dir, line)
                line = os.path.abspath(os.path.expanduser(line))
                if os.path.exists(line) and os.path.isfile(line):
                    ext = os.path.splitext(line)[1].lower()
                    if ext in AUDIO_EXTENSIONS:
                        imported_files.append(line)
            if imported_files:
                self.player.add_to_playlist(imported_files)
        except Exception:
            logger.warning("Failed to import M3U playlist", exc_info=True)

    def on_export_m3u(self, button):
        if not self.player.playlist:
            return
        dialog = Gtk.FileDialog(title="Export M3U Playlist")
        dialog.set_modal(True)
        m3u_filter = Gtk.FileFilter()
        m3u_filter.set_name("M3U playlist files")
        m3u_filter.add_mime_type("audio/x-mpegurl")
        m3u_filter.add_pattern("*.m3u")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(m3u_filter)
        dialog.set_filters(filters)
        dialog.set_initial_name("playlist.m3u")
        dialog.save(self.player, None, self._on_m3u_export_selected)

    def _on_m3u_export_selected(self, dialog, result):
        try:
            file = dialog.save_finish(result)
            if file:
                filepath = file.get_path()
                if filepath:
                    if not filepath.lower().endswith('.m3u'):
                        filepath += '.m3u'
                    self._export_m3u_file(filepath)
        except GLib.Error:
            pass

    def _export_m3u_file(self, filepath):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                for item in self.player.playlist:
                    if hasattr(item, 'path') and os.path.exists(item.path):
                        if hasattr(item, 'duration') and item.duration > 0 and hasattr(item, 'title'):
                            f.write(f"#EXTINF:{item.duration},{item.title}\n")
                        f.write(f"{item.path}\n")
        except Exception:
            logger.warning("Failed to export M3U playlist", exc_info=True)

    def on_remove_duplicates(self, button):
        if not self.player.playlist:
            return
        seen_paths = set()
        duplicates_removed = 0
        unique_playlist = []
        for item in self.player.playlist:
            try:
                norm_path = os.path.realpath(item.path)
            except Exception:
                norm_path = item.path
            if norm_path in seen_paths:
                duplicates_removed += 1
                continue
            seen_paths.add(norm_path)
            unique_playlist.append(item)
        if duplicates_removed > 0:
            self.player.playlist = unique_playlist
            self.player._update_playlist_display()
            self.player.save_playlist()
            pass
        else:
            pass

    def on_row_activated(self, column_view, position):
        if hasattr(self, 'filter_model') and self.filter_model.get_filter():
            item = self.selection_model.get_item(position)
            if item is not None:
                stored_string = item.get_string()
                if '|' in stored_string:
                    track_title, track_id = stored_string.split('|', 1)
                else:
                    track_title = stored_string
                    track_id = None
                    for playlist_item in self.player.playlist:
                        if playlist_item.title == track_title or playlist_item.path.endswith(track_title):
                            track_id = playlist_item.id
                            break
                    if track_id is None:
                        track_id = hashlib.md5(track_title.encode()).hexdigest()[:16]
                for i, playlist_item in enumerate(self.player.playlist):
                    if playlist_item.id == track_id:
                        self.player.play_track(i)
                        break
        else:
            item = self.playlist_store.get_item(position)
            if item is not None:
                self.player.play_track(position)

    def on_fix_tags(self, button):
        if not hasattr(self.player, 'playlist') or not self.player.playlist:
            self.player.set_status_message("No tracks in playlist to fix")
            return
        if not METADATA_FIXER_AVAILABLE:
            self.player.set_status_message("Metadata fixer not available")
            return
        self._run_fix_tags()

    def _run_fix_tags(self):
        if not self.player.playlist:
            return
        file_paths = [item.path for item in self.player.playlist if hasattr(item, 'path') and item.path and os.path.exists(item.path)]
        if not file_paths:
            self.player.set_status_message("No valid files to fix")
            return
        self.player.set_status_message(f"Fixing metadata for {len(file_paths)} tracks...")

        def fix_in_background():
            fixer = MetadataFixer()
            try:
                results = fixer.fix_files_threaded(
                    file_paths,
                    metadata=None,
                    cleanup_options=None
                )
                successful = sum(1 for r in results.values() if r.get('success', False))
                failed = len(results) - successful
                GLib.idle_add(self._on_fix_tags_complete, successful, failed)
            except Exception:
                GLib.idle_add(self._on_fix_tags_complete, 0, len(file_paths))
        import threading
        thread = threading.Thread(target=fix_in_background, daemon=True)
        thread.start()

    def _on_fix_tags_complete(self, successful: int, failed: int):
        self.player.set_status_message(f"Fixed {successful} tracks, {failed} failed")
        if hasattr(self.player, '_update_playlist_display'):
            self.player._update_playlist_display()

class EvoVisualizerTab(Gtk.Box):

    def __init__(self, player, available_presets=None):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.available_presets = available_presets or []


        # Audio probe tracking: keep probe lifetime aligned with the sink/pad.
        self.audio_probe_id = None
        self._probe_sink = None
        self._probe_pad = None
        self._probe_connecting = False
        self._latest_audio = None
        self._latest_audio_seq = 0
        self._last_consumed_audio_seq = 0
        self._audio_consume_timer_id = None
        self._rendering_enabled = True

        self._audio_lock = threading.Lock()
        self._audio_channels = 2
        self._audio_sample_rate = 44100.0

        self._last_vis_update = 0.0
        self.set_visible(True)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_hexpand_set(True)
        self.set_vexpand_set(True)
        self.set_halign(Gtk.Align.FILL)
        self.set_valign(Gtk.Align.FILL)
        self.add_css_class("visualizer-widget")

        # Select renderer implementation.
        # Evo is the only supported renderer; use it when available.
        use_evo = EVO_AVAILABLE and EvoRenderer is not None
        self.visualizer_type = None

        if use_evo:
            try:
                self.visualizer = EvoRenderer(EvoConfig())
                self.visualizer.set_hexpand(True)
                self.visualizer.set_vexpand(True)
                self.visualizer.set_halign(Gtk.Align.FILL)
                self.visualizer.set_valign(Gtk.Align.FILL)
                self.visualizer.set_visible(True)
                self.visualizer.add_css_class("visualizer-glarea")
                self.append(self.visualizer)
                self.visualizer_type = "evo"
            except Exception:
                logger.exception("EvoRenderer initialization failed")
                self.visualizer = None
        else:
            # Evo is the only supported renderer; no fallback.
            pass

        if not hasattr(self, 'visualizer') or self.visualizer is None:
            placeholder = Gtk.Label(label="Visualizer not available")
            placeholder.set_hexpand(True)
            placeholder.set_vexpand(True)
            placeholder.set_halign(Gtk.Align.CENTER)
            placeholder.set_valign(Gtk.Align.CENTER)
            placeholder.add_css_class("placeholder-label")
            self.append(placeholder)
        elif self.visualizer_type in ("opengl", "evo"):
            def update_audio_wrapper(audio_data):
                self.visualizer.update_audio(audio_data)
            self.update_audio = update_audio_wrapper

            def audio_data_provider():
                """Provide current audio data for the visualizer.

                Performance/stability change: when we don't have real audio yet,
                return silence instead of synthesizing fake audio.
                """
                if hasattr(self, '_last_audio_data') and self._last_audio_data is not None:
                    return self._last_audio_data
                if NUMPY_AVAILABLE:
                    import numpy as np
                    return np.zeros(512, dtype=np.float32)
                return None
            if hasattr(self.visualizer, 'set_audio_callback'):
                self.visualizer.set_audio_callback(audio_data_provider)

    def _get_audio_sink_pad(self):
        """Return (sink_pad, sink) for the current player audio sink.

        Prefer a pad **after** format conversion so the probe sees the
        F32LE stereo data the FFT expects. Fall back to the bin's ghost
        pad only when no internal post-conversion pad is available.
        """
        try:
            player_obj = self.player.player if hasattr(self.player, 'player') else self.player
            if not hasattr(player_obj, 'get_property'):
                return None, None
            audio_sink = player_obj.get_property("audio-sink")
            if not audio_sink:
                return None, None

            # If this is our vis_audio_sink bin, probe AFTER the capsfilter
            # so we are guaranteed to see F32LE interleaved stereo.
            vis_caps = audio_sink.get_by_name("vis_caps")
            if vis_caps is not None:
                src = vis_caps.get_static_pad("src")
                if src:
                    return src, audio_sink

            # Fallback: the bin's ghost pad (before conversion — format
            # not guaranteed, but better than nothing).
            sink_pad = audio_sink.get_static_pad("sink")
            if sink_pad:
                return sink_pad, audio_sink
            pads = audio_sink.get_pads()
            if pads:
                return pads[0], audio_sink
        except Exception:
            return None, None
        return None, None

    def cleanup(self):
        """Stop rendering/audio updates and remove the EvoVisualizer audio probe.

        Goal: no stale probe attached to an old sink/pad after tab switches.
        """
        try:
            self._rendering_enabled = False

            if self._audio_consume_timer_id is not None:
                try:
                    GLib.source_remove(self._audio_consume_timer_id)
                except Exception:
                    pass
                self._audio_consume_timer_id = None

            if self.audio_probe_id is None:
                # Still clear identities
                self._probe_sink = None
                self._probe_pad = None
                return

            # Remove probe only from the pad it was attached to.
            if self._probe_pad is not None:
                try:
                    self._probe_pad.remove_probe(self.audio_probe_id)
                except Exception:
                    pass

            self.audio_probe_id = None
            self._probe_sink = None
            self._probe_pad = None

            # Cancel any pending deferred probe-retry loop.
            if getattr(self, '_probe_retry_id', None) is not None:
                try:
                    GLib.source_remove(self._probe_retry_id)
                except Exception:
                    pass
                self._probe_retry_id = None

        except Exception:
            logger.exception("Audio probe cleanup error")

    def _reconnect_audio_probe(self):
        """Reconnect audio probe whenever sink/pad changes.

        Fixes stale probe risk: a probe must be removed from the exact pad it
        was attached to, and reattached when the player rebuilds its pipeline
        (new audio sink).
        """
        if self._probe_connecting:
            return

        sink_pad, audio_sink = self._get_audio_sink_pad()
        if sink_pad is None or audio_sink is None:
            # Sink/pad not ready yet (e.g. pipeline not fully built). Retry a
            # few times on a timer so audio connection self-heals once playback
            # starts, instead of silently leaving the visuals disconnected.
            if getattr(self, '_probe_retry_id', None) is None:
                self._probe_retry_count = 0
                self._probe_retry_id = GLib.timeout_add(250, self._retry_audio_probe)
            return

        need_reconnect = False
        if self.audio_probe_id is None:
            need_reconnect = True
        elif self._probe_sink is not audio_sink:
            need_reconnect = True
        elif self._probe_pad is not sink_pad:
            need_reconnect = True

        if not need_reconnect:
            return

        self._probe_connecting = True
        try:
            # Remove old probe from old pad if identities changed.
            if self.audio_probe_id is not None and self._probe_pad is not None:
                try:
                    self._probe_pad.remove_probe(self.audio_probe_id)
                except Exception:
                    pass

            self.audio_probe_id = sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._milkdrop_audio_probe_callback,
                None,
            )
            self._probe_sink = audio_sink
            self._probe_pad = sink_pad

            # Probe attached successfully: stop any deferred retry loop.
            self._probe_retry_id = None
            self._probe_retry_count = 0

            # Keep the FFT sample rate in sync with the sink we built.
            if hasattr(self, 'player') and self.player is not None:
                try:
                    self._audio_sample_rate = float(getattr(
                        self.player, '_audio_sample_rate', 44100.0))
                except Exception:
                    pass

            # Ensure we have a consumer that pushes latest_audio into renderer
            # at render-time cadence.

            if self._audio_consume_timer_id is None:
                # ~30Hz consumer; renderer itself runs at GLArea cadence.
                self._audio_consume_timer_id = GLib.timeout_add(33, self._consume_latest_audio)

        except Exception:
            # Best-effort; avoid leaving broken half-state.
            self.audio_probe_id = None
            self._probe_sink = None
            self._probe_pad = None
        finally:
            self._probe_connecting = False


    def _retry_audio_probe(self):
        """Periodically re-attempt probe attachment until the sink is ready.

        Cancelled automatically once ``_reconnect_audio_probe`` succeeds (it
        clears ``_probe_retry_id``). Gives up after a bounded number of tries.
        """
        if getattr(self, 'audio_probe_id', None) is not None:
            self._probe_retry_id = None
            return False
        self._probe_retry_count = getattr(self, '_probe_retry_count', 0) + 1
        if self._probe_retry_count > 40:
            self._probe_retry_id = None
            return False
        self._reconnect_audio_probe()
        if getattr(self, 'audio_probe_id', None) is not None:
            self._probe_retry_id = None
            return False
        return True

    def _milkdrop_audio_probe_callback(self, pad, info, user_data):
        """Pad-probe callback: capture latest audio but don't push directly.

        Avoids hundreds of renderer.update_audio() calls per second.
        The consumer timer pushes at a controlled cadence.
        """
        try:
            if not self._rendering_enabled:
                return Gst.PadProbeReturn.OK

            buffer = info.get_buffer()
            if not buffer:
                return Gst.PadProbeReturn.OK

            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                return Gst.PadProbeReturn.OK

            try:
                import numpy as np
                audio, rate, channels = self._buffer_to_float32(pad, map_info)
                if audio is None or len(audio) == 0:
                    return Gst.PadProbeReturn.OK
                # Sanitize and store a private copy, protected by the lock so the
                # GLib consumer thread can never read a half-written reference.
                audio_array = np.asarray(audio, dtype=np.float32)
                audio_array = np.nan_to_num(audio_array, nan=0.0, posinf=1.0, neginf=0.0)
                audio_array = np.clip(audio_array, -1.0, 1.0)
                with self._audio_lock:
                    self._latest_audio = audio_array.copy()
                    self._audio_channels = channels
                    self._audio_sample_rate = float(rate)
                    self._latest_audio_seq = (self._latest_audio_seq + 1) & 0xFFFFFFFF
            finally:
                buffer.unmap(map_info)
        except Exception:
            logger.exception("Error in EvoVisualizerTab audio probe callback")
        return Gst.PadProbeReturn.OK

    def _buffer_to_float32(self, pad, map_info):
        """Convert a mapped GStreamer buffer to a float32 sample array.

        Inspects the pad's negotiated caps so we no longer assume F32LE: the
        format is converted correctly for S16LE/S24LE/S32LE/F64LE as well.
        Returns (array, rate_hz, channels) or (None, rate, channels) when the
        format cannot be interpreted.
        """
        import numpy as np
        rate = float(getattr(self, '_audio_sample_rate', 44100.0))
        channels = getattr(self, '_audio_channels', 2) or 2
        fmt = None
        caps = None
        try:
            if pad is not None and hasattr(pad, 'get_current_caps'):
                caps = pad.get_current_caps()
        except Exception:
            caps = None
        if caps is not None and caps.get_size() > 0:
            try:
                struct = caps.get_structure(0)
                fmt = struct.get_string("format")
                ch = struct.get_int("channels")
                if ch is not None and ch[0]:
                    channels = ch[1]
                r = struct.get_int("rate")
                if r is not None and r[0]:
                    rate = float(r[1])
            except Exception:
                pass
        data = map_info.data
        if fmt == "F32LE":
            audio = np.frombuffer(data, dtype=np.float32)
        elif fmt == "F64LE":
            audio = np.frombuffer(data, dtype=np.float64).astype(np.float32)
        elif fmt == "S16LE":
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif fmt == "S32LE":
            audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif fmt == "S24LE":
            n = len(data) // 3
            if n == 0:
                return None, rate, channels
            raw = np.frombuffer(data[:n * 3], dtype=np.uint8).reshape(n, 3)
            ints = (raw[:, 0].astype(np.int32)
                    | (raw[:, 1].astype(np.int32) << 8)
                    | (raw[:, 2].astype(np.int32) << 16))
            ints = np.where(ints >= (1 << 23), ints - (1 << 24), ints)
            audio = ints.astype(np.float32) / 8388608.0
        else:
            # Unsupported/unexpected format: do not mis-parse raw bytes.
            return None, rate, channels
        return audio, rate, channels

    def _consume_latest_audio(self):
        """Push latest captured audio into renderer at ~30Hz.

        EvoRenderer.update_audio() expects a dict with keys:
        bass/bass_att, mid/mid_att, treb/treb_att, beat.

        The pad-probe captures raw samples (any supported sample format,
        converted to float32), so we derive genuine frequency bands via an FFT
        plus a beat proxy from bass transients.
        """
        if not self._rendering_enabled:
            return False

        if not hasattr(self, 'visualizer') or self.visualizer is None or not hasattr(self.visualizer, 'update_audio'):
            return True

        with self._audio_lock:
            seq = self._latest_audio_seq
            if seq == self._last_consumed_audio_seq:
                return True
            audio = self._latest_audio
            if audio is None:
                return True
            # Work on a private copy so the probe thread can't mutate under us.
            audio = audio.copy()

        try:
            # audio is a 1D float32 array (interleaved); derive crude frequency bands
            import numpy as np

            waveform_samples = None

            if not isinstance(audio, np.ndarray) or audio.size == 0:
                audio_levels = {
                    'bass': 0.0, 'bass_att': 0.0,
                    'mid': 0.0, 'mid_att': 0.0,
                    'treb': 0.0, 'treb_att': 0.0,
                    'beat': 0.0,
                }
            else:
                x = np.asarray(audio, dtype=np.float32)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

                # --- Real frequency analysis via FFT ---
                # The raw samples are a *time-domain* window. Splitting that
                # window into "bass/mid/treb" buckets by *sample index* measures
                # nothing about pitch, so all three bands collapsed to the same
                # value (~1.0) and the visualizer stopped tracking the music.
                # Instead compute a magnitude spectrum and integrate energy into
                # genuine musical bands, then drive a beat from bass transients.
                n = x.size
                if n < 2:
                    audio_levels = {
                        'bass': 0.0, 'bass_att': 0.0,
                        'mid': 0.0, 'mid_att': 0.0,
                        'treb': 0.0, 'treb_att': 0.0,
                        'beat': 0.0,
                    }
                else:
                    # Collapse interleaved channels to mono using the channel
                    # count negotiated on the sink (not a hardcoded stereo pair).
                    channels = getattr(self, '_audio_channels', 2) or 1
                    if channels > 1:
                        frames = n // channels
                        if frames >= 1:
                            xx = x[:frames * channels].reshape(frames, channels).mean(axis=1)
                        else:
                            xx = x
                    else:
                        xx = x

                    waveform_samples = xx

                    # typically 44.1k/48k; 44.1k is a safe default).
                    rate = float(getattr(self, '_audio_sample_rate', 44100.0))
                    nn = xx.size
                    if nn >= 2:
                        windowed = xx * np.hanning(nn)
                        spec = np.abs(np.fft.rfft(windowed))
                        freqs = np.fft.rfftfreq(nn, d=1.0 / rate)

                        def band_energy(lo, hi):
                            mask = (freqs >= lo) & (freqs < hi)
                            if not np.any(mask):
                                return 0.0
                            return float(np.sqrt(np.mean(spec[mask] ** 2) + 1e-12))

                        bass_e = band_energy(20.0, 250.0)
                        mid_e = band_energy(250.0, 2000.0)
                        treb_e = band_energy(2000.0, 8000.0)
                        total_e = band_energy(20.0, rate / 2.0) or 1e-9

                        # Normalize each band by total spectral energy so the
                        # levels track the *balance* of the music (not merely
                        # loudness), with a gain that keeps them in [0,1].
                        def norm(v):
                            return float(min(1.0, max(0.0, (v / total_e) * 1.0)))

                        bass = norm(bass_e * 1.6)
                        mid = norm(mid_e * 1.4)
                        treb = norm(treb_e * 2.2)

                        # Beat: a transient in the bass band vs. a slow average.
                        if not hasattr(self, '_beat_bass_avg'):
                            self._beat_bass_avg = 0.0
                        self._beat_bass_avg = self._beat_bass_avg * 0.9 + bass_e * 0.1
                        beat_transient = max(0.0, bass_e - self._beat_bass_avg * 1.2)
                        beat = float(min(1.0, max(bass, beat_transient * 3.0)))
                    else:
                        bass = mid = treb = beat = 0.0

                    audio_levels = {
                        'bass': bass,
                        'bass_att': bass,
                        'mid': mid,
                        'mid_att': mid,
                        'treb': treb,
                        'treb_att': treb,
                        'beat': beat,
                        # Carry the raw (mono) samples so the renderer can still
                        # draw the waveform; EvoRenderer.update_audio accepts
                        # this dict form.
                        'waveform': waveform_samples,
                    }

            self.visualizer.update_audio(audio_levels)
            self._last_consumed_audio_seq = seq
        except Exception:
            logger.exception("Error consuming latest audio for visualizer")
            return True

        return True



class PlayerTab(Gtk.Box):
    def __init__(self, player):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.player = player
        self.render_lock = threading.RLock()
        self._frame_counter = 0
        self._last_beat_time = 0
        self._beat_times = []
        self.add_css_class("player-tab")
        self.set_spacing(8)
        self.set_margin_top(8)
        self.set_margin_bottom(8)
        self.set_margin_start(8)
        self.set_margin_end(8)
        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.append(top_row)
        self.cover_art_widget = CoverArtWidget(player, size=200)
        self.cover_art_widget.add_css_class("player-cover")
        top_row.append(self.cover_art_widget)
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        info_box.set_hexpand(True)
        top_row.append(info_box)
        self.track_label = Gtk.Label(label="No track playing")
        self.track_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.track_label.add_css_class("track-label")
        self.track_label.set_halign(Gtk.Align.START)
        info_box.append(self.track_label)
        self.time_label = Gtk.Label(label="--:-- / --:--")
        self.time_label.add_css_class("time-label")
        self.time_label.set_halign(Gtk.Align.START)
        info_box.append(self.time_label)
        seek_adjustment = Gtk.Adjustment(
            value=0, lower=0, upper=100, step_increment=0.1,
            page_increment=1, page_size=0
        )
        self.progress = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=seek_adjustment)
        self.progress.set_hexpand(True)
        self.progress.set_size_request(-1, 20)
        self.progress.add_css_class("seek-bar")
        self.progress.set_draw_value(False)
        self.progress_controller = Gtk.GestureClick()
        self.progress_controller.connect("pressed", self.on_progress_pressed)
        self.progress.add_controller(self.progress_controller)
        info_box.append(self.progress)
        main_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        main_controls.set_halign(Gtk.Align.CENTER)
        main_controls.set_homogeneous(True)
        self.append(main_controls)
        self.prev_btn = self._create_icon_button("media-skip-backward-symbolic", "Previous")
        self.play_btn = self._create_icon_button("media-playback-start-symbolic", "Play")
        self.pause_btn = self._create_icon_button("media-playback-pause-symbolic", "Pause")
        self.stop_btn = self._create_icon_button("media-playback-stop-symbolic", "Stop")
        self.next_btn = self._create_icon_button("media-skip-forward-symbolic", "Next")
        for btn in [self.prev_btn, self.play_btn, self.pause_btn, self.stop_btn, self.next_btn]:
            btn.add_css_class("control-button")
            btn.set_size_request(32, 32)
            main_controls.append(btn)
        self.play_btn.add_css_class("suggested-action")
        self.prev_btn.connect("clicked", self.player.on_prev)
        self.play_btn.connect("clicked", self.player.on_play)
        self.pause_btn.connect("clicked", self.player.on_pause)
        self.stop_btn.connect("clicked", self.player.on_stop)
        self.next_btn.connect("clicked", self.player.on_next)
        secondary_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        secondary_controls.set_halign(Gtk.Align.CENTER)
        self.append(secondary_controls)
        self.repeat_btn = self._create_icon_button("media-playlist-repeat-symbolic", "Repeat")
        self.shuffle_btn = self._create_icon_button("media-playlist-shuffle-symbolic", "Shuffle")
        self.beat_btn = self._create_icon_button("process-working-symbolic", "Beat")
        self.download_btn = self._create_icon_button("folder-download-symbolic", "Download")
        self.autonext_btn = self._create_icon_button("go-next-symbolic", "Auto Next")
        for btn in [self.repeat_btn, self.shuffle_btn, self.beat_btn, self.download_btn, self.autonext_btn]:
            btn.add_css_class("icon-button")
            btn.set_size_request(24, 24)
            secondary_controls.append(btn)
        self.repeat_btn.connect("clicked", lambda b: self.player.toggle_repeat_mode())
        self.shuffle_btn.connect("clicked", lambda b: self.player.toggle_shuffle_mode())
        self.beat_btn.connect("clicked", lambda b: self.player.toggle_beat_aware())
        self.download_btn.connect("clicked", lambda b: self.player.toggle_auto_download_covers())
        self.autonext_btn.connect("clicked", lambda b: self.player.toggle_auto_play_next())
        vol_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        vol_row.set_halign(Gtk.Align.CENTER)
        self.append(vol_row)
        self.volume_button = self._create_volume_button()
        self.volume_button.add_css_class("volume-button")
        self.volume_button.set_size_request(24, 24)
        vol_row.append(self.volume_button)
        lower, upper = 0, 100
        value = max(lower, min(upper, 70))
        vol_adjustment = Gtk.Adjustment(
            value=value, lower=lower, upper=upper, step_increment=1,
            page_increment=10, page_size=0
        )
        self.volume_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=vol_adjustment)
        self.volume_scale.set_hexpand(True)
        self.volume_scale.set_draw_value(False)
        self.volume_scale.set_size_request(150, 20)
        self.volume_scale.connect("value-changed", self.player.on_volume_changed)
        vol_row.append(self.volume_scale)
        self.volume_label = Gtk.Label(label="70%")
        self.volume_label.add_css_class("volume-label")
        self.volume_label.set_size_request(35, -1)
        vol_row.append(self.volume_label)
        self.update_playback_state(False)

    def _create_volume_button(self):
        button = Gtk.Button()
        button.add_css_class("icon-button")
        button.set_tooltip_text("Mute/Unmute (M)")
        button.set_size_request(32, 28)
        button.connect("clicked", self.player.on_mute_toggle)
        self._update_volume_button_icon()
        return button

    def _update_volume_button_icon(self):
        if not hasattr(self, 'volume_button') or not hasattr(self, 'player') or not self.player:
            return
        current_volume = self.player.get_property("volume")
        if current_volume == 0 or getattr(self.player, '_is_muted', False):
            icon_name = "audio-volume-muted-symbolic"
        elif current_volume < 0.3:
            icon_name = "audio-volume-low-symbolic"
        elif current_volume < 0.7:
            icon_name = "audio-volume-medium-symbolic"
        else:
            icon_name = "audio-volume-high-symbolic"
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        self.volume_button.set_child(icon)

    def _create_icon_button(self, icon_name, tooltip_text):
        button = Gtk.Button()
        button.add_css_class("icon-button")
        button.set_tooltip_text(tooltip_text)
        button.set_size_request(36, 32)
        icon_image = Gtk.Image.new_from_icon_name(icon_name)
        icon_image.set_pixel_size(16)
        button.set_child(icon_image)
        return button

    def update_playback_state(self, is_playing):
        if is_playing:
            self.play_btn.set_visible(False)
            self.pause_btn.set_visible(True)
            self.add_css_class("playing")
        else:
            self.play_btn.set_visible(True)
            self.pause_btn.set_visible(False)
            self.remove_css_class("playing")

    def on_progress_pressed(self, gesture, n_press, x, y):
        width = self.progress.get_width()
        ratio = x / width if width > 0 else 0
        ratio = max(0.0, min(1.0, ratio))
        player_obj = self.player
        if hasattr(player_obj, 'player'):
            player_obj = player_obj.player
        duration = player_obj.query_duration(Gst.Format.TIME)[1]
        if duration > 0:
            seek_time = int(ratio * duration)
            player_obj.seek(1.0, Gst.Format.TIME,
                                  Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                                  Gst.SeekType.SET, seek_time,
                                  Gst.SeekType.NONE, 0)

    def update_track_info(self, title, artist="", album=""):
        if not title:
            title = "No track playing"
            artist = ""
        try:
            screen_size = _get_default_screen_size()
            if screen_size is None:
                return
            screen_width, screen_height = screen_size
            if hasattr(self, 'get_surface') and self.get_surface():
                x, y = self.get_surface().get_position()
                max_x = screen_width - 450
                max_y = screen_height - 350
                if x > max_x or y > max_y or x < 0 or y < 0:
                    new_x = max(50, min(x, max_x))
                    new_y = max(50, min(y, max_y))
                    self.move(new_x, new_y)
        except Exception:
            pass
        if artist and album:
            info_text = f"<b>{GLib.markup_escape_text(title)}</b>\n" \
                       f"<small>{GLib.markup_escape_text(artist)} • {GLib.markup_escape_text(album)}</small>"
        elif artist:
            info_text = f"<b>{GLib.markup_escape_text(title)}</b>\n" \
                       f"<small>{GLib.markup_escape_text(artist)}</small>"
        else:
            info_text = f"<b>{GLib.markup_escape_text(title)}</b>"
        self.track_label.set_markup(info_text)
        self.track_label.set_tooltip_text(f"{title}\n{artist}{' • ' + album if album else ''}")
        window = self.get_root()
        if window:
            window.set_title(f"{title} - LinAmp")

class WinampWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
            self.set_title("LinAmp")
            self.add_css_class("linamp-window")
            self.header_bar = Gtk.HeaderBar()
            self.header_bar.add_css_class("app-header")
            self.set_titlebar(self.header_bar)
            title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
            title_box.add_css_class("title-box")
            title_label = Gtk.Label(label="LinAmp")
            title_label.add_css_class("app-title")
            title_label.set_halign(Gtk.Align.START)
            self.header_nowplaying = Gtk.Label(label="Ready")
            self.header_nowplaying.add_css_class("app-subtitle")
            self.header_nowplaying.set_halign(Gtk.Align.START)
            self.header_nowplaying.set_ellipsize(Pango.EllipsizeMode.END)
            title_box.append(title_label)
            title_box.append(self.header_nowplaying)
            self.header_bar.set_title_widget(title_box)
            search_btn = Gtk.Button()
            search_btn.set_tooltip_text("Search playlist")
            search_btn.add_css_class("icon-button")
            search_btn.set_child(Gtk.Image.new_from_icon_name("system-search-symbolic"))
            search_btn.connect("clicked", lambda *_: self.focus_playlist_search())
            self.header_bar.pack_end(search_btn)
  
            self.set_resizable(True)
            self._active_timers = set()
            self.playlist = []
            self.current_track = -1
            self.shuffled_indices = []
            self.shuffle_position = 0
            self._is_muted = False
            self._previous_volume = 1.0
            self._playback_in_progress = False
            self.beat_detection_timer = 0
            self._last_avg_level = 0.0
            self._last_beat_timestamp = 0.0
            self._bus_watch_id = None
            self._main_bus_signal_watch_attached = False
            self._internal_buses_signal_watches_attached = []
            self.is_compact_mode = False
            self.current_window_width = 650
            self.last_beat_time = 0
            self.beat_threshold = 0.1
            self.beat_interval_history = []
            self.settings = PlayerSettings()
            self.load_settings()
            self.auto_play_next = self.settings.auto_play_next
            self.shuffle_mode = self.settings.shuffle_mode
            self.repeat_mode = self.settings.repeat_mode
            self.beat_aware_enabled = self.settings.beat_aware_enabled
            self.beat_threshold = self.settings.beat_threshold
            self.auto_download_covers = self.settings.auto_download_covers
            self.connect("close-request", self.on_close_request)
            self.connect("destroy", self.on_window_destroy)
            self.connect("notify::default-width", self.on_window_size_changed)
            self.connect("notify::default-height", self.on_window_size_changed)
            self.connect("notify::width", self.on_window_size_changed)
            self.connect("notify::height", self.on_window_size_changed)
        except Exception:
            raise

        def suppress_source_remove_warning(log_domain, log_level, message, user_data=None):
            if "Source ID" in str(message) and "was not found" in str(message):
                return
            return
        GLib.log_set_handler("GLib", GLib.LogLevelFlags.LEVEL_WARNING | GLib.LogLevelFlags.LEVEL_CRITICAL,
                            suppress_source_remove_warning, None)
        self.setup_player()
        self.apply_xmms_css()
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_box.add_css_class("main-container")
        self.main_box.set_hexpand(True)
        self.main_box.set_vexpand(True)
        self.main_box.set_halign(Gtk.Align.FILL)
        self.main_box.set_valign(Gtk.Align.FILL)
        self.main_box.set_hexpand_set(True)
        self.main_box.set_vexpand_set(True)
        self.set_child(self.main_box)
        self.notebook = Gtk.Notebook()
        self.notebook.add_css_class("main-notebook")
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_halign(Gtk.Align.FILL)
        self.notebook.set_valign(Gtk.Align.FILL)
        self.notebook.set_hexpand_set(True)
        self.notebook.set_vexpand_set(True)
        self.notebook.set_scrollable(False)

        def on_critical_log(log_domain, log_level, message, user_data=None):
            if "CRITICAL" in str(log_level):
                pass
            return
        GLib.log_set_handler("Gtk", GLib.LogLevelFlags.LEVEL_CRITICAL, on_critical_log, None)
        self.main_box.append(self.notebook)
        self.player_tab = PlayerTab(self)
        self.equalizer_tab = EqualizerTab(self)
        from pathlib import Path
        self.available_presets = []
        presets_dir = Path("./presets")
        _t0 = time.time()
        if presets_dir.exists():
            # NOTE: this only enumerates filenames (Path objects) - it does NOT
            # read/parse/compile any preset. If this scan ever shows up as the
            # 2-minute stall, the bottleneck is filesystem enumeration over the
            # 10k+ file tree, not preset loading.
            self.available_presets = [p for p in presets_dir.rglob("*.milk") if p.exists()]
        _scan_dt = time.time() - _t0
        print(f"PRESET SCAN: found {len(self.available_presets)} presets in ./presets/ ({_scan_dt:.3f}s)")
        if EVO_AVAILABLE:
            try:
                self.visualizer_tab = EvoVisualizerTab(self, self.available_presets)
            except Exception as e:
                self.visualizer_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                self.visualizer_tab.add_css_class("visualizer-widget")
                error_label = Gtk.Label(label=f"Visualizer failed to load: {e}")
                error_label.add_css_class("placeholder-label")
                self.visualizer_tab.append(error_label)
        else:
            self.visualizer_tab = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            self.visualizer_tab.add_css_class("visualizer-widget")
            error_label = Gtk.Label(label="Visualizer unavailable: EvoRenderer missing")
            error_label.add_css_class("placeholder-label")
            self.visualizer_tab.append(error_label)
        self.playlist_tab = PlaylistTab(self)
        self.player_tab.set_hexpand(True)
        self.player_tab.set_vexpand(True)
        self.equalizer_tab.set_hexpand(True)
        self.equalizer_tab.set_vexpand(True)
        self.playlist_tab.set_hexpand(True)
        self.playlist_tab.set_vexpand(True)
        self.notebook.append_page(self.player_tab, self._create_tab_widget("Player", "media-playback-start-symbolic"))
        self.notebook.append_page(self.equalizer_tab, self._create_tab_widget("Equalizer", "audio-equalizer-symbolic"))
        self.notebook.append_page(self.playlist_tab, self._create_tab_widget("Playlist", "view-list-symbolic"))
        self.notebook.set_current_page(0)
        self.notebook.set_hexpand(True)
        self.notebook.set_vexpand(True)
        self.notebook.set_halign(Gtk.Align.FILL)
        self.notebook.set_valign(Gtk.Align.FILL)
        self.notebook.set_hexpand_set(True)
        self.notebook.set_vexpand_set(True)
        if self.visualizer_tab:
            print(f"Visualizer tab created, type: {type(self.visualizer_tab)}")
            self.visualizer_tab.set_hexpand(True)
            self.visualizer_tab.set_vexpand(True)
            self.visualizer_tab.set_halign(Gtk.Align.FILL)
            self.visualizer_tab.set_valign(Gtk.Align.FILL)
            try:
                tab_widget = self._create_tab_widget("Visualizer", "applications-graphics-symbolic")
                self.notebook.append_page(self.visualizer_tab, tab_widget)
                print(f"Visualizer tab added to notebook, total pages: {self.notebook.get_n_pages()}")

                def on_switch_page(notebook, page, page_num):
                    print(f"Switch page called: page_num={page_num}, page={page}, visualizer_tab={self.visualizer_tab}")
                    if page == self.visualizer_tab:
                        if hasattr(self.visualizer_tab, 'cleanup'):
                            # re-enable rendering/audio for this tab
                            self.visualizer_tab._rendering_enabled = True
                        if hasattr(self.visualizer_tab, '_reconnect_audio_probe'):
                            self.visualizer_tab._reconnect_audio_probe()
                        if hasattr(self.visualizer_tab, 'visualizer') and self.visualizer_tab.visualizer is not None:
                            try:
                                viz = self.visualizer_tab.visualizer
                                if hasattr(viz, 'realize') and hasattr(viz, 'gl_ready') and not getattr(viz, 'gl_ready', False):
                                    try:
                                        viz.realize()
                                    except Exception:
                                        pass
                                viz.queue_render()
                            except Exception:
                                pass
                    elif page == self.player_tab and hasattr(self, 'visualizer_tab'):
                        if hasattr(self.visualizer_tab, 'cleanup'):
                            self.visualizer_tab.cleanup()

                self.notebook.connect("switch-page", on_switch_page)
            except Exception as e:
                print(f"Error adding visualizer tab: {e}")
                import traceback
                traceback.print_exc()
                pass

        GLib.idle_add(self.apply_responsive_layout)
        self.player_tab.play_btn.connect("clicked", self.on_play)
        self.player_tab.pause_btn.connect("clicked", self.on_pause)
        self.player_tab.stop_btn.connect("clicked", self.on_stop)
        self.player_tab.prev_btn.connect("clicked", self.on_prev)
        self.player_tab.next_btn.connect("clicked", self.on_next)
        self.player_tab.volume_scale.connect("value-changed", self.on_volume_changed)
        self.player_tab.repeat_btn.connect("clicked", lambda b: self.toggle_repeat_mode())
        self.player_tab.shuffle_btn.connect("clicked", lambda b: self.toggle_shuffle_mode())
        self.player_tab.beat_btn.connect("clicked", lambda b: self.toggle_beat_aware())
        self.player_tab.download_btn.connect("clicked", lambda b: self.toggle_auto_download_covers())
        self.player_tab.autonext_btn.connect("clicked", lambda b: self.toggle_auto_play_next())
        self.status_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.status_bar.set_size_request(-1, 28)
        self.status_bar.add_css_class("statusbar")
        self.status_label = Gtk.Label()
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_hexpand(True)
        self.status_label.set_margin_start(12)
        self.status_label.set_margin_end(12)
        self.status_bar.append(self.status_label)
        self.visualizer_status_label = Gtk.Label(label="")
        self.visualizer_status_label.set_margin_end(12)
        self.visualizer_status_label.set_visible(False)
        self.status_bar.append(self.visualizer_status_label)
        self.main_box.append(self.status_bar)
        self.setup_drag_drop()
        timer1 = GLib.timeout_add(500, self._apply_equalizer_settings_delayed)
        self._active_timers.add(timer1)
        timer2 = GLib.timeout_add(1000, self._apply_ui_settings_delayed)
        self._active_timers.add(timer2)
        timer3 = GLib.timeout_add(UPDATE_DISPLAY_INTERVAL, self.update_display)
        self._active_timers.add(timer3)
        timer4 = GLib.timeout_add(PERIODIC_SAVE_INTERVAL, self.periodic_auto_save)
        self._active_timers.add(timer4)
        self.load_playlist()

    def on_window_size_changed(self, widget, pspec):
        width = widget.get_width()
        if width <= 0:
            width = self.current_window_width
        was_compact = self.is_compact_mode
        if width < 600:
            self.is_compact_mode = True
        else:
            self.is_compact_mode = False
        if was_compact != self.is_compact_mode or width != self.current_window_width:
            self.apply_responsive_layout()
        self.current_window_width = width

    def apply_responsive_layout(self):
        window_width = self.current_window_width if hasattr(self, 'current_window_width') else 600
        if window_width <= 400:
            cover_size = 200
        elif window_width <= 500:
            cover_size = 100
        elif window_width <= 600:
            cover_size = 150
        else:
            cover_size = 250
        if self.is_compact_mode:
            if hasattr(self.player_tab, 'volume_scale'):
                self.player_tab.volume_scale.set_size_request(120, -1)
            for btn in [self.player_tab.prev_btn, self.player_tab.play_btn,
                       self.player_tab.pause_btn, self.player_tab.stop_btn,
                       self.player_tab.next_btn]:
                btn.set_size_request(28, 24)
                btn.set_margin_start(2)
                btn.set_margin_end(2)
            if hasattr(self.playlist_tab, 'search_entry'):
                self.playlist_tab.search_entry.set_margin_start(4)
                self.playlist_tab.search_entry.set_margin_end(4)
                self.playlist_tab.search_entry.set_size_request(-1, 32)
            if hasattr(self.player_tab, 'cover_art_widget'):
                compact_size = max(60, cover_size // 2)
                self.player_tab.cover_art_widget.set_size(compact_size)
                cover_size = compact_size
            if hasattr(self.player_tab, 'progress'):
                self.player_tab.progress.set_size_request(-1, 20)
        elif self.current_window_width < 800:
            if hasattr(self.player_tab, 'volume_scale'):
                self.player_tab.volume_scale.set_size_request(160, -1)
            for btn in [self.player_tab.prev_btn, self.player_tab.play_btn,
                       self.player_tab.pause_btn, self.player_tab.stop_btn,
                       self.player_tab.next_btn]:
                btn.set_size_request(36, 30)
                btn.set_margin_start(4)
                btn.set_margin_end(4)
            if hasattr(self.playlist_tab, 'search_entry'):
                self.playlist_tab.search_entry.set_margin_start(8)
                self.playlist_tab.search_entry.set_margin_end(8)
                self.playlist_tab.search_entry.set_size_request(-1, 36)
            if hasattr(self.player_tab, 'cover_art_widget'):
                medium_size = max(100, cover_size - 30)
                self.player_tab.cover_art_widget.set_size(medium_size)
                cover_size = medium_size
            if hasattr(self.player_tab, 'progress'):
                self.player_tab.progress.set_size_request(-1, 24)
        else:
            if hasattr(self.player_tab, 'volume_scale'):
                self.player_tab.volume_scale.set_size_request(200, -1)
            for btn in [self.player_tab.prev_btn, self.player_tab.play_btn,
                       self.player_tab.pause_btn, self.player_tab.stop_btn,
                       self.player_tab.next_btn]:
                btn.set_size_request(44, 36)
                btn.set_margin_start(6)
                btn.set_margin_end(6)
            if hasattr(self.playlist_tab, 'search_entry'):
                self.playlist_tab.search_entry.set_margin_start(16)
                self.playlist_tab.search_entry.set_margin_end(16)
                self.playlist_tab.search_entry.set_size_request(-1, 40)
            if self.current_window_width >= 1200:
                large_size = min(350, cover_size + 100)
            elif self.current_window_width >= 1000:
                large_size = min(300, cover_size + 50)
            else:
                large_size = cover_size
            if hasattr(self.player_tab, 'cover_art_widget'):
                cover_widget = self.player_tab.cover_art_widget
                if hasattr(self, 'item') and self.item:
                    cached_cover_path = CoverArt.get_cached_cover_path(self.item.path)
                    if cached_cover_path and cached_cover_path.exists():
                        cover_widget.load_cover_from_path(cached_cover_path)
                self.player_tab.cover_art_widget.set_size(large_size)
                cover_size = large_size
        if hasattr(self.player_tab, 'progress'):
            self.player_tab.progress.set_size_request(-1, 28)
        if self.is_compact_mode:
            self.main_box.add_css_class("compact-layout")
            if hasattr(self, 'notebook'):
                self.notebook.add_css_class("compact-notebook")
        else:
            self.main_box.remove_css_class("compact-layout")
            if hasattr(self, 'notebook'):
                self.notebook.remove_css_class("compact-notebook")

    def set_status_message(self, message):
        self.status_label.set_label(message)

    def _create_tab_widget(self, text, icon_name):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.add_css_class("tab-widget")
        box.set_visible(True)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        icon.set_visible(True)
        box.append(icon)
        label = Gtk.Label(label=text)
        label.add_css_class("tab-label")
        box.append(label)
        return box

    def save_settings(self):
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(asdict(self.settings), f, indent=2)
            pass
        except Exception:
            pass

    def load_settings(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, 'r') as f:
                    content = f.read()
                if not content.strip():
                    return
                settings_dict = json.loads(content)
                self.settings = PlayerSettings(**settings_dict)
                self._apply_settings_to_state()
                if hasattr(self.settings, 'last_played_track') and self.settings.last_played_track:
                    self._resume_last_played()
            except (json.JSONDecodeError, TypeError, KeyError):
                self.settings = PlayerSettings()

    def _safe_source_remove(self, timer_id):
        if timer_id is None or timer_id == 0:
            return True
        if not hasattr(self, '_active_timers'):
            self._active_timers = set()
        if timer_id in self._active_timers:
            self._active_timers.discard(timer_id)
            try:
                GLib.source_remove(timer_id)
                return True
            except (TypeError, ValueError):
                return False
        else:
            return False

    def cleanup(self):
        if hasattr(self, '_cleanup_in_progress') and self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        self._update_settings_from_state()
        self.save_settings()
        self.stop_beat_detection()
        if hasattr(self, '_active_timers'):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        if hasattr(self, 'player') and self.player:
            bus = self.player.get_bus()
            if bus:
                try:
                    if hasattr(self, '_bus_watch_id') and self._bus_watch_id is not None:
                        bus.disconnect(self._bus_watch_id)
                        self._bus_watch_id = None
                except Exception:
                    pass

                if self._main_bus_signal_watch_attached:
                    try:
                        bus.remove_signal_watch()
                        self._main_bus_signal_watch_attached = False
                    except Exception:
                        pass
            try:
                self.player.set_state(Gst.State.NULL)
            except Exception:
                pass
        if hasattr(self, 'equalizer') and self.equalizer:
            try:
                self.equalizer.set_state(Gst.State.NULL)
            except Exception:
                pass
        if hasattr(self, 'system_audio_pipeline') and self.system_audio_pipeline:
            try:
                self.system_audio_pipeline.set_state(Gst.State.NULL)
                self.system_audio_pipeline = None
            except Exception:
                pass
        if hasattr(self, '_internal_buses') and self._internal_buses:
            for internal_bus in self._internal_buses:
                try:
                    if internal_bus in self._internal_buses_signal_watches_attached:
                        internal_bus.remove_signal_watch()
                        self._internal_buses_signal_watches_attached.remove(internal_bus)
                except Exception:
                    pass
            self._internal_buses = []
        try:
            if hasattr(self, 'player') and self.player:
                self.player.set_state(Gst.State.NULL)
                self.player = None
            if hasattr(self, 'equalizer') and self.equalizer:
                self.equalizer.set_state(Gst.State.NULL)
                self.equalizer = None
        except Exception:
            pass
        if hasattr(self, 'cover_art_widget') and self.cover_art_widget:
            if hasattr(self.cover_art_widget, 'executor') and self.cover_art_widget.executor:
                self.cover_art_widget.executor.shutdown(wait=False)
        self._cleanup_in_progress = False

    def on_window_destroy(self, *args):
        import sys
        try:
            if hasattr(self, 'full_cleanup') and not getattr(self, '_cleanup_in_progress', False):
                self.full_cleanup()
        except Exception:
            pass
        sys.exit(0)

    def _resume_last_played(self):
        if not self.settings.last_played_track:
            return
        track_index = None
        for i, item in enumerate(self.playlist):
            if item.path == self.settings.last_played_track:
                track_index = i
                break
        if track_index is not None:
            if self.play_track(track_index):
                if self.settings.last_played_position > 2.0:
                    GLib.timeout_add(1000, self._seek_to_position, self.settings.last_played_position)
        else:
            pass

    def _seek_to_position(self, position):
        try:
            if hasattr(self, 'player') and self.player and self.playing:
                seek_pos = int(position * Gst.SECOND)
                self.player.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, seek_pos)
        except Exception:
            pass
        return False

    def _update_settings_from_state(self):
        if not hasattr(self, 'settings'):
            return
        self.settings.auto_play_next = self.auto_play_next
        self.settings.shuffle_mode = self.shuffle_mode
        self.settings.repeat_mode = self.repeat_mode
        self.settings.beat_aware_enabled = self.beat_aware_enabled
        self.settings.beat_threshold = self.beat_threshold
        self.settings.auto_download_covers = self.auto_download_covers
        if self.current_track >= 0 and self.current_track < len(self.playlist):
            current_item = self.playlist[self.current_track]
            self.settings.last_played_track = current_item.path
            if self.playing and hasattr(self, 'player') and self.player:
                try:
                    success, position = self.player.query_position(Gst.Format.TIME)
                    if success:
                        self.settings.last_played_position = position / Gst.SECOND
                except Exception:
                    self.settings.last_played_position = 0.0
        if hasattr(self, 'player') and self.player:
            try:
                volume = self.player.get_property("volume")
                if volume is not None:
                    self.settings.volume = volume
            except Exception:
                pass
        if hasattr(self, 'get_default_size'):
            size = self.get_default_size()
            self.settings.window_size = (size.width, size.height)
        if hasattr(self, 'get_position'):
            self.settings.window_position = self.get_position()
        if hasattr(self, 'equalizer') and self.equalizer:
            self.settings.equalizer_settings = []
            for i in range(10):
                try:
                    value = self.equalizer.get_property('band' + str(i))
                    self.settings.equalizer_settings.append(value)
                except Exception:
                    pass
                    self.settings.equalizer_settings.append(0.0)

    def _apply_settings_to_state(self):
        if not hasattr(self, 'settings'):
            return
        self.auto_play_next = self.settings.auto_play_next
        self.shuffle_mode = self.settings.shuffle_mode
        self.repeat_mode = self.settings.repeat_mode
        self.beat_aware_enabled = self.settings.beat_aware_enabled
        self.beat_threshold = self.settings.beat_threshold
        self.auto_download_covers = self.settings.auto_download_covers
        if hasattr(self, 'player') and self.player:
            try:
                volume = max(0.0, min(1.0, self.settings.volume))
                self.player.set_property("volume", volume)
            except Exception:
                pass
        if hasattr(self, 'player_tab') and hasattr(self.player_tab, 'volume_scale'):
            try:
                volume_percent = self.settings.volume * 100
                clamped_volume = max(0, min(100, volume_percent))
                self.player_tab.volume_scale.set_value(clamped_volume)
                if hasattr(self.player_tab, 'volume_label'):
                    self.player_tab.volume_label.set_label(f"{int(clamped_volume)}%")
            except Exception:
                pass
        if hasattr(self, 'move') and self.settings.window_position:
            x, y = self.settings.window_position
            screen_size = _get_default_screen_size()
            if screen_size is None:
                raise RuntimeError("Display geometry unavailable")
            screen_width, screen_height = screen_size
            max_x = screen_width - 450
            max_y = screen_height - 350
            x = max(50, min(x, max_x))
            y = max(50, min(y, max_y))
            self.move(x, y)
        if hasattr(self, 'equalizer') and self.equalizer and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                self.equalizer.set_property('band' + str(i), value)
        if hasattr(self, 'equalizer_tab') and self.settings.equalizer_settings:
            try:
                for i, value in enumerate(self.settings.equalizer_settings[:10]):
                    if i < len(self.equalizer_tab.band_scales):
                        scale, value_label = self.equalizer_tab.band_scales[i]
                        clamped_value = max(-12, min(12, value))
                        scale.set_value(clamped_value)
                        value_label.set_text(f"{clamped_value:+.1f} dB")
            except Exception:
                pass

    def _apply_ui_settings_delayed(self):
        if not hasattr(self, 'settings'):
            return False
        if hasattr(self, 'player_tab') and hasattr(self.player_tab, 'volume_scale'):
            try:
                volume_percent = self.settings.volume * 100
                clamped_volume = max(0, min(100, volume_percent))
                self.player_tab.volume_scale.set_value(clamped_volume)
                if hasattr(self.player_tab, 'volume_label'):
                    self.player_tab.volume_label.set_label(f"{int(clamped_volume)}%")
            except Exception:
                pass
        if hasattr(self, 'equalizer_tab') and self.settings.equalizer_settings:
            try:
                for i, value in enumerate(self.settings.equalizer_settings[:10]):
                    if i < len(self.equalizer_tab.band_scales):
                        scale, value_label = self.equalizer_tab.band_scales[i]
                        clamped_value = max(-12, min(12, value))
                        scale.set_value(clamped_value)
                        value_label.set_text(f"{value:+.1f} dB")
            except Exception:
                pass
        return

    def _apply_equalizer_settings_delayed(self):
        if hasattr(self, 'equalizer') and self.equalizer and hasattr(self, 'settings') and self.settings.equalizer_settings:
            for i, value in enumerate(self.settings.equalizer_settings[:10]):
                try:
                    self.equalizer.set_property('band' + str(i), value)
                    pass
                except Exception:
                    pass
        return False

    def save_settings_on_track_change(self):
        self._update_settings_from_state()
        self.save_settings()

    def save_settings_on_stop(self):
        self._update_settings_from_state()
        self.save_settings()

    def periodic_auto_save(self):
        if self.playing:
            self.auto_save_settings()
        return True

    def apply_xmms_css(self):
        css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme", "gtk.css")
        css_provider = Gtk.CssProvider()
        try:
            css_provider.load_from_path(css_file)
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(),
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_USER
            )
        except Exception as e:
            print(f"Error loading CSS from file: {e}")

    def setup_player(self):
        self.player = Gst.ElementFactory.make("playbin", "player")
        if not self.player:
            GLib.idle_add(self.set_status_message, "GStreamer initialization failed")
            return
        self.playing = False
        self.equalizer = None
        self.audio_sink = None
        self.audio_convert = None
        self.audio_resample = None
        self._audio_sample_rate = 44100.0
        self._audio_channels = 2
        self.system_audio_pipeline = None
        self.system_audio_probe_id = None
        try:
            audio_sink = self._build_visualizer_audio_sink()
            if not audio_sink:
                audio_sink = Gst.ElementFactory.make("autoaudiosink", "audio_sink")
            if not audio_sink:
                audio_sink = Gst.ElementFactory.make("pulsesink", "audio_sink")
            if not audio_sink:
                audio_sink = Gst.ElementFactory.make("alsasink", "audio_sink")
            if audio_sink:
                self.player.set_property("audio-sink", audio_sink)
                self._setup_equalizer()
        except Exception:
            pass
        try:
            self.player.set_property("volume", DEFAULT_VOLUME)
        except Exception:
            pass
        try:
            self.player.connect("about-to-finish", self.on_song_finished)
        except Exception:
            pass
        self._setup_player_bus()
        GLib.idle_add(self._setup_system_audio_capture)

    def _setup_equalizer(self):
        try:
            self.equalizer = Gst.ElementFactory.make("equalizer-10bands", "equalizer")
            if not self.equalizer:
                return False
            self.player.set_property("audio-filter", self.equalizer)
            for i in range(10):
                try:
                    self.equalizer.set_property(f'band{i}', 0.0)
                except Exception:
                    pass
            return True
        except Exception:
            self.equalizer = None
            return False

    def _build_visualizer_audio_sink(self):
        """Build an audio-sink bin that delivers a known format to the visualizer.

        The visualizer's pad-probe reads buffer bytes as ``np.float32`` and runs
        an FFT to derive bass/mid/treb bands. That assumption only holds if the
        sink actually emits float32 stereo at a known sample rate. A bare
        ``autoaudiosink`` negotiates whatever the system sink wants (often S16LE
        integer PCM), so the float32 reinterpretation produces garbage and the
        visuals stop tracking the music.

        Force ``F32LE, 2ch, 44100 Hz`` so the probe's interpretation is correct.
        Falls back to ``None`` (caller falls back to a plain autoaudiosink) if any
        element is unavailable.
        """
        try:
            convert = Gst.ElementFactory.make("audioconvert", "vis_convert")
            resample = Gst.ElementFactory.make("audioresample", "vis_resample")
            capsfilter = Gst.ElementFactory.make("capsfilter", "vis_caps")
            sink = Gst.ElementFactory.make("autoaudiosink", "vis_sink")
            if not (convert and resample and capsfilter and sink):
                return None

            caps = Gst.Caps.from_string(
                "audio/x-raw, format=F32LE, rate=44100, channels=2, "
                "layout=interleaved"
            )
            capsfilter.set_property("caps", caps)

            audio_bin = Gst.Bin.new("vis_audio_sink")
            for element in (convert, resample, capsfilter, sink):
                audio_bin.add(element)

            if not (convert.link(resample) and resample.link(capsfilter)
                    and capsfilter.link(sink)):
                return None

            sink_pad = convert.get_static_pad("sink")
            audio_bin.add_pad(Gst.GhostPad.new("sink", sink_pad))

            self.audio_convert = convert
            self.audio_resample = resample
            self.audio_sink = audio_bin
            self._audio_sample_rate = 44100.0
            self._audio_channels = 2
            return audio_bin
        except Exception:
            return None

    def _setup_player_bus(self):
        try:
            bus = self.player.get_bus()
            if bus:
                if self._main_bus_signal_watch_attached:
                    try:
                        bus.remove_signal_watch()
                        self._main_bus_signal_watch_attached = False
                    except Exception:
                        pass
                if hasattr(self, '_bus_watch_id') and self._bus_watch_id is not None:
                    try:
                        bus.disconnect(self._bus_watch_id)
                        self._bus_watch_id = None
                    except Exception:
                        pass
                bus.add_signal_watch()
                self._main_bus_signal_watch_attached = True
                self._bus_watch_id = bus.connect("message", self.on_bus_message)
        except Exception:
            pass

    def _setup_system_audio_capture(self):
        """Set up system audio capture for real-time visualization.
        Called via GLib.idle_add - must return False.
        """
        self.system_audio_pipeline = Gst.Pipeline.new("system_audio_capture")
        source = Gst.ElementFactory.make("pulsesrc", "system_audio_source")
        if not source:
            raise RuntimeError("Could not create pulsesrc element for system audio capture")
        monitor_device = os.environ.get("LINAMP_PULSE_MONITOR", "default")
        source.set_property("device", monitor_device)
        audioconvert = Gst.ElementFactory.make("audioconvert", "system_audio_convert")
        audioresample = Gst.ElementFactory.make("audioresample", "system_audio_resample")
        capsfilter = Gst.ElementFactory.make("capsfilter", "system_audio_caps")
        capsfilter.set_property("caps", Gst.Caps.from_string("audio/x-raw,format=F32LE,channels=2,rate=44100"))
        if not all([audioconvert, audioresample, capsfilter]):
            raise RuntimeError("Could not create audio processing elements")
        self.system_audio_pipeline.add(source)
        self.system_audio_pipeline.add(audioconvert)
        self.system_audio_pipeline.add(audioresample)
        self.system_audio_pipeline.add(capsfilter)
        if not source.link(audioconvert):
            raise RuntimeError("Could not link source to converter")
        if not audioconvert.link(audioresample):
            raise RuntimeError("Could not link converter to resample")
        if not audioresample.link(capsfilter):
            raise RuntimeError("Could not link resample to capsfilter")
        sink_pad = capsfilter.get_static_pad("src")
        if sink_pad:
            self.system_audio_probe_id = sink_pad.add_probe(
                Gst.PadProbeType.BUFFER,
                self._system_audio_probe_callback,
                None
            )
        state_ret = self.system_audio_pipeline.set_state(Gst.State.PLAYING)
        if state_ret == Gst.StateChangeReturn.FAILURE:
            try:
                self.system_audio_pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.system_audio_pipeline = None
            self.system_audio_probe_id = None
            print("System Audio: capture pipeline unavailable (PulseAudio/monitor device "
                  "not present); continuing without system audio visualization.")
            return False
        return False

    def _system_audio_probe_callback(self, pad, info, user_data):
        """Callback for system audio capture."""
        try:
            buffer = info.get_buffer()
            if buffer:
                success, map_info = buffer.map(Gst.MapFlags.READ)
                if success:
                    try:
                        data = map_info.data
                        if data:
                            import struct
                            samples = len(data) // 8
                            if samples > 0:
                                sample_data = data[:min(len(data), 4096)]
                                audio_samples = []
                                for i in range(0, len(sample_data), 8):
                                    if i + 7 < len(sample_data):
                                        left = struct.unpack('<f', sample_data[i:i+4])[0]
                                        right = struct.unpack('<f', sample_data[i+4:i+8])[0]
                                        audio_samples.append((left + right) / 2.0)
                                if audio_samples and hasattr(self, 'update_audio'):
                                    self.update_audio(audio_samples)
                                    avg_level = sum(abs(s) for s in audio_samples) / len(audio_samples)
                                    if len(audio_samples) > 0 and len(audio_samples) % 60 == 0:
                                        print(f"System Audio: Samples: {len(audio_samples)}, avg level: {avg_level:.4f}")
                    finally:
                        buffer.unmap(map_info)
        except Exception as e:
            print(f"System Audio: Error in probe callback: {e}")
        return Gst.PadProbeReturn.OK

    def cleanup_pipeline(self, pipeline):
        if not pipeline:
            return
        if hasattr(self, '_pipeline_cleanup_in_progress'):
            if self._pipeline_cleanup_in_progress:
                return
        self._pipeline_cleanup_in_progress = True
        try:
            bus = pipeline.get_bus()
            if bus:
                if pipeline == self.player:
                    if self._main_bus_signal_watch_attached:
                        try:
                            bus.remove_signal_watch()
                            self._main_bus_signal_watch_attached = False
                        except Exception:
                            pass
                    if hasattr(self, '_bus_watch_id') and self._bus_watch_id is not None:
                        try:
                            bus.disconnect(self._bus_watch_id)
                            self._bus_watch_id = None
                        except Exception:
                            pass
        except Exception:
            pass
        try:
            pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        self._pipeline_cleanup_in_progress = False

    def cleanup_timers(self):
        if hasattr(self, '_active_timers'):
            for timer_id in list(self._active_timers):
                self._safe_source_remove(timer_id)
            self._active_timers.clear()
        timer_attributes = [
            'beat_detection_timer',
            'update_display_timer',
            'auto_save_timer'
        ]
        for timer_attr in timer_attributes:
            if hasattr(self, timer_attr):
                timer_id = getattr(self, timer_attr)
                if timer_id:
                    try:
                        GLib.source_remove(timer_id)
                    except Exception:
                        pass
                    setattr(self, timer_attr, 0)

    def full_cleanup(self):
        if hasattr(self, '_cleanup_in_progress') and self._cleanup_in_progress:
            return
        self._cleanup_in_progress = True
        self.on_stop(None)
        if hasattr(self, 'player') and self.player:
            try:
                self.player.set_state(Gst.State.NULL)
                self.player.get_state(Gst.CLOCK_TIME_NONE)
            except Exception:
                pass
        self.cleanup_timers()
        if hasattr(self, 'player') and self.player:
            self.cleanup_pipeline(self.player)
            self.player = None
        if hasattr(self, 'equalizer') and self.equalizer:
            try:
                self.equalizer.set_state(Gst.State.NULL)
            except Exception:
                pass
            self.equalizer = None
        if hasattr(self, 'system_audio_pipeline') and self.system_audio_pipeline:
            try:
                self.system_audio_pipeline.set_state(Gst.State.NULL)
                self.system_audio_pipeline.get_state(Gst.CLOCK_TIME_NONE)
                if hasattr(self, 'system_audio_probe_id') and self.system_audio_probe_id is not None:
                    capsfilter = self.system_audio_pipeline.get_by_name("system_audio_caps")
                    if capsfilter:
                        sink_pad = capsfilter.get_static_pad("src")
                        if sink_pad:
                            sink_pad.remove_probe(self.system_audio_probe_id)
                    self.system_audio_probe_id = None
                self.system_audio_pipeline = None
            except Exception as e:
                print(f"Warning: Error cleaning up system audio pipeline: {e}")
        if hasattr(self, 'player_tab') and hasattr(self.player_tab, 'cover_art_widget'):
            try:
                self.player_tab.cover_art_widget.cleanup()
            except Exception:
                pass
        # Intentionally avoid pkill-based cleanup.
        # This app can be embedded/restarted; pkilling other processes can race
        # with GL/audio resources and destabilize rendering.
        self._cleanup_in_progress = False


    def on_close_request(self, window):
        try:
            self.full_cleanup()
        except Exception:
            pass
        if self.get_application():
            self.get_application().quit()
        return True

    def setup_drag_drop(self):
        self.dnd = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        self.dnd.connect("drop", self.on_file_dropped)
        self.add_controller(self.dnd)

    def on_file_dropped(self, drop, file, x, y):
        path = file.get_path()
        if path:
            self.add_to_playlist([path])
            if not self.playing and self.playlist:
                self.play_track(0)
        return True

    def _set_player_state_thread_safe(self, state):
        try:
            return self.player.set_state(state)
        except Exception:
            return Gst.StateChangeReturn.FAILURE

    def _set_player_property_thread_safe(self, property_name, value):
        try:
            self.player.set_property(property_name, value)
            return True
        except Exception:
            return False

    def play_file(self, filepath, title=None):
        if not os.path.exists(filepath):
            return False
        if not hasattr(self, 'player') or not self.player:
            return False
        try:
            if hasattr(self.player_tab, 'track_label'):
                GLib.idle_add(self.player_tab.track_label.set_text, f"Loading: {os.path.basename(filepath)}...")
            uri = f"file://{os.path.abspath(filepath)}"
            self.player.set_state(Gst.State.NULL)
            self.player.set_property("uri", uri)
            state_change = self.player.set_state(Gst.State.PLAYING)
            if state_change == Gst.StateChangeReturn.FAILURE:
                GLib.idle_add(self.set_status_message, "Failed to start playback")
                return False
            GLib.timeout_add(100, self._verify_playback_state, filepath, title)
            return True
        except Exception:
            return False

    def _verify_playback_state(self, filepath, title):
        try:
            if not hasattr(self, 'player') or not self.player:
                return False
            current_state = self.player.get_state(Gst.CLOCK_TIME_NONE)[1]
            if current_state == Gst.State.PLAYING:
                self.playing = True
                track_name = title or os.path.basename(filepath)
                GLib.idle_add(self.player_tab.track_label.set_text, track_name)
                GLib.idle_add(self.set_title, f"LinAmp - {track_name}")
            if hasattr(self, 'visualizer_tab') and self.visualizer_tab is not None:
                self.visualizer_tab.player = self.player
                if hasattr(self.visualizer_tab, '_rendering_enabled'):
                    self.visualizer_tab._rendering_enabled = True
                if hasattr(self.visualizer_tab, '_reconnect_audio_probe'):
                    self.visualizer_tab._reconnect_audio_probe()
            else:
                if self.playlist and self.current_track < len(self.playlist) - 1:
                    GLib.idle_add(self.play_track, self.current_track + 1)


        except Exception:
            if self.current_track < 0:
                self.play_track(0)
            else:
                try:
                    success, state = self.player.get_state(Gst.CLOCK_TIME_NONE)
                    current_uri = self.player.get_property("uri") if hasattr(self.player, 'get_property') else None
                    if current_uri and self.playlist and self.current_track < len(self.playlist):
                        current_path = self.playlist[self.current_track].path
                        expected_uri = f"file://{os.path.abspath(current_path)}"
                        needs_restart = (state != Gst.State.PLAYING) and (current_uri != expected_uri)
                    else:
                        needs_restart = state != Gst.State.PLAYING
                    if state == Gst.State.NULL or needs_restart:
                        if 0 <= self.current_track < len(self.playlist):
                            self.play_track(self.current_track)
                    else:
                        if state in [Gst.State.READY, Gst.State.PAUSED]:
                            self.player.seek_simple(
                                Gst.Format.TIME,
                                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                                0
                            )
                        GLib.idle_add(self.player.set_state, Gst.State.PLAYING)
                        self.playing = True
                        if self.beat_aware_enabled:
                            self.start_beat_detection()
                except Exception:
                    if 0 <= self.current_track < len(self.playlist):
                        self.play_track(self.current_track)

    def on_pause(self, button):
        if self.playing:
            GLib.idle_add(self.player.set_state, Gst.State.PAUSED)
            self.playing = False
            self.stop_beat_detection()

    def on_play(self, button):
        if self.playing:
            self.on_pause(button)
        else:
            if self.current_track >= 0 and self.current_track < len(self.playlist):
                self.play_track(self.current_track)
            elif self.playlist:
                self.play_track(0)
            else:
                GLib.idle_add(self.set_status_message, "No tracks in playlist - add files first")

    def stop(self):
        self.on_stop(None)

    def on_stop(self, button):
        if hasattr(self, '_cleanup_in_progress') and self._cleanup_in_progress:
            return
        if hasattr(self, 'player') and self.player:
            try:
                self.player.disconnect_by_func(self.on_song_finished)
            except (TypeError, Exception):
                pass
            try:
                self.player.set_state(Gst.State.NULL)
                self.cleanup_pipeline(self.player)
            except Exception:
                pass
        self.playing = False
        self.stop_beat_detection()
        if hasattr(self, 'player_tab'):
            self.player_tab.time_label.set_text("0:00 / 0:00")
            self.player_tab.progress.set_value(0)
            self.player_tab.track_label.set_text("No track playing")
            if hasattr(self.player_tab, 'cover_art_widget'):
                self.player_tab.cover_art_widget.clear()
        self.set_title("LinAmp - XMMS Style")
        self.save_settings_on_stop()

    def on_volume_changed(self, scale):
        volume = scale.get_value() / 100.0
        try:
            if hasattr(self, 'player_tab') and hasattr(self.player_tab, 'volume_label'):
                self.player_tab.volume_label.set_label(f"{int(scale.get_value())}%")
            self.player.set_property("volume", volume)
            if hasattr(self, '_is_muted') and self._is_muted and volume > 0:
                self._is_muted = False
                self._previous_volume = volume
            self._update_volume_button_icon()
        except Exception:
            pass
        self.auto_save_settings()

    def on_mute_toggle(self, button):
        if not hasattr(self, 'player') or not self.player:
            return
        try:
            current_volume = self.player.get_property("volume")
            if hasattr(self, '_is_muted') and self._is_muted:
                self.player.set_property("volume", self._previous_volume)
                self._is_muted = False
            elif current_volume > 0:
                self._previous_volume = current_volume
                self.player.set_property("volume", 0.0)
                self._is_muted = True
            else:
                self._previous_volume = 1.0
                self.player.set_property("volume", 1.0)
                self._is_muted = False
            self._update_volume_button_icon()
            actual_volume = self.player.get_property("volume")
            if hasattr(self, 'player_tab') and hasattr(self.player_tab, 'volume_scale'):
                self.player_tab.volume_scale.set_value(actual_volume * 100)
        except Exception:
            pass

    def _update_volume_button_icon(self):
        if not hasattr(self, 'player_tab') or not hasattr(self.player_tab, 'volume_button') or not self.player:
            return
        current_volume = self.player.get_property("volume")
        if current_volume == 0 or getattr(self, '_is_muted', False):
            icon_name = "audio-volume-muted-symbolic"
        elif current_volume < 0.3:
            icon_name = "audio-volume-low-symbolic"
        elif current_volume < 0.7:
            icon_name = "audio-volume-medium-symbolic"
        else:
            icon_name = "audio-volume-high-symbolic"
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        self.player_tab.volume_button.set_child(icon)

    def on_song_finished(self, element):
        GLib.idle_add(self._reset_pipeline_state)

    def _reset_pipeline_state(self):
        if self.player is not None:
            try:
                self.player.set_state(Gst.State.READY)
            except Exception:
                pass
        else:
            pass
        self.playing = False
        if self.auto_play_next:
            next_index = self.get_next_track_index()
            if next_index is not None:
                self.play_next_track()
            else:
                if hasattr(self.player_tab, 'cover_art_widget'):
                    self.player_tab.cover_art_widget.clear()

    def on_bus_message(self, bus, message):
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            error_msg = str(err.message).lower()
            if "internal" in error_msg and "stream" in error_msg:
                GLib.idle_add(self.set_status_message, "Internal stream error - recovering...")
                try:
                    current_uri = self.player.get_property("uri")
                    GLib.idle_add(self.player.set_state, Gst.State.NULL)
                    GLib.timeout_add(100, lambda: GLib.idle_add(self._attempt_recovery, current_uri))
                except Exception:
                    GLib.idle_add(self.set_status_message, "Recovery failed - trying next track")
                    if self.playlist and self.current_track < len(self.playlist) - 1:
                        GLib.idle_add(self.play_track, self.current_track + 1)
            else:
                GLib.idle_add(self.set_status_message, f"Error: {err.message}")
                if self.playlist and self.current_track < len(self.playlist) - 1:
                    GLib.idle_add(self.play_track, self.current_track + 1)
        elif message.type == Gst.MessageType.EOS:
            GLib.idle_add(self._handle_eos)
        return Gst.BusSyncReply.PASS

    def _handle_eos(self):
        self.player.set_state(Gst.State.READY)
        self.playing = False
        if self.auto_play_next:
            if self.repeat_mode == "one":
                self.play_track(self.current_track)
            elif self.repeat_mode == "all":
                self.play_next_track()
            else:
                next_index = self.get_next_track_index()
                if next_index is not None:
                    self.play_track(next_index)
                else:
                    self.on_stop(None)
        else:
            self.on_stop(None)

    def _on_state_changed(self, bus, message):
        if message.src == self.player:
            old_state, new_state, pending_state = message.parse_state_changed()
            if new_state == Gst.State.PLAYING:
                self.playing = True
            elif new_state in [Gst.State.PAUSED, Gst.State.NULL]:
                self.playing = False
        return Gst.BusSyncReply.PASS

    def _attempt_recovery(self, uri):
        try:
            if not uri:
                GLib.idle_add(self.set_status_message, "Recovery failed - trying next track")
                return False
            self.player.set_property("uri", uri)
            GLib.idle_add(self.player.set_state, Gst.State.PLAYING)
            GLib.timeout_add(500, self._check_recovery_success)
            return False
        except Exception:
            GLib.idle_add(self.set_status_message, "Recovery error - trying next track")
            if self.playlist and self.current_track < len(self.playlist) - 1:
                GLib.idle_add(self.play_track, self.current_track + 1)
            return False

    def _check_recovery_success(self):
        try:
            success, state = self.player.get_state(Gst.State.NULL)
            if success and state == Gst.State.PLAYING:
                GLib.idle_add(self.set_status_message, "Recovery successful")
                return False
            else:
                GLib.idle_add(self.set_status_message, "Recovery failed - trying next track")
                if self.playlist and self.current_track < len(self.playlist) - 1:
                    GLib.idle_add(self.play_track, self.current_track + 1)
                return False
        except Exception:
            GLib.idle_add(self.set_status_message, "Recovery check failed - trying next track")
            if self.playlist and self.current_track < len(self.playlist) - 1:
                GLib.idle_add(self.play_track, self.current_track + 1)
            return False

    def update_display(self):
        try:
            if hasattr(self, 'player') and self.player and self.playing:
                success, position = self.player.query_position(Gst.Format.TIME)
                if not success:
                    position = 0
                success, duration = self.player.query_duration(Gst.Format.TIME)
                if not success:
                    duration = 0
                position_sec = position // Gst.SECOND
                duration_sec = duration // Gst.SECOND
                pos_str = f"{position_sec // 60}:{position_sec % 60:02d}"
                dur_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"
                if hasattr(self, 'player_tab') and self.player_tab:
                    if hasattr(self.player_tab, 'time_label') and self.player_tab.time_label:
                        self.player_tab.time_label.set_text(f"{pos_str} / {dur_str}")
                        self.player_tab.time_label.add_css_class("time-label")
                    if hasattr(self.player_tab, 'progress') and self.player_tab.progress:
                        if duration_sec > 0 and position_sec >= 0:
                            fraction = position_sec / duration_sec
                            fraction = max(0.0, min(1.0, fraction))
                            current_value = self.player_tab.progress.get_value()
                            if abs(fraction * 100 - current_value) > 0.1:
                                self.player_tab.progress.set_value(fraction * 100)
                                if fraction > 0.75:
                                    self.player_tab.progress.add_css_class("near-end")
                                    self.player_tab.progress.remove_css_class("halfway")
                                elif fraction > 0.5:
                                    self.player_tab.progress.remove_css_class("near-end")
                                    self.player_tab.progress.add_css_class("halfway")
                                else:
                                    self.player_tab.progress.remove_css_class("near-end")
                                    self.player_tab.progress.remove_css_class("halfway")
                        else:
                            self.player_tab.progress.set_value(0)
                    if hasattr(self.player_tab, 'track_label') and self.player_tab.track_label:
                        self.player_tab.track_label.add_css_class("playing-indicator")
                try:
                    if hasattr(self, "header_nowplaying") and self.header_nowplaying and self.playlist and 0 <= self.current_track < len(self.playlist):
                        current = self.playlist[self.current_track]
                        title = getattr(current, "title", "") or "Playing"
                        self.header_nowplaying.set_text(f"{title}  •  {pos_str} / {dur_str}")
                except Exception:
                    pass
            status_parts = []
            if self.repeat_mode != "none":
                status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
            if self.shuffle_mode:
                status_parts.append("SHUFFLE")
            if not self.auto_play_next:
                status_parts.append("NO AUTO-NEXT")
            if self.beat_aware_enabled:
                status_parts.append("BEAT-AWARE")
            status_text = " | ".join(status_parts) if status_parts else "Ready"
            if self.playing:
                status_text = f"🎵 {status_text}"
                if hasattr(self, 'status_bar'):
                    self.status_bar.add_css_class("playing")
            else:
                if hasattr(self, 'status_bar'):
                    self.status_bar.remove_css_class("playing")
                try:
                    if hasattr(self, "header_nowplaying") and self.header_nowplaying:
                        if self.playlist and 0 <= self.current_track < len(self.playlist):
                            current = self.playlist[self.current_track]
                            title = getattr(current, "title", "") or "Ready"
                            self.header_nowplaying.set_text(title)
                        else:
                            self.header_nowplaying.set_text("Ready")
                except Exception:
                    pass
            GLib.idle_add(self.set_status_message, status_text)
        except Exception as e:
            if "Gtk.Statusbar.remove() takes exactly 3 arguments (2 given)" in str(e):
                pass
            elif "query_position" in str(e) or "query_duration" in str(e):
                pass
            elif "has no attribute" in str(e) and ("time_label" in str(e) or "progress" in str(e) or "track_label" in str(e)):
                pass
            else:
                pass
        return True

    def focus_playlist_search(self):
        try:
            if hasattr(self, "notebook"):
                self.notebook.set_current_page(2)
            if hasattr(self, "playlist_tab") and hasattr(self.playlist_tab, "search_entry") and self.playlist_tab.search_entry:
                self.playlist_tab.search_entry.grab_focus()
        except Exception:
            pass

    def play_track(self, index):
        if 0 <= index < len(self.playlist):
            item = self.playlist[index]
            if self.repeat_mode == "one" and index == self.current_track:
                GLib.idle_add(self.player.set_state, Gst.State.NULL)
            if self.play_file(item.path, item.title):
                self.current_track = index
                self.save_settings_on_track_change()
                if self.shuffle_mode and index in self.shuffled_indices:
                    self.shuffle_position = self.shuffled_indices.index(index)
                if hasattr(self, 'playlist_tab') and hasattr(self.playlist_tab, 'selection_model'):
                    try:
                        self.playlist_tab.selection_model.set_selected(index)
                    except Exception:
                        pass
                self._load_cover_art_for_track(item)
                return True
        return False

    def _load_cover_art_for_track(self, item):
        if not hasattr(self, 'player_tab') or not hasattr(self.player_tab, 'cover_art_widget'):
            return
        cover_widget = self.player_tab.cover_art_widget
        current_audio_path = cover_widget.current_cover_path
        if current_audio_path:
            if current_audio_path.startswith(str(COVER_CACHE_DIR)):
                try:
                    cache_filename = Path(current_audio_path).name
                    cache_key = cache_filename.split('.')[0]
                    item_cache_key = CoverArt.get_cache_key(item.path)
                    if cache_key == item_cache_key:
                        return
                except Exception:
                    pass
            elif current_audio_path == item.path:
                return
        allow_download = getattr(self, 'auto_download_covers', False)
        cover, cache_path = extract_cover_art(item.path, allow_download=False)
        if cover:
            cover_widget.load_cover(cover, item.path)
        else:
            if allow_download:
                import threading
                def download_and_load():
                    cover = download_cover_for_audio(item.path)
                    if cover:
                        GLib.idle_add(cover_widget.load_cover, cover, item.path)
                threading.Thread(target=download_and_load, daemon=True).start()
            else:
                cover_widget.load_cover(None, item.path)

    def on_prev(self, button):
        self.play_previous_track()

    def play_previous_track(self):
        if not self.playlist:
            return
        prev_index = self.get_previous_track_index()
        if prev_index is not None:
            self.play_track(prev_index)

    def get_previous_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_previous_shuffled_index()
        else:
            if self.current_track > 0:
                return self.current_track - 1
            elif self.repeat_mode == "all":
                return len(self.playlist) - 1
            else:
                return None

    def get_previous_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position > 0:
            self.shuffle_position -= 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.shuffle_position = len(self.shuffled_indices) - 1
            return self.shuffled_indices[self.shuffle_position]
        else:
            return None

    def toggle_repeat_mode(self):
        modes = ["none", "one", "all"]
        current_index = modes.index(self.repeat_mode)
        self.repeat_mode = modes[(current_index + 1) % len(modes)]
        self.update_status_display()
        self.auto_save_settings()

    def toggle_shuffle_mode(self):
        self.shuffle_mode = not self.shuffle_mode
        if self.shuffle_mode:
            self.regenerate_shuffle_list()
            if self.current_track >= 0:
                try:
                    self.shuffle_position = self.shuffled_indices.index(self.current_track)
                except ValueError:
                    self.shuffle_position = -1
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_play_next(self):
        self.auto_play_next = not self.auto_play_next
        self.update_status_display()
        self.auto_save_settings()

    def toggle_beat_aware(self):
        self.beat_aware_enabled = not self.beat_aware_enabled
        self.update_status_display()
        self.auto_save_settings()

    def toggle_auto_download_covers(self):
        if not COVER_DOWNLOAD_AVAILABLE:
            self.set_status_message("Cover download requires requests and PIL libraries")
            return
        self.auto_download_covers = not self.auto_download_covers
        status = "enabled" if self.auto_download_covers else "disabled"
        self.set_status_message(f"Auto download covers: {status}")
        self.update_status_display()
        self.auto_save_settings()
        if self.auto_download_covers and self.current_track >= 0 and self.current_track < len(self.playlist):
            current_item = self.playlist[self.current_track]
            self._load_cover_art_for_track(current_item)

    def auto_save_settings(self):
        if hasattr(self, '_auto_save_timer') and self._auto_save_timer:
            self._safe_source_remove(self._auto_save_timer)
        self._auto_save_timer = GLib.timeout_add(AUTO_SAVE_DELAY, self._auto_save_callback)
        self._active_timers.add(self._auto_save_timer)

    def _auto_save_callback(self):
        self._update_settings_from_state()
        self.save_settings()
        if self.beat_aware_enabled:
            self.start_beat_detection()
        else:
            self.stop_beat_detection()
        self._auto_save_timer = 0
        return False

    def update_status_display(self):
        status_parts = []
        if self.repeat_mode != "none":
            status_parts.append(f"Repeat: {self.repeat_mode.upper()}")
        if self.shuffle_mode:
            status_parts.append("SHUFFLE")
        if not self.auto_play_next:
            status_parts.append("NO AUTO-NEXT")
        if self.beat_aware_enabled:
            status_parts.append("BEAT-AWARE")
        if self.auto_download_covers:
            status_parts.append("AUTO-DL")
        status_text = " | ".join(status_parts) if status_parts else "Ready"
        GLib.idle_add(self.set_status_message, status_text)
        self.update_button_states()

    def update_button_states(self):
        if not hasattr(self, 'player_tab'):
            return
        if hasattr(self.player_tab, 'repeat_btn'):
            if self.repeat_mode == "none":
                self.player_tab.repeat_btn.remove_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: Off")
            elif self.repeat_mode == "one":
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: One Track")
            else:
                self.player_tab.repeat_btn.add_css_class("active")
                self.player_tab.repeat_btn.set_tooltip_text("Repeat: All")
        if hasattr(self.player_tab, 'shuffle_btn'):
            if self.shuffle_mode:
                self.player_tab.shuffle_btn.add_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: On")
            else:
                self.player_tab.shuffle_btn.remove_css_class("active")
                self.player_tab.shuffle_btn.set_tooltip_text("Shuffle: Off")
        if hasattr(self.player_tab, 'beat_btn'):
            if self.beat_aware_enabled:
                self.player_tab.beat_btn.add_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: On")
            else:
                self.player_tab.beat_btn.remove_css_class("active")
                self.player_tab.beat_btn.set_tooltip_text("Beat Detection: Off")
        if hasattr(self.player_tab, 'download_btn'):
            if self.auto_download_covers:
                self.player_tab.download_btn.add_css_class("active")
                self.player_tab.download_btn.set_tooltip_text("Auto Download Covers: On")
            else:
                self.player_tab.download_btn.remove_css_class("active")
                self.player_tab.download_btn.set_tooltip_text("Auto Download Covers: Off")
        if hasattr(self.player_tab, 'autonext_btn'):
            if self.auto_play_next:
                self.player_tab.autonext_btn.add_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: On")
            else:
                self.player_tab.autonext_btn.remove_css_class("active")
                self.player_tab.autonext_btn.set_tooltip_text("Auto Next: Off")

    def on_next(self, button):
        self.play_next_track()

    def play_next_track(self):
        if not self.playlist:
            return
        next_index = self.get_next_track_index()
        if next_index is not None:
            if hasattr(self, 'playlist_tab') and hasattr(self.playlist_tab, 'selection_model'):
                try:
                    self.playlist_tab.selection_model.set_selected(next_index)
                except Exception:
                    pass
            self.play_track(next_index)

    def get_next_track_index(self):
        if not self.playlist:
            return None
        if self.shuffle_mode:
            return self.get_next_shuffled_index()
        else:
            if self.current_track < len(self.playlist) - 1:
                return self.current_track + 1
            elif self.repeat_mode == "all":
                return 0
            else:
                return None

    def get_next_shuffled_index(self):
        if not self.shuffled_indices:
            self.regenerate_shuffle_list()
        if self.shuffle_position < len(self.shuffled_indices) - 1:
            self.shuffle_position += 1
            return self.shuffled_indices[self.shuffle_position]
        elif self.repeat_mode == "all":
            self.regenerate_shuffle_list()
            self.shuffle_position = 0
            return self.shuffled_indices[0]
        else:
            return None

    def regenerate_shuffle_list(self):
        self.original_indices = list(range(len(self.playlist)))
        self.shuffled_indices = self.original_indices.copy()
        random.shuffle(self.shuffled_indices)
        self.shuffle_position = -1

    def start_beat_detection(self):
        if not self.beat_aware_enabled or not self.playing:
            return
        self.beat_detection_timer = GLib.timeout_add(BEAT_DETECTION_INTERVAL, self.detect_beat)
        self._active_timers.add(self.beat_detection_timer)

    def stop_beat_detection(self):
        self._safe_source_remove(self.beat_detection_timer)
        self.beat_detection_timer = 0

    def detect_beat(self):
        if not self.beat_aware_enabled or not self.playing:
            return False
        try:
            current_time = time.time()
            if current_time - self.last_beat_time < 0.1:
                return False
            levels = []
            if hasattr(self, 'audio_levels') and self.audio_levels:
                levels = self.audio_levels
            if len(levels) > 0:
                    avg_level = sum(levels) / len(levels)
                    peak_level = max(levels)
                    if hasattr(self, '_last_avg_level'):
                        level_increase = avg_level - self._last_avg_level
                        if (level_increase > 0.15 and peak_level > 0.7) or peak_level > 0.95:
                            self.last_beat_time = current_time
                            self._last_avg_level = avg_level
                            current_beat_time = current_time
                            if hasattr(self, '_last_beat_timestamp'):
                                beat_interval = current_beat_time - self._last_beat_timestamp
                                if 0.2 <= beat_interval <= 2.0:
                                    self.beat_interval_history.append(beat_interval)
                                    if len(self.beat_interval_history) > BEAT_HISTORY_SIZE:
                                        self.beat_interval_history.pop(0)
                                    if len(self.beat_interval_history) >= 2:
                                        sorted_intervals = sorted(self.beat_interval_history)
                                        median_interval = sorted_intervals[len(sorted_intervals) // 2]
                                        if median_interval > 0:
                                            bpm = 60.0 / median_interval
                                            if 60 <= bpm <= 200:
                                                self.on_beat_detected(bpm)
                                                return True
                            self._last_beat_timestamp = current_beat_time
                            return True
                    self._last_avg_level = avg_level
            return False
        except Exception:
            return False

    def on_beat_detected(self, bpm):
        pass

    def add_to_playlist(self, file_paths):
        added_items = []
        for path in file_paths:
            try:
                normalized_path = os.path.abspath(os.path.expanduser(str(path)))
                if '../' in path or '\0' in path or path.startswith('/proc/') or path.startswith('/sys/'):
                    continue
                if not os.path.exists(normalized_path):
                    continue
                if not os.path.isfile(normalized_path):
                    continue
            except Exception:
                continue
            item = PlaylistItem(path=normalized_path, title=os.path.basename(normalized_path))
            self.playlist.append(item)
            added_items.append(item)
        if hasattr(self, 'playlist_tab') and added_items:
            for item in added_items:
                self.playlist_tab.playlist_store.append(f"{item.title}|{item.id}")
            self.playlist_tab.update_statistics()
        self.save_playlist()

    def add_folder_to_playlist(self, folder_path):
        try:
            normalized_path = os.path.abspath(os.path.expanduser(str(folder_path)))
            if '../' in folder_path or '\0' in folder_path or folder_path.startswith('/proc/') or folder_path.startswith('/sys/'):
                return
            if not os.path.exists(normalized_path) or not os.path.isdir(normalized_path):
                return
        except Exception:
            return
        GLib.idle_add(self.set_status_message, f"Scanning folder: {os.path.basename(normalized_path)}...")

        def scan_folder():
            added_items = []
            try:
                for root, dirs, files in os.walk(normalized_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in AUDIO_EXTENSIONS:
                            try:
                                rel_path = os.path.relpath(file_path, normalized_path)
                                title = rel_path if rel_path != file_path else file
                                item = PlaylistItem(path=file_path, title=title)
                                added_items.append(item)
                            except Exception:
                                pass
            except Exception:
                pass
            GLib.idle_add(self._on_folder_scan_complete, added_items)
        import threading
        thread = threading.Thread(target=scan_folder, daemon=True)
        thread.start()

    def _on_folder_scan_complete(self, added_items):
        if added_items:
            self.playlist.extend(added_items)
            if hasattr(self, 'playlist_tab'):
                for item in added_items:
                    self.playlist_tab.playlist_store.append(f"{item.title}|{item.id}")
                self.playlist_tab.update_statistics()
            self.save_playlist()
            GLib.idle_add(self.set_status_message, f"Added {len(added_items)} tracks to playlist")
        else:
            GLib.idle_add(self.set_status_message, "No audio files found in folder")

    def save_playlist(self, filepath: str = None) -> bool:
        if not hasattr(self, 'playlist') or not self.playlist:
            pass
            return False
        if not filepath:
            playlist_dir = os.path.expanduser("~/.config/linamp")
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
            playlist_dir = os.path.dirname(filepath)
        try:
            os.makedirs(playlist_dir, exist_ok=True)
        except (OSError, PermissionError):
            pass
            return False
        try:
            playlist_data = []
            for item in self.playlist:
                try:
                    if hasattr(item, 'to_dict') and callable(item.to_dict):
                        if hasattr(item, 'path') and item.path and os.path.exists(item.path):
                            playlist_data.append(item.to_dict())
                        else:
                            pass
                except Exception:
                    pass
            if not playlist_data:
                pass
                return False
        except Exception:
            pass
            return False
        temp_file = None
        try:
            fd, temp_file = tempfile.mkstemp(
                prefix='.playlist_',
                suffix='.tmp',
                dir=playlist_dir,
                text=True
            )
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(playlist_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(filepath):
                os.replace(temp_file, filepath)
            else:
                os.rename(temp_file, filepath)
            return True
        except (OSError, IOError):
            pass
        except (OSError, IOError, TypeError):
            pass
        except Exception:
            pass
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        return False

    def _clear_playlist_store(self) -> bool:
        if not hasattr(self, 'playlist_tab') or not hasattr(self.playlist_tab, 'playlist_store'):
            pass
            return False
        store = self.playlist_tab.playlist_store
        frozen = False
        try:
            store.freeze_notify()
            frozen = True
            if hasattr(store, 'splice') and callable(store.splice):
                store.splice(0, store.get_n_items(), [])
            else:
                while store.get_n_items() > 0:
                    store.remove(store.get_n_items() - 1)
            return True
        except Exception:
            pass
            return False
        finally:
            if frozen:
                try:
                    store.thaw_notify()
                except Exception:
                    pass

    def _update_playlist_display(self) -> bool:
        if not hasattr(self, 'playlist_tab') or not hasattr(self.playlist_tab, 'playlist_store'):
            return False
        store = self.playlist_tab.playlist_store
        frozen = False
        try:
            if not self.playlist:
                self._clear_playlist_store()
                return True
            store.freeze_notify()
            frozen = True
            current_count = store.get_n_items()
            target_count = len(self.playlist)
            if current_count > target_count:
                if hasattr(store, 'splice') and callable(store.splice):
                    store.splice(target_count, current_count - target_count, [])
                else:
                    while store.get_n_items() > target_count:
                        store.remove(store.get_n_items() - 1)
            display_names = []
            for i, item in enumerate(self.playlist):
                try:
                    if isinstance(item, PlaylistItem):
                        display_names.append(f"{item.get_display_name()}|{item.id}")
                    elif isinstance(item, dict):
                        title = item.get('title') or os.path.basename(item.get('path', ''))
                        path = item.get('path', '')
                        track_id = item.get('id') or hashlib.md5(f"{path}:{title}".encode()).hexdigest()[:16]
                        display_names.append(f"{title}|{track_id}")
                    else:
                        s = str(item)
                        track_id = hashlib.md5(s.encode()).hexdigest()[:16]
                        display_names.append(f"{s}|{track_id}")
                except Exception:
                    display_names.append("<Invalid Item>|invalid")
            if hasattr(store, 'splice') and callable(store.splice):
                if current_count < target_count:
                    store.splice(current_count, 0, display_names[current_count:])
                else:
                    store.splice(0, min(current_count, target_count), display_names[:target_count])
            else:
                while store.get_n_items() > 0:
                    store.remove(store.get_n_items() - 1)
                for name in display_names:
                    store.append(name)
            return True
        except Exception:
            return False
        finally:
            if frozen:
                try:
                    store.thaw_notify()
                except Exception:
                    pass
            if hasattr(self, 'playlist_tab') and hasattr(self.playlist_tab, 'update_statistics'):
                self.playlist_tab.update_statistics()

    def load_playlist(self, filepath: str = None) -> bool:
        if not filepath:
            playlist_dir = os.path.expanduser("~/.config/linamp")
            filepath = os.path.join(playlist_dir, "playlist.json")
        else:
            filepath = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(filepath):
            return False
        if not os.access(filepath, os.R_OK):
            pass
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        pass
                        return False
                except json.JSONDecodeError:
                    pass
                    return False
        except (IOError, OSError):
            pass
            return False
        except Exception:
            pass
            return False
        playlist = []
        invalid_items = 0
        for i, item_data in enumerate(data, 1):
            try:
                if not isinstance(item_data, dict):
                    pass
                    invalid_items += 1
                    continue
                try:
                    item = PlaylistItem.from_dict(item_data)
                    if not item.exists():
                        pass
                    playlist.append(item)
                except ValueError:
                    pass
                    invalid_items += 1
            except Exception:
                pass
                invalid_items += 1
        self.playlist = playlist
        if hasattr(self, 'playlist_tab') and self.playlist_tab:
            self._clear_playlist_store()
            self._update_playlist_display()
        return len(playlist) > 0

    def cleanup_invalid_tracks(self):
        if not self.playlist:
            return
        original_count = len(self.playlist)
        valid_tracks = []
        for item in self.playlist:
            if item.exists():
                valid_tracks.append(item)
            else:
                pass
        if len(valid_tracks) != original_count:
            self.playlist = valid_tracks
            self._update_playlist_display()
            self.save_playlist()
            self.set_status_message(f"Removed {original_count - len(valid_tracks)} invalid tracks")

class LinAmpApp(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='org.example.linamp.xmms')
        self.win = None
        Gst.init(None)
        self.set_property('application-id', 'org.example.linamp.xmms')
        Gtk.Settings.get_default().set_property('gtk-application-prefer-dark-theme', True)

    def do_activate(self):
        try:
            if not self.win:
                self.win = WinampWindow(application=self, title="LinAmp")
                self.win.present()
                self.win.set_visible(True)
                self.win.grab_focus()
                GLib.timeout_add(500, self.apply_dark_theme_css)
            else:
                self.win.present()
        except Exception as e:
            print(f"Error activating application: {e}")

    def apply_dark_theme_css(self):
        css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme", "gtk.css")
        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(css_file)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_USER
        )
        return False

    def do_shutdown(self):
        if hasattr(self, 'win') and self.win is not None:
            self.win.full_cleanup()
        Gtk.Application.do_shutdown(self)

    def do_startup(self):
        Gtk.Application.do_startup(self)
        self.setup_menu()
        resize_action = Gio.SimpleAction.new("resize_to_cover_art", None)
        resize_action.connect("activate", self.on_resize_to_cover_art)
        self.add_action(resize_action)
        self.set_accels_for_action("app.resize_to_cover_art", ["<Control>r"])
        visualizer_action = Gio.SimpleAction.new("show_visualizer", None)
        visualizer_action.connect("activate", self.on_show_visualizer)
        self.add_action(visualizer_action)
        self.set_accels_for_action("app.show_visualizer", ["<Control>v"])
        if len(sys.argv) > 1:
            if not self.win:
                self.win = WinampWindow(application=self, title="LinAmp")
            self.win.add_to_playlist(sys.argv[1:])
            if self.win.playlist:
                self.win.cleanup_invalid_tracks()
                if self.win.playlist:
                    self.win.play_track(0)

    def setup_menu(self):
        open_action = Gio.SimpleAction.new("open", None)
        open_action.connect("activate", self.on_open)
        open_folder_action = Gio.SimpleAction.new("open_folder", None)
        open_folder_action.connect("activate", self.on_open_folder)
        add_to_playlist_action = Gio.SimpleAction.new("add_to_playlist", None)
        add_to_playlist_action.connect("activate", self.on_add_to_playlist)
        add_folder_to_playlist_action = Gio.SimpleAction.new("add_folder_to_playlist", None)
        add_folder_to_playlist_action.connect("activate", self.on_add_folder_to_playlist)
        cleanup_action = Gio.SimpleAction.new("cleanup_invalid", None)
        cleanup_action.connect("activate", self.on_cleanup_invalid)
        resize_cover_action = Gio.SimpleAction.new("resize_to_cover_art", None)
        resize_cover_action.connect("activate", self.on_resize_to_cover_art)
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(open_action)
        self.add_action(open_folder_action)
        self.add_action(add_to_playlist_action)
        self.add_action(add_folder_to_playlist_action)
        self.add_action(cleanup_action)
        self.add_action(resize_cover_action)
        self.add_action(quit_action)
        menu = Gio.Menu()
        file_menu = Gio.Menu()
        file_menu.append("Open File", "app.open")
        file_menu.append("Open Folder", "app.open_folder")
        file_menu.append("Add to Playlist", "app.add_to_playlist")
        file_menu.append("Add Folder to Playlist", "app.add_folder_to_playlist")
        file_menu.append("Cleanup Invalid Tracks", "app.cleanup_invalid")
        file_menu.append("Resize Window to Cover Art", "app.resize_to_cover_art")
        file_menu.append("Quit", "app.quit")
        menu.append_submenu("File", file_menu)
        self.set_menubar(menu)

    def on_open(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Open Audio File",
            parent=self.win,
            action=Gtk.FileChooserAction.OPEN
        )
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        dialog.add_filter(audio_filter)
        dialog.connect("response", self.on_file_chooser_response)
        dialog.set_visible(True)

    def on_add_to_playlist(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Add to Playlist",
            parent=self.win,
            action=Gtk.FileChooserAction.OPEN,
            select_multiple=True
        )
        audio_filter = Gtk.FileFilter()
        audio_filter.set_name("Audio files")
        audio_filter.add_mime_type("audio/*")
        dialog.add_filter(audio_filter)
        dialog.connect("response", self.on_add_to_playlist_response)
        dialog.set_visible(True)

    def on_open_folder(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Open Music Folder",
            parent=self.win,
            action=Gtk.FileChooserAction.SELECT_FOLDER
        )
        dialog.connect("response", self.on_folder_chooser_response)
        dialog.set_visible(True)

    def on_add_folder_to_playlist(self, action, param):
        dialog = Gtk.FileChooserNative.new(
            title="Add Music Folder to Playlist",
            parent=self.win,
            action=Gtk.FileChooserAction.SELECT_FOLDER
        )
        dialog.connect("response", self.on_add_folder_to_playlist_response)
        dialog.set_visible(True)

    def on_file_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            file = dialog.get_property("file")
            if file:
                self.win.playlist.clear()
                if hasattr(self.win, 'playlist_tab'):
                    store = self.win.playlist_tab.playlist_store
                    while store.get_n_items() > 0:
                        store.remove(0)
                self._handle_playlist_addition(file.get_path(), is_folder=False)
                self.win.play_track(0)
        dialog.destroy()

    def on_add_to_playlist_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            files = dialog.get_files()
            if files:
                for file in files:
                    file_path = file.get_path()
                    if file_path:
                        self._handle_playlist_addition(file_path, is_folder=False)
        dialog.destroy()

    def on_folder_chooser_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                self.win.playlist.clear()
                if hasattr(self.win, 'playlist_tab'):
                    store = self.win.playlist_tab.playlist_store
                    while store.get_n_items() > 0:
                        store.remove(0)
        dialog.destroy()

    def on_add_folder_to_playlist_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            folder = dialog.get_file()
            if folder:
                self._handle_playlist_addition(folder.get_path(), is_folder=True)
        dialog.destroy()

    def _handle_playlist_addition(self, file_path, is_folder=False):
        if not file_path or not hasattr(self, 'win') or not self.win:
            return False
        try:
            was_empty = not self.win.playlist
            if is_folder:
                self.win.add_folder_to_playlist(file_path)
            else:
                self.win.add_to_playlist([file_path])
            if was_empty and self.win.playlist:
                self.win.play_track(0)
            return True
        except Exception:
            return False

    def on_cleanup_invalid(self, action, param):
        if hasattr(self, 'win') and self.win:
            self.win.cleanup_invalid_tracks()

    def on_resize_to_cover_art(self, action, param):
        if hasattr(self.win, 'cover_art_widget') and self.win.cover_art_widget:
            self.win.resize_to_cover_art()

    def on_show_visualizer(self, action, param):
        self.win.notebook.set_current_page(3)

def main():
    app = LinAmpApp()
    app.run(sys.argv)

if __name__ == '__main__':
    sys.exit(main())