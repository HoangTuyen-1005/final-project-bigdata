"""Build resized requests for Airbnb image CDN URLs."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


AIRBNB_IMAGE_HOSTS = frozenset({"a0.muscache.com", "z1.muscache.cn"})


def resized_airbnb_image_url(url: str, max_width: int) -> str | None:
    """Cap the server-side width for recognized CDN paths."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme != "https" or host not in AIRBNB_IMAGE_HOSTS:
            return None
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            return None

        path = parsed.path
        if host == "a0.muscache.com" and path.startswith("/pictures/"):
            path = "/im" + path
        if not path.startswith("/im/pictures/"):
            return None

        query = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key.lower() == "im_w":
                if value.isascii() and value.isdecimal() and len(value) <= 5:
                    requested_width = int(value)
                    if 0 < requested_width < max_width:
                        max_width = requested_width
            elif key.lower() != "aki_policy":
                query.append((key, value))
        query.append(("im_w", str(max_width)))
        return urlunsplit((parsed.scheme, parsed.netloc, path, urlencode(query), ""))
    except ValueError:
        return None
