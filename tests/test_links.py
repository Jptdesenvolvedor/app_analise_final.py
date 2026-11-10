"""Verifica se as URLs fixas de fontes.json ainda respondem.

Precisa de internet, então só roda com: pytest -m network
"""

import urllib.request

import pytest

from analise.sources import todas_as_urls_estaticas

pytestmark = pytest.mark.network


@pytest.mark.parametrize("url", sorted(set(todas_as_urls_estaticas())))
def test_url_responde(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (link-check)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            assert r.status < 400
    except urllib.error.HTTPError as e:
        # 403/429 = site bloqueia robôs, mas a URL existe.
        assert e.code in (403, 429), f"{url} -> HTTP {e.code}"
