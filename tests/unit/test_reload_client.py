import io
import urllib.error
import urllib.request

import pytest

from mlops.serving import reload_client


def test_request_reload_posts_to_endpoint_with_matching_auth_header(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return io.BytesIO(b'{"status":"reloaded","model_version":"7"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = reload_client.request_reload("http://api:8000/", "secret", timeout=4.0)

    request, timeout = calls[0]
    assert request.full_url == "http://api:8000/admin/reload"
    assert request.get_method() == "POST"
    assert request.get_header("X-admin-token") == "secret"
    assert timeout == 4.0
    assert result == {"status": "reloaded", "model_version": "7"}


def test_request_reload_omits_auth_header_when_token_is_not_configured(monkeypatch):
    requests = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: requests.append(request) or io.BytesIO(b'{"status":"reloaded"}'),
    )

    reload_client.request_reload("http://api:8000")

    assert requests[0].get_header("X-admin-token") is None


def test_main_uses_compose_environment_values_without_printing_token(monkeypatch, capsys):
    token = "do-not-print-this-token"
    monkeypatch.setenv("API_URL", "http://api:8000")
    monkeypatch.setenv("ADMIN_TOKEN", token)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda _request, timeout: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    with pytest.raises(SystemExit) as raised:
        reload_client.main()

    assert raised.value.code == 1
    assert token not in capsys.readouterr().out
