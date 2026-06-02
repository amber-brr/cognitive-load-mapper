import sys, types

_saved_modules = {}


class DummyColumn:
    def metric(self, *args, **kwargs):
        pass
    def text_area(self, *args, **kwargs):
        pass


def stub_columns(n):
    return tuple(DummyColumn() for _ in range(n))


def stub_selectbox(*args, **kwargs):
    # Return the first option if available
    options = args[1] if len(args) > 1 else kwargs.get("options", [])
    return options[0] if options else None


class StubContextManager:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def __getattr__(self, name):
        return lambda *a, **kw: None


class StubResponse:
    def __init__(self, url=""):
        self.url = url
    def json(self):
        # Return different responses based on URL
        if "/articles/" in self.url and not self.url.endswith("/articles"):
            # Single article endpoint
            return {
                "article_id": 1,
                "title": "Test Article",
                "publication_name": "Test Pub",
                "gradient_shape": "ramp",
                "mean_complexity": 0.5,
                "post_url": "http://example.com",
                "paragraphs": []
            }
        else:
            # List articles endpoint
            return [{"article_id": 1, "title": "Test", "publication_name": "Test Pub"}]
    def raise_for_status(self):
        pass


def stub_get(url, *args, **kwargs):
    return StubResponse(url)


def stub_post(*args, **kwargs):
    return StubResponse()


def setup_module(module):
    """Set up test stubs scoped to this test module only."""
    # Stub streamlit
    st_stub = types.ModuleType("streamlit")

    # Default stubs that return None
    for attr in [
        "set_page_config", "title", "header", "subheader", "caption", "divider",
        "radio", "metric", "line_chart", "expander", "write", "markdown",
        "button", "spinner", "success", "info", "error", "stop", "text_area",
    ]:
        setattr(st_stub, attr, lambda *a, **kw: None)

    st_stub.columns = stub_columns
    st_stub.selectbox = stub_selectbox
    st_stub.sidebar = StubContextManager()

    # Stub requests module
    requests_stub = types.ModuleType("requests")
    requests_exceptions = types.ModuleType("exceptions")
    requests_exceptions.ConnectionError = Exception
    requests_exceptions.HTTPError = Exception
    requests_stub.exceptions = requests_exceptions
    requests_stub.get = stub_get
    requests_stub.post = stub_post

    # Stub pandas
    pandas_stub = types.ModuleType("pandas")

    stubs = {
        "streamlit": st_stub,
        "requests": requests_stub,
        "pandas": pandas_stub,
    }

    # Save current state and register stubs
    for k, v in stubs.items():
        _saved_modules[k] = sys.modules.get(k)
        sys.modules[k] = v

    # Force fresh import of app with stubs active
    sys.modules.pop("app", None)


def teardown_module(module):
    """Restore original sys.modules state after test module completes."""
    for k, original in _saved_modules.items():
        if original is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = original
    sys.modules.pop("app", None)
    _saved_modules.clear()

def test_publications_excludes_letters_from_an_american():
    from app import PUBLICATIONS
    assert "heathercoxrichardson.substack.com" not in PUBLICATIONS


def test_publications_has_five_entries():
    from app import PUBLICATIONS
    assert len(PUBLICATIONS) == 5


def test_publications_values_are_human_readable():
    from app import PUBLICATIONS
    for key, name in PUBLICATIONS.items():
        assert "." not in name, f"Expected readable name, got domain-like value: {name}"
        assert name.strip() == name
