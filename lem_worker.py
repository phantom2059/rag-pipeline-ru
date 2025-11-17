import pymorphy2
from functools import lru_cache
from typing import Optional

_morph: Optional[pymorphy2.MorphAnalyzer] = None

def _apply_inspect_shim():
    """Applies a shim for inspect.getargspec for Python 3.11+ compatibility."""
    import inspect
    from collections import namedtuple
    if not hasattr(inspect, 'getargspec'):
        ArgSpec = namedtuple('ArgSpec', 'args varargs keywords defaults')
        def _getargspec(func):
            fs = inspect.getfullargspec(func)
            return ArgSpec(fs.args, fs.varargs, fs.varkw, fs.defaults)
        inspect.getargspec = _getargspec

def _get_morph() -> pymorphy2.MorphAnalyzer:
    global _morph
    if _morph is None:
        _apply_inspect_shim()  # Shim is applied here, just before pymorphy2 is used.
        _morph = pymorphy2.MorphAnalyzer()
    return _morph

@lru_cache(maxsize=500000)
def lemma_token(w: str) -> str:
    p = _get_morph().parse(w)
    return p[0].normal_form if p else w

def lemmatize_text(s: str) -> str:
    return " ".join(lemma_token(w) for w in s.split())
