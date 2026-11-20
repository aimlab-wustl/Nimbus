"""MkDocs hook: the docstrings use Sphinx roles such as :mod:`nimbus.cli`; show only the code part."""
import re

_ROLE = re.compile(r":(?:mod|func|class|meth|attr|data|obj|exc|py:\w+):(?=<code>)")


def on_page_content(html, **kwargs):
    return _ROLE.sub("", html)
