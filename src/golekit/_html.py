"""Extract the text a reader would see from an HTML document."""

from html.parser import HTMLParser

_HIDDEN_TAGS = {"head", "script", "style", "template", "title"}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.hidden_stack:
            if tag == self.hidden_stack[-1]:
                self.hidden_stack.append(tag)
            return
        if tag in _HIDDEN_TAGS or any(key == "hidden" for key, _ in attrs):
            if tag not in _VOID_TAGS:
                self.hidden_stack.append(tag)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden_stack:
            if tag == self.hidden_stack[-1]:
                self.hidden_stack.pop()
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_stack:
            self.parts.append(data)

    def text(self) -> str:
        lines = "".join(self.parts).splitlines()
        return "\n".join(" ".join(line.split()) for line in lines if line.strip())


def visible_text(markup: str) -> str:
    """Return the visible text of an HTML document, one line per block."""
    parser = _VisibleHTML()
    parser.feed(markup)
    parser.close()
    return parser.text()
