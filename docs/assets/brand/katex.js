// Render the math of pymdownx.arithmatex (generic mode) with KaTeX, also after instant navigation.
(function () {
  function render() {
    if (!window.renderMathInElement) return;
    document.querySelectorAll(".arithmatex").forEach((el) =>
      window.renderMathInElement(el, {
        delimiters: [
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true },
        ],
      })
    );
  }
  // the KaTeX scripts are loaded before this one, so the page can be rendered right away
  render();
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(render);
})();
