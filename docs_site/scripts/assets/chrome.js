document.addEventListener("DOMContentLoaded", function () {
  var COPY = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  var CHECK = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
  var CHEV = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 6 15 12 9 18"/></svg>';

  // External links open in a new tab; internal site/anchor links stay in place.
  document.querySelectorAll("a[href]").forEach(function (a) {
    var h = a.getAttribute("href") || "";
    if (/^https?:\/\//i.test(h)) { a.target = "_blank"; a.rel = "noopener"; }
  });

  function wireCopy(btn, getText) {
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var text = getText();
      var done = function () {
        btn.classList.add("copied"); btn.innerHTML = CHECK;
        setTimeout(function () { btn.classList.remove("copied"); btn.innerHTML = COPY; }, 1300);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else { done(); }
    });
  }

  // Notebook code cells: header bar (language chip + copy).
  document.querySelectorAll(".jp-CodeCell .jp-InputArea-editor").forEach(function (editor) {
    if (editor.querySelector(".thrml-code-head")) return;
    var head = document.createElement("div");
    head.className = "thrml-code-head";
    head.innerHTML = '<span class="thrml-lang">python</span>' +
      '<span class="thrml-tools"><button class="thrml-copy" type="button" title="Copy code" aria-label="Copy code">' + COPY + '</button></span>';
    editor.insertBefore(head, editor.firstChild);
    wireCopy(head.querySelector(".thrml-copy"), function () {
      var pre = editor.querySelector(".highlight pre"); return pre ? pre.innerText : "";
    });
  });

  // Docs-page code cards.
  document.querySelectorAll(".thrml-codecard").forEach(function (card) {
    var btn = card.querySelector(".thrml-copy"); if (!btn) return;
    if (!btn.innerHTML.trim()) btn.innerHTML = COPY;
    wireCopy(btn, function () { var pre = card.querySelector("pre"); return pre ? pre.innerText : ""; });
  });

  // Collapsed setup cells fold behind a pill toggle.
  document.querySelectorAll(".jp-CodeCell.celltag_hide-input").forEach(function (cell) {
    if (cell.querySelector(".thrml-toggle")) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thrml-toggle";
    btn.title = "Toggle setup code";
    btn.innerHTML = '<span class="thrml-chev">' + CHEV + '</span><span>setup code</span>';
    btn.addEventListener("click", function () { cell.classList.toggle("thrml-open"); });
    cell.insertBefore(btn, cell.firstChild);
  });

  // Mobile sidebar toggle.
  var burger = document.querySelector(".thrml-burger");
  var sb = document.querySelector(".thrml-sidebar");
  if (burger && sb) {
    var bd = document.createElement("div");
    bd.className = "thrml-backdrop";
    document.body.appendChild(bd);
    var HAM = burger.innerHTML;
    var XICON = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>';
    function txSetNav(open) {
      sb.classList.toggle("thrml-open", open); bd.classList.toggle("thrml-open", open);
      burger.innerHTML = open ? XICON : HAM;
      burger.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    }
    burger.addEventListener("click", function () { txSetNav(!sb.classList.contains("thrml-open")); });
    bd.addEventListener("click", function () { txSetNav(false); });
  }
});
