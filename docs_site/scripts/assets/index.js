(function () {
  document.querySelectorAll('.codecard .copy').forEach(function (btn) {
    var original = btn.innerHTML;
    btn.addEventListener('click', function () {
      var card = btn.closest('.codecard');
      var pre = card ? card.querySelector('pre') : null;
      var text = pre ? pre.innerText : '';
      var done = function () {
        btn.classList.add('done'); btn.textContent = 'copied';
        setTimeout(function () { btn.classList.remove('done'); btn.innerHTML = original; }, 1200);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(text).then(done, done); }
      else { done(); }
    });
  });

  // subtle scroll-in for sections
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
    document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });
  } else {
    document.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('in'); });
  }
})();
