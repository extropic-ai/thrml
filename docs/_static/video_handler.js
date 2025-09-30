(function () {
  function mountFooterVideo() {
    const footer = document.querySelector('.md-footer');
    if (!footer) return;

    let wrap = footer.querySelector('.footer-video');
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'footer-video';

      const srcWebm = new URL('videos/extropic-footer.webm', document.baseURI).toString();
      const srcMp4  = new URL('videos/extropic-footer.mp4',  document.baseURI).toString();

      wrap.innerHTML = `
        <video autoplay muted playsinline loop preload="metadata" aria-hidden="true" tabindex="-1">
          <source src="${srcWebm}" type="video/webm">
          <source src="${srcMp4}" type="video/mp4">
        </video>`;
      footer.appendChild(wrap);
    }
  }

  if (window.document$) window.document$.subscribe(mountFooterVideo);
  else window.addEventListener('DOMContentLoaded', mountFooterVideo);
})();