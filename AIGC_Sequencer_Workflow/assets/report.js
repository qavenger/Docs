(() => {
  const progress = document.querySelector('.progress > span');
  const backtop = document.querySelector('.backtop');
  const links = [...document.querySelectorAll('.toc a')];
  const sections = links
    .map(link => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  function updateScroll() {
    const doc = document.documentElement;
    const total = doc.scrollHeight - doc.clientHeight;
    const ratio = total > 0 ? doc.scrollTop / total : 0;
    progress.style.width = `${Math.min(100, ratio * 100)}%`;
    backtop.classList.toggle('show', doc.scrollTop > 700);
  }

  const observer = new IntersectionObserver(entries => {
    const visible = entries
      .filter(entry => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === `#${visible.target.id}`);
    });
  }, { rootMargin: '-15% 0px -68% 0px', threshold: [0, 0.2, 0.6] });

  sections.forEach(section => observer.observe(section));
  window.addEventListener('scroll', updateScroll, { passive: true });
  backtop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  document.querySelector('[data-print]')?.addEventListener('click', () => window.print());
  updateScroll();
})();
