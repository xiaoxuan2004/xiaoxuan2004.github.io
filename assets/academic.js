/* Academic homepage interactions: navigation, accessible photo carousel, current year. */
(() => {
  'use strict';

  // 1. Mobile navigation and active section indication.
  const nav = document.querySelector('.navigation');
  const toggle = document.querySelector('.menu-toggle');
  const closeMenu = () => {
    nav.classList.remove('menu-open');
    toggle.setAttribute('aria-expanded', 'false');
  };
  toggle.addEventListener('click', () => {
    const open = nav.classList.toggle('menu-open');
    toggle.setAttribute('aria-expanded', String(open));
  });
  nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeMenu();
  });
  const links = [...nav.querySelectorAll('ul a')];
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      links.forEach(link => {
        if (link.hash === '#' + visible[0].target.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }, {rootMargin: '-12% 0px -55% 0px', threshold: 0});
    document.querySelectorAll('main > section').forEach(section => observer.observe(section));
  }
  document.getElementById('year').textContent = new Date().getFullYear();

  // 2. Photo carousel: buttons, dots, keyboard navigation and touch swipes.
  const gallery = document.querySelector('.life-gallery');
  const track = gallery.querySelector('.gallery-track');
  const slides = [...gallery.querySelectorAll('.gallery-slide')];
  const dots = gallery.querySelector('.gallery-dots');
  const pause = gallery.querySelector('.gallery-pause');
  const counter = document.getElementById('gallery-counter');
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let current = 0;
  let paused = reduceMotion.matches;
  let startedByUser = false;
  let hovered = false;
  let galleryVisible = !('IntersectionObserver' in window);
  let timer;
  slides.forEach((slide, index) => {
    slide.setAttribute('role', 'group');
    slide.setAttribute('aria-roledescription', 'slide');
    slide.setAttribute('aria-label', `${index + 1} of ${slides.length}`);
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.setAttribute('aria-label', `Show photo ${index + 1}`);
    dot.addEventListener('click', () => show(index));
    dots.append(dot);
  });
  const schedule = () => {
    clearTimeout(timer);
    const interacting = hovered || gallery.contains(document.activeElement);
    if (galleryVisible && !paused && !document.hidden && (startedByUser || !interacting) && slides.length > 1) {
      timer = setTimeout(() => show(current + 1), 4000);
    }
  };
  function show(index) {
    current = (index + slides.length) % slides.length;
    track.style.transform = `translateX(-${current * 100}%)`;
    slides.forEach((slide, i) => {
      slide.setAttribute('aria-hidden', String(i !== current));
      slide.inert = i !== current;
    });
    [...dots.children].forEach((dot, i) => {
      if (i === current) dot.setAttribute('aria-current', 'true');
      else dot.removeAttribute('aria-current');
    });
    counter.textContent = `${String(current + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
    schedule();
  }
  function updatePause() {
    pause.textContent = paused ? 'Play' : 'Pause';
    pause.setAttribute('aria-label', paused ? 'Play slideshow' : 'Pause slideshow');
    schedule();
  }
  gallery.querySelector('.gallery-prev').addEventListener('click', () => show(current - 1));
  gallery.querySelector('.gallery-next').addEventListener('click', () => show(current + 1));
  pause.addEventListener('click', () => {
    paused = !paused;
    startedByUser = !paused;
    if (startedByUser && slides.length > 1) show(current + 1);
    updatePause();
  });
  gallery.addEventListener('mouseenter', () => { hovered = true; schedule(); });
  gallery.addEventListener('mouseleave', () => { hovered = false; schedule(); });
  gallery.addEventListener('focusin', schedule);
  gallery.addEventListener('focusout', () => setTimeout(schedule, 0));
  gallery.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      show(current + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  let startX = 0;
  let startY = 0;
  gallery.addEventListener('touchstart', event => {
    startX = event.changedTouches[0].clientX;
    startY = event.changedTouches[0].clientY;
  }, {passive: true});
  gallery.addEventListener('touchend', event => {
    const dx = event.changedTouches[0].clientX - startX;
    const dy = event.changedTouches[0].clientY - startY;
    if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy)) show(current + (dx < 0 ? 1 : -1));
  }, {passive: true});
  document.addEventListener('visibilitychange', schedule);
  if ('IntersectionObserver' in window) {
    const galleryObserver = new IntersectionObserver(([entry]) => {
      galleryVisible = entry.isIntersecting;
      schedule();
    });
    galleryObserver.observe(gallery);
  }
  reduceMotion.addEventListener('change', () => {
    if (reduceMotion.matches) { paused = true; updatePause(); }
  });
  show(0);
  updatePause();
})();
