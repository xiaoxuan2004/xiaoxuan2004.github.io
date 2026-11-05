/* Shared interactions for Xuan Xiao's portfolio and the PIC project page. */
(() => {
  "use strict";
  const root = document.documentElement;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const themeButton = document.getElementById("theme");
  let savedTheme;
  try { savedTheme = localStorage.getItem("xuan-theme") || localStorage.getItem("theme"); } catch (_) {}
  function applyTheme(theme) {
    root.dataset.theme = theme;
    if (themeButton) {
      themeButton.textContent = theme === "light" ? "☀️" : "🌙";
      themeButton.setAttribute("aria-pressed", String(theme === "light"));
      themeButton.title = theme === "light" ? "Switch to dark theme" : "Switch to light theme";
    }
    document.dispatchEvent(new Event("portfolio-theme-change"));
  }
  applyTheme(savedTheme === "light" ? "light" : "dark");
  themeButton?.addEventListener("click", () => {
    const next = root.dataset.theme === "light" ? "dark" : "light";
    applyTheme(next);
    try { localStorage.setItem("xuan-theme", next); } catch (_) {}
  });

  const nav = document.querySelector("nav");
  const menuButton = document.querySelector(".menu-toggle");
  function closeMenu() {
    nav?.classList.remove("menu-open");
    menuButton?.setAttribute("aria-expanded", "false");
  }
  menuButton?.addEventListener("click", () => {
    const open = nav.classList.toggle("menu-open");
    menuButton.setAttribute("aria-expanded", String(open));
  });
  nav?.querySelectorAll("a").forEach(link => link.addEventListener("click", closeMenu));
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeMenu();
  });

  const typing = document.getElementById("typing");
  if (typing && !reduceMotion.matches) {
    const words = ["AI for Science", "Multi-omics Analysis & Algorithms", "Computer Science × Medical Technology", "An INTP who loves climbing"];
    let word = 0, letter = words[0].length, deleting = false;
    function tick() {
      const text = words[word];
      typing.textContent = text.slice(0, letter);
      if (!deleting && letter < text.length) {
        letter += 1; setTimeout(tick, 90);
      } else if (!deleting) {
        deleting = true; setTimeout(tick, 1600);
      } else if (letter > 0) {
        letter -= 1; setTimeout(tick, 45);
      } else {
        deleting = false; word = (word + 1) % words.length; setTimeout(tick, 300);
      }
    }
    setTimeout(tick, 1400);
  }

  const canvas = document.getElementById("bg");
  if (canvas && !reduceMotion.matches) {
    const ctx = canvas.getContext("2d");
    let width, height, points = [], accent;
    const pointer = { x: -999, y: -999 };
    function updateAccent() { accent = getComputedStyle(root).getPropertyValue("--accent").trim(); }
    function resize() {
      width = innerWidth; height = innerHeight;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      canvas.width = width * dpr; canvas.height = height * dpr;
      canvas.style.width = width + "px"; canvas.style.height = height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      points = Array.from({ length: Math.min(80, Math.floor(width * height / 18000)) }, () => ({ x: Math.random() * width, y: Math.random() * height, vx: (Math.random() - .5) * .4, vy: (Math.random() - .5) * .4 }));
    }
    updateAccent(); resize();
    addEventListener("resize", resize);
    document.addEventListener("portfolio-theme-change", updateAccent);
    addEventListener("pointermove", e => { pointer.x = e.clientX; pointer.y = e.clientY; }, { passive: true });
    function draw() {
      if (!document.hidden) {
        ctx.clearRect(0, 0, width, height);
        points.forEach((p, i) => {
          p.x += p.vx; p.y += p.vy;
          if (p.x < 0 || p.x > width) p.vx *= -1;
          if (p.y < 0 || p.y > height) p.vy *= -1;
          const dx = p.x - pointer.x, dy = p.y - pointer.y, distance = Math.hypot(dx, dy);
          if (distance > 0 && distance < 110) { p.x += dx / distance; p.y += dy / distance; }
          ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, Math.PI * 2); ctx.fillStyle = accent; ctx.globalAlpha = .6; ctx.fill();
          for (let j = i + 1; j < points.length; j++) {
            const q = points[j], d = Math.hypot(p.x - q.x, p.y - q.y);
            if (d < 130) {
              ctx.globalAlpha = (1 - d / 130) * .25; ctx.strokeStyle = accent; ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
            }
          }
        });
        ctx.globalAlpha = 1;
      }
      requestAnimationFrame(draw);
    }
    draw();
  }

  const carousel = document.getElementById("carousel");
  if (carousel) {
    const track = carousel.querySelector(".track"), slides = Array.from(track.children);
    const dots = carousel.querySelector(".dots"), pauseButton = carousel.querySelector(".pause");
    let current = 0, timer, startX = null, manuallyPaused = reduceMotion.matches;
    slides.forEach((_, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.setAttribute("aria-label", "Project image " + (index + 1));
      button.addEventListener("click", () => { go(index); restart(); }); dots.appendChild(button);
    });
    function go(index) {
      current = (index + slides.length) % slides.length;
      track.style.transform = `translateX(-${current * 100}%)`;
      slides.forEach((slide, index) => slide.setAttribute("aria-hidden", String(index !== current)));
      Array.from(dots.children).forEach((dot, index) => {
        dot.classList.toggle("on", index === current);
        dot.setAttribute("aria-current", index === current ? "true" : "false");
      });
    }
    function restart() {
      clearInterval(timer);
      if (!manuallyPaused && !reduceMotion.matches && !carousel.matches(":hover") && !carousel.contains(document.activeElement)) timer = setInterval(() => go(current + 1), 5500);
    }
    function reflectPause() {
      if (!pauseButton) return;
      pauseButton.textContent = manuallyPaused ? "Play" : "Pause";
      pauseButton.setAttribute("aria-label", manuallyPaused ? "Play slideshow" : "Pause slideshow");
    }
    carousel.querySelector(".prev").addEventListener("click", () => { go(current - 1); restart(); });
    carousel.querySelector(".next").addEventListener("click", () => { go(current + 1); restart(); });
    pauseButton?.addEventListener("click", () => { manuallyPaused = !manuallyPaused; reflectPause(); restart(); });
    carousel.addEventListener("mouseenter", () => clearInterval(timer));
    carousel.addEventListener("mouseleave", restart);
    carousel.addEventListener("focusin", () => clearInterval(timer));
    carousel.addEventListener("focusout", () => setTimeout(restart, 0));
    carousel.addEventListener("touchstart", e => { startX = e.touches[0].clientX; }, { passive: true });
    carousel.addEventListener("touchend", e => {
      if (startX === null) return;
      const delta = e.changedTouches[0].clientX - startX;
      if (Math.abs(delta) > 40) { go(current + (delta < 0 ? 1 : -1)); restart(); }
      startX = null;
    }, { passive: true });
    go(0); reflectPause(); restart();
  }

  const revealElements = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reduceMotion.matches) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { entry.target.classList.add("visible"); observer.unobserve(entry.target); }
    }), { threshold: .08 });
    revealElements.forEach(element => observer.observe(element));
  } else revealElements.forEach(element => element.classList.add("visible"));

  const sections = Array.from(document.querySelectorAll("section[id]"));
  const navigationLinks = Array.from(document.querySelectorAll('nav a[href^="#"]'));
  const progress = document.getElementById("progress"), topButton = document.getElementById("top");
  let scrollPending = false;
  function updateScroll() {
    const max = root.scrollHeight - root.clientHeight;
    if (progress) progress.style.width = (max > 0 ? root.scrollTop / max * 100 : 0) + "%";
    topButton?.classList.toggle("show", root.scrollTop > 500);
    let current = "";
    sections.forEach(section => { if (root.scrollTop + 150 >= section.offsetTop) current = section.id; });
    navigationLinks.forEach(link => link.classList.toggle("active", link.getAttribute("href") === "#" + current));
    scrollPending = false;
  }
  addEventListener("scroll", () => { if (!scrollPending) { scrollPending = true; requestAnimationFrame(updateScroll); } }, { passive: true });
  topButton?.addEventListener("click", () => scrollTo({ top: 0, behavior: reduceMotion.matches ? "auto" : "smooth" }));
  const year = document.getElementById("year");
  if (year) year.textContent = new Date().getFullYear();
  updateScroll();
})();
