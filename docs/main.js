const revealTargets = document.querySelectorAll(".panel, .stat-card");

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("show");
      }
    });
  },
  { threshold: 0.18 }
);

revealTargets.forEach((el, idx) => {
  el.classList.add("reveal");
  el.style.transitionDelay = `${Math.min(idx * 70, 260)}ms`;
  observer.observe(el);
});
