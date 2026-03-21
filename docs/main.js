const revealTargets = document.querySelectorAll(
  ".panel, .model-card, .pipe-stage, .agent-stage, .demo-step, .flow-node, .diagram-box, .example-cmd, .demo-video-placeholder, .hardware-placeholder, .ga-hero-card, .ga-detail-card"
);

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
