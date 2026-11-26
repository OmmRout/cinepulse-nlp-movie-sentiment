/**
 * Ambient Cinema Projector Light Beams & Warm Golden Bokeh Embers
 * Zero external dependencies, hardware-friendly, honors prefers-reduced-motion.
 */
(function () {
  const canvas = document.getElementById("cinema-bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let width = 0;
  let height = 0;
  let dpr = 1;
  let mouseX = 0.5;
  let mouseY = 0.3;

  const EMBER_COLORS = [
    "rgba(246, 211, 138, ",
    "rgba(229, 178, 93, ",
    "rgba(223, 94, 82, ",
    "rgba(247, 239, 230, ",
  ];

  const particles = [];
  const PARTICLE_COUNT = 68;

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function createParticle(randomY) {
    const depth = 0.25 + Math.random() * 0.95;
    return {
      x: Math.random() * width,
      y: randomY ? Math.random() * height : height + 20,
      radius: (1.2 + Math.random() * 3.4) * depth,
      vy: -(0.12 + Math.random() * 0.38) * depth,
      vx: (Math.random() - 0.5) * 0.16,
      phase: Math.random() * Math.PI * 2,
      sway: 0.2 + Math.random() * 0.45,
      alpha: 0.12 + Math.random() * 0.42,
      colorPrefix: EMBER_COLORS[Math.floor(Math.random() * EMBER_COLORS.length)],
      depth,
    };
  }

  function initParticles() {
    particles.length = 0;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push(createParticle(true));
    }
  }

  function drawProjectorBeams(t) {
    // Base warm obsidian radial glow
    const baseGrad = ctx.createRadialGradient(
      width * (0.48 + (mouseX - 0.5) * 0.08),
      height * 0.18,
      20,
      width * 0.5,
      height * 0.45,
      Math.max(width, height) * 0.85
    );
    baseGrad.addColorStop(0, "rgba(74, 42, 24, 0.24)");
    baseGrad.addColorStop(0.45, "rgba(42, 18, 22, 0.16)");
    baseGrad.addColorStop(1, "rgba(11, 8, 10, 0.0)");
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, width, height);

    // Sweeping Projector Beam 1 (Warm Champagne Amber)
    const sweep1 = Math.sin(t * 0.00035) * 0.16;
    ctx.save();
    ctx.translate(width * 0.12, -40);
    ctx.rotate(0.42 + sweep1);
    const beam1 = ctx.createLinearGradient(0, 0, 0, height * 1.3);
    beam1.addColorStop(0, "rgba(246, 211, 138, 0.095)");
    beam1.addColorStop(0.55, "rgba(229, 178, 93, 0.032)");
    beam1.addColorStop(1, "rgba(229, 178, 93, 0)");
    ctx.fillStyle = beam1;
    ctx.beginPath();
    ctx.moveTo(-60, 0);
    ctx.lineTo(60, 0);
    ctx.lineTo(width * 0.42, height * 1.35);
    ctx.lineTo(-width * 0.18, height * 1.35);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // Sweeping Projector Beam 2 (Soft Crimson-Terracotta Accent)
    const sweep2 = Math.cos(t * 0.00028) * 0.14;
    ctx.save();
    ctx.translate(width * 0.88, -50);
    ctx.rotate(-0.48 + sweep2);
    const beam2 = ctx.createLinearGradient(0, 0, 0, height * 1.25);
    beam2.addColorStop(0, "rgba(223, 94, 82, 0.07)");
    beam2.addColorStop(0.5, "rgba(168, 118, 184, 0.024)");
    beam2.addColorStop(1, "rgba(223, 94, 82, 0)");
    ctx.fillStyle = beam2;
    ctx.beginPath();
    ctx.moveTo(-50, 0);
    ctx.lineTo(50, 0);
    ctx.lineTo(width * 0.25, height * 1.28);
    ctx.lineTo(-width * 0.36, height * 1.28);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function renderFrame(timestamp) {
    ctx.clearRect(0, 0, width, height);
    drawProjectorBeams(timestamp);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      if (!prefersReduced) {
        p.y += p.vy;
        p.x += p.vx + Math.sin(timestamp * 0.0008 + p.phase) * p.sway * 0.22;
      }

      if (p.y < -25 || p.x < -30 || p.x > width + 30) {
        Object.assign(p, createParticle(false));
      }

      const px = p.x + (mouseX - 0.5) * 18 * p.depth;
      const py = p.y + (mouseY - 0.5) * 12 * p.depth;
      const pulse = 0.78 + 0.22 * Math.sin(timestamp * 0.0018 + p.phase);
      const glowRadius = p.radius * 3.4;

      const grad = ctx.createRadialGradient(px, py, 0, px, py, glowRadius);
      grad.addColorStop(0, p.colorPrefix + (p.alpha * pulse).toFixed(3) + ")");
      grad.addColorStop(0.45, p.colorPrefix + (p.alpha * 0.32 * pulse).toFixed(3) + ")");
      grad.addColorStop(1, p.colorPrefix + "0)");

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, glowRadius, 0, Math.PI * 2);
      ctx.fill();
    }

    if (!prefersReduced) {
      window.requestAnimationFrame(renderFrame);
    }
  }

  window.addEventListener("resize", () => {
    resize();
    initParticles();
  });

  window.addEventListener("mousemove", (e) => {
    mouseX = e.clientX / Math.max(width, 1);
    mouseY = e.clientY / Math.max(height, 1);
  });

  resize();
  initParticles();
  window.requestAnimationFrame(renderFrame);
})();
