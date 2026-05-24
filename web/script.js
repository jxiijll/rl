const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", (event) => {
    const targetId = link.getAttribute("href");
    if (!targetId || targetId === "#") return;
    const target = document.querySelector(targetId);
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block: "start" });
    history.pushState(null, "", targetId);
  });
});

// Copy BibTeX
const copyButton = document.querySelector("#copy-bibtex");
const bibtexCode = document.querySelector("#bibtex-code");
if (copyButton && bibtexCode) {
  copyButton.addEventListener("click", async () => {
    const text = bibtexCode.textContent.trim();
    try {
      await navigator.clipboard.writeText(text);
      copyButton.textContent = "Copied";
      setTimeout(() => { copyButton.textContent = "Copy"; }, 1600);
    } catch {
      copyButton.textContent = "Select text";
    }
  });
}

// Scroll reveal
const revealItems = document.querySelectorAll(".reveal");
if (prefersReducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  revealItems.forEach((item) => revealObserver.observe(item));
}

// Active nav link highlighting
const navLinks = document.querySelectorAll(".site-nav-links a[href^='#']");
const sections = Array.from(document.querySelectorAll("section[id], header[id]"));
if (navLinks.length && sections.length && "IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute("id");
          navLinks.forEach((link) => {
            link.classList.toggle("active", link.getAttribute("href") === `#${id}`);
          });
        }
      });
    },
    { rootMargin: "-30% 0px -65% 0px" }
  );
  sections.forEach((s) => sectionObserver.observe(s));
}

// Back to top button
const backToTop = document.querySelector(".back-to-top");
if (backToTop) {
  window.addEventListener("scroll", () => {
    backToTop.classList.toggle("visible", window.scrollY > 500);
  }, { passive: true });
}

// Lag intuition: play the bar-grow animation when scrolled into view
const lagFigure = document.querySelector(".lag-intuition");
if (lagFigure) {
  const lagRows = lagFigure.querySelectorAll(".lag-row");
  const play = () => {
    lagRows.forEach((row, i) => {
      setTimeout(() => row.classList.add("in-view"), 90 * i);
    });
  };
  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    lagRows.forEach((row) => row.classList.add("in-view"));
  } else {
    const lagObs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            play();
            lagObs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.25 }
    );
    lagObs.observe(lagFigure);
  }
}

// Count-up animation for metric cards
const countTargets = document.querySelectorAll("[data-count-target]");
if (countTargets.length) {
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const formatValue = (value, decimals, suffix) => {
    const str = decimals > 0 ? value.toFixed(decimals) : Math.round(value).toString();
    return suffix ? `${str}${suffix}` : str;
  };
  const animate = (el) => {
    const target = parseFloat(el.dataset.countTarget);
    if (Number.isNaN(target)) return;
    const decimals = parseInt(el.dataset.countDecimals || "0", 10);
    const suffix = el.dataset.countSuffix || "";
    const duration = 1100;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      el.textContent = formatValue(target * ease(t), decimals, suffix);
      if (t < 1) requestAnimationFrame(step);
      else el.textContent = formatValue(target, decimals, suffix);
    };
    requestAnimationFrame(step);
  };
  if (prefersReducedMotion || !("IntersectionObserver" in window)) {
    countTargets.forEach((el) => {
      const decimals = parseInt(el.dataset.countDecimals || "0", 10);
      const suffix = el.dataset.countSuffix || "";
      el.textContent = formatValue(parseFloat(el.dataset.countTarget), decimals, suffix);
    });
  } else {
    const countObs = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            animate(entry.target);
            countObs.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.4 }
    );
    countTargets.forEach((el) => {
      el.textContent = "0";
      countObs.observe(el);
    });
  }
}

// Training curve chart
const TRAINING_DATA = [
  {u:1,auc:13.999},{u:2,auc:14.902},{u:3,auc:16.312},{u:4,auc:15.841},
  {u:5,auc:30.221},{u:6,auc:15.551},{u:7,auc:16.047},{u:8,auc:14.695},
  {u:9,auc:14.411},{u:10,auc:14.471},{u:11,auc:14.474},{u:12,auc:14.601},
  {u:13,auc:15.052},{u:14,auc:14.710},{u:15,auc:14.964},{u:16,auc:14.124},
  {u:17,auc:14.384},{u:18,auc:14.385},{u:19,auc:14.811},{u:20,auc:14.924},
  {u:21,auc:14.802},{u:22,auc:14.887},{u:23,auc:15.131},{u:24,auc:15.466},
  {u:25,auc:14.323},{u:26,auc:14.375},{u:27,auc:14.219},{u:28,auc:14.086},
  {u:29,auc:14.634},{u:30,auc:14.066},{u:31,auc:14.141},{u:32,auc:14.225},
  {u:33,auc:13.742,best:true},
  {u:34,auc:14.089},{u:35,auc:14.216},{u:36,auc:14.340},{u:37,auc:15.607},
  {u:38,auc:15.241},{u:39,auc:15.337},
  {u:40,auc:330.781},{u:41,auc:460.716},{u:42,auc:466.552},{u:43,auc:549.110},
  {u:44,auc:539.647},{u:45,auc:611.568},{u:46,auc:609.812},{u:47,auc:679.937},
  {u:48,auc:679.937},{u:49,auc:693.821},{u:50,auc:599.888},{u:51,auc:582.575},
  {u:52,auc:644.636},{u:53,auc:641.688},{u:54,auc:635.933},{u:55,auc:807.139},
  {u:56,auc:990.169},{u:57,auc:871.089},{u:58,auc:846.363},{u:59,auc:906.305},
  {u:60,auc:779.795},{u:61,auc:944.389},{u:62,auc:972.259},{u:63,auc:877.519},
  {u:64,auc:964.020},{u:65,auc:780.167},{u:66,auc:880.899},{u:67,auc:1110.994},
  {u:68,auc:1162.900},{u:69,auc:929.821},{u:70,auc:750.742},{u:71,auc:812.693},
  {u:72,auc:750.986},{u:73,auc:649.673},{u:74,auc:830.947},{u:75,auc:714.993},
  {u:76,auc:783.926},{u:77,auc:785.188},{u:78,auc:782.858},{u:79,auc:784.699},
  {u:80,auc:632.932}
];

function drawTrainingChart() {
  const canvas = document.getElementById("training-chart");
  if (!canvas || !canvas.getContext) return;

  const Y_MAX = 32;   // values above this are clipped to the top edge
  const Y_MIN = 12.5;
  const X_MIN = 1;
  const X_MAX = 80;
  const DRIFT_START = 39.5;

  const SJF_BASE = 14.274;
  const SDP_BASE = 14.073;

  // Colors (matching CSS variables)
  const C = {
    teal: "#138a8a",
    blue: "#2d5be3",
    ink: "#172033",
    softInk: "#526076",
    mutedInk: "#718093",
    line: "#dce5ea",
    panel: "#ffffff",
    coral: "#cf5d48",
    gold: "#a17217",
    gray: "#8da0b8",
  };

  let rafId;

  function render() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return;

    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const W = cssW, H = cssH;
    const PAD = { top: 28, right: 20, bottom: 42, left: 52 };
    const cW = W - PAD.left - PAD.right;
    const cH = H - PAD.top - PAD.bottom;

    function xp(u) { return PAD.left + (u - X_MIN) / (X_MAX - X_MIN) * cW; }
    function yp(v) { return PAD.top + (1 - (Math.min(v, Y_MAX) - Y_MIN) / (Y_MAX - Y_MIN)) * cH; }

    // Background
    ctx.fillStyle = C.panel;
    ctx.fillRect(0, 0, W, H);

    // Drift zone shading
    const driftX = xp(DRIFT_START);
    const grad1 = ctx.createLinearGradient(driftX, 0, driftX + cW * 0.18, 0);
    grad1.addColorStop(0, "rgba(207,93,72,0.0)");
    grad1.addColorStop(1, "rgba(207,93,72,0.08)");
    ctx.fillStyle = grad1;
    ctx.fillRect(driftX, PAD.top, W - driftX - PAD.right, cH);
    ctx.fillStyle = "rgba(207,93,72,0.06)";
    ctx.fillRect(driftX + cW * 0.18, PAD.top, W - driftX - cW * 0.18 - PAD.right, cH);

    // Horizontal grid lines
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 1;
    [13, 14, 15, 16, 18, 20, 24, 28, 32].forEach((y) => {
      if (y < Y_MIN || y > Y_MAX) return;
      const py = yp(y);
      ctx.beginPath();
      ctx.moveTo(PAD.left, py);
      ctx.lineTo(PAD.left + cW, py);
      ctx.stroke();
    });

    // Vertical grid lines
    [10, 20, 30, 40, 50, 60, 70, 80].forEach((x) => {
      const px = xp(x);
      ctx.strokeStyle = x === 40 ? "rgba(207,93,72,0.25)" : C.line;
      ctx.lineWidth = x === 40 ? 1.5 : 1;
      ctx.setLineDash(x === 40 ? [4, 3] : []);
      ctx.beginPath();
      ctx.moveTo(px, PAD.top);
      ctx.lineTo(px, PAD.top + cH);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // SJF reference line
    ctx.strokeStyle = C.gray;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(PAD.left, yp(SJF_BASE));
    ctx.lineTo(PAD.left + cW, yp(SJF_BASE));
    ctx.stroke();

    // Slowdown Priority reference line
    ctx.strokeStyle = C.gold;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(PAD.left, yp(SDP_BASE));
    ctx.lineTo(PAD.left + cW, yp(SDP_BASE));
    ctx.stroke();
    ctx.setLineDash([]);

    // Fill under line (only the "stable" region for cleaner look)
    const fillGrad = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + cH);
    fillGrad.addColorStop(0, "rgba(19,138,138,0.10)");
    fillGrad.addColorStop(1, "rgba(19,138,138,0.01)");
    ctx.beginPath();
    TRAINING_DATA.forEach((d, i) => {
      const px = xp(d.u), py = yp(d.auc);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.lineTo(xp(TRAINING_DATA[TRAINING_DATA.length - 1].u), PAD.top + cH);
    ctx.lineTo(xp(TRAINING_DATA[0].u), PAD.top + cH);
    ctx.closePath();
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Data line — clipped to chart area
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD.left, PAD.top, cW, cH);
    ctx.clip();

    const lineGrad = ctx.createLinearGradient(xp(1), 0, xp(80), 0);
    lineGrad.addColorStop(0, C.teal);
    lineGrad.addColorStop(0.5, C.blue);
    lineGrad.addColorStop(1, C.coral);
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.beginPath();
    TRAINING_DATA.forEach((d, i) => {
      const px = xp(d.u), py = yp(d.auc);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.restore();

    // Best checkpoint marker
    const best = TRAINING_DATA.find((d) => d.best);
    if (best) {
      const bx = xp(best.u), by = yp(best.auc);
      // Glow
      ctx.beginPath();
      ctx.arc(bx, by, 9, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(19,138,138,0.18)";
      ctx.fill();
      // Circle
      ctx.beginPath();
      ctx.arc(bx, by, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = C.teal;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
      // Label
      ctx.fillStyle = C.ink;
      ctx.font = `bold 10px Inter, ui-sans-serif, sans-serif`;
      ctx.textAlign = "center";
      const labelY = by - 14;
      ctx.fillText("Best ✓ update 33", bx, labelY);
      ctx.fillStyle = C.teal;
      ctx.font = `bold 11px Inter, ui-sans-serif, sans-serif`;
      ctx.fillText("13.742", bx, labelY - 12);
    }

    // Drift annotation
    const annotX = xp(59);
    ctx.fillStyle = C.coral;
    ctx.font = `bold 9px Inter, ui-sans-serif, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("Drift zone — values rise to 1000+", annotX, PAD.top + 14);

    // Axes
    ctx.strokeStyle = "#c8d6e5";
    ctx.lineWidth = 1;
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top);
    ctx.lineTo(PAD.left, PAD.top + cH);
    ctx.lineTo(PAD.left + cW, PAD.top + cH);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = C.mutedInk;
    ctx.font = `10px Inter, ui-sans-serif, sans-serif`;
    ctx.textAlign = "right";
    [13, 14, 15, 16, 18, 20, 24, 28, 32].forEach((y) => {
      if (y < Y_MIN || y > Y_MAX) return;
      ctx.fillText(y, PAD.left - 6, yp(y) + 3.5);
    });

    // X-axis labels
    ctx.textAlign = "center";
    [1, 10, 20, 30, 40, 50, 60, 70, 80].forEach((x) => {
      ctx.fillText(x, xp(x), PAD.top + cH + 14);
    });

    // Axis titles
    ctx.fillStyle = C.softInk;
    ctx.font = `bold 9px Inter, ui-sans-serif, sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText("Training Update", PAD.left + cW / 2, H - 6);

    ctx.save();
    ctx.translate(11, PAD.top + cH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("CFS Lag AUC ↓", 0, 0);
    ctx.restore();

  }

  render();

  window.addEventListener("resize", () => {
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(render);
  });
}

drawTrainingChart();
