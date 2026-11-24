set -euo pipefail
mkdir -p src
cat > src/motion.css <<'CSS'
/* Motion for the account dashboard. */

/* Hero background drifts slowly behind the headline. */
.hero-bg {
  animation: drift 20s ease-in-out infinite alternate;
}
@keyframes drift {
  from { transform: translateY(0) scale(1.05); }
  to { transform: translateY(-40px) scale(1.05); }
}

/* Cards lift and grow on hover. */
.card {
  transition: transform 200ms ease-out, box-shadow 200ms ease-out;
}
.card:hover {
  transform: translateY(-4px) scale(1.03);
  box-shadow: 0 8px 24px rgb(0 0 0 / 0.15);
}

/* Side drawer slides in from the right. */
.drawer {
  transform: translateX(100%);
  transition: transform 300ms ease-out;
}
.drawer.is-open {
  transform: translateX(0);
}

/* Testimonials carousel scrolls on its own. */
.carousel-track {
  animation: carousel 30s linear infinite;
}
@keyframes carousel {
  from { transform: translateX(0); }
  to { transform: translateX(-50%); }
}

/* "Saved" toast fades in after the user saves a form. */
.toast {
  animation: toast-in 200ms ease-out forwards;
}
@keyframes toast-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Primary button changes color on hover and keyboard focus. */
.btn-primary {
  background-color: #2563eb;
  transition: background-color 150ms ease-out;
}
.btn-primary:hover,
.btn-primary:focus-visible {
  background-color: #1d4ed8;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
CSS
