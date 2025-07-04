document.addEventListener('DOMContentLoaded', () => {
    const links = document.querySelectorAll('.nav-link');
    const current = window.location.pathname.replace(/\/$/, ''); // sin slash final
    links.forEach(link => {
      const href = link.getAttribute('href').replace(/\/$/, '');
      link.classList.toggle('active', href === current);
    });
  });