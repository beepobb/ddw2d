document.addEventListener('DOMContentLoaded', function() {
    // Get the current page
    var currentPage = window.location.pathname;

    // Remove 'active' class from all links
    document.querySelectorAll('nav a').forEach(function(link) {
        link.classList.remove('active');
    });

    // Update aria-current and add 'active' class to the current page's link
    document.querySelectorAll('nav a').forEach(function(link) {
        if (link.getAttribute('href') === currentPage) {
            link.setAttribute('aria-current', 'page');
            link.classList.add('active');
        } else {
            link.setAttribute('aria-current', 'false');
        }
    });
});
