document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname === '/home' || window.location.pathname === '/') {
        var newYearElement = document.getElementById('new-year');
        var betterWorldElement = document.getElementById('better-world');
        
        // Delay the fade-in for "Better World" by 2 seconds (2000 milliseconds)
        setTimeout(function() {
            betterWorldElement.style.animationDelay = '0s';
            betterWorldElement.classList.add('fade-in-text');
            betterWorldElement.style.opacity = 1;
        }, 1000);

        document.querySelectorAll('.btn-box').forEach(function(btnboxElement) {
            setTimeout(function() {
                btnboxElement.style.animationDelay = '1s';
                btnboxElement.classList.add('fade-in-up');
            }, 950);
        })

        // Start the fade-in for "New Year"
        newYearElement.classList.add('fade-in-text');
    }
});

