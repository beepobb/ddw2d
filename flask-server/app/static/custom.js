document.addEventListener('DOMContentLoaded', function() {
    var newYearElement = document.getElementById('new-year');
    var betterWorldElement = document.getElementById('better-world');
    // Delay the fade-in for "Better World" by 2 seconds (2000 milliseconds)
    setTimeout(function() {
        betterWorldElement.style.animationDelay = '0s';
        betterWorldElement.classList.add('fade-in-text');
        betterWorldElement.style.opacity = 1;
    }, 1000);

    // Start the fade-in for "New Year"
    newYearElement.classList.add('fade-in-text');
});