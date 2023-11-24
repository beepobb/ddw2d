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
        

    if (window.location.pathname === '/about') {
        var aboutElement = document.getElementById('about');

        setPaddingClass();

        // Check the window size on page load
        function setPaddingClass() {
            if (window.innerWidth < 1000) {
                // If the window size is less than 1000 pixels, add the class
                aboutElement.classList.remove('py-md-5');
                aboutElement.classList.add('p-md-5');
            } else {
                aboutElement.classList.remove('p-md-5');
                aboutElement.classList.add('py-md-5');
            }
        }
    }

    if (window.location.pathname === '/tool') {
        var slider = document.getElementById("myRange");
        var output = document.getElementById("demo");
        output.innerHTML = slider.value;

        slider.oninput = function() {
        output.innerHTML = this.value;
        }
    }
});

