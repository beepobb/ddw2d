document.addEventListener('DOMContentLoaded', function() {
    const countrySelect = document.getElementById('country');

    // Fetch available countries from the server
    fetch('/get_available_countries')
        .then(response => response.json())
        .then(data => {
            // Assuming the response has a 'countries' property
            const availableCountries = data.countries;

            // Populate the dropdown options with available countries
            availableCountries.forEach(country => {
                const option = document.createElement('option');
                option.value = country;
                option.textContent = country;
                countrySelect.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching available countries:', error);
        });
});
async function runLinearRegression() {
    const country = document.getElementById('country').value;
    const money = document.getElementById('money').value;

    // Use fetch to send a request to your server with the user input
    const response = await fetch('/run_linear_regression', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ country, money }),
    });

    if (!response.ok) {
        console.error('Error:', response.statusText);
        return;
    }

    const result = await response.json();

    // Display the result on the webpage
    const resultElement = document.getElementById('result');
    resultElement.innerHTML = `Predicted cost in target currency: ${result.prediction}`;
}

// Update the displayed value next to the slider
document.getElementById('money').addEventListener('input', function() {
    document.getElementById('moneyValue').innerText = this.value;
});


