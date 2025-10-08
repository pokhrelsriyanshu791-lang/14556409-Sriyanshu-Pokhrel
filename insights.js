let chart = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize chart
    initializeChart();
    
    // Load default data (3 days)
    loadData(72, 'All');
    
    // Setup event listeners
    setupEventListeners();
});

function setupEventListeners() {
    const timeframeSelect = document.getElementById('timeframe');
    const categorySelect = document.getElementById('category');
    const customDaysDiv = document.getElementById('custom-days');
    const customDaysInput = document.getElementById('custom-days-input');
    
    timeframeSelect.addEventListener('change', function() {
        if (this.value === 'custom') {
            customDaysDiv.style.display = 'block';
        } else {
            customDaysDiv.style.display = 'none';
            const hours = parseInt(this.value);
            const category = categorySelect.value;
            loadData(hours, category);
        }
    });
    
    customDaysInput.addEventListener('change', function() {
        if (this.value && this.value > 0) {
            const hours = parseInt(this.value) * 24;
            const category = categorySelect.value;
            loadData(hours, category);
        }
    });
    
    categorySelect.addEventListener('change', function() {
        const timeframe = timeframeSelect.value;
        let hours;
        
        if (timeframe === 'custom') {
            const days = parseInt(customDaysInput.value) || 3;
            hours = days * 24;
        } else {
            hours = parseInt(timeframe);
        }
        
        loadData(hours, this.value);
    });
}

function initializeChart() {
    const ctx = document.getElementById('demandChart').getContext('2d');
    
    chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Sales Count',
                data: [],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(255, 205, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 205, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Top Products by Demand'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeInOutQuart'
            }
        }
    });
}

function showLoading() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('chart-container').style.display = 'none';
    document.getElementById('table-container').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('chart-container').style.display = 'block';
    document.getElementById('table-container').style.display = 'block';
}

function fadeIn(element) {
    element.classList.add('fade-in');
    element.classList.add('show');
}

function loadData(hours, category) {
    showLoading();
    
    const url = `/api/insights/top_products?hours=${hours}&category=${encodeURIComponent(category)}`;
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            updateChart(data.products);
            updateTable(data.products, data.total);
            hideLoading();
            
            // Fade in effect
            setTimeout(() => {
                fadeIn(document.getElementById('chart-container'));
                fadeIn(document.getElementById('table-container'));
            }, 100);
        })
        .catch(error => {
            console.error('Error loading data:', error);
            hideLoading();
            
            // Show error message
            document.getElementById('products-table').innerHTML = 
                '<tr><td colspan="3" class="text-center text-danger">Error loading data</td></tr>';
        });
}

function updateChart(products) {
    const labels = products.map(p => p.name);
    const data = products.map(p => p.count);
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update('active');
}

function updateTable(products, total) {
    const tbody = document.getElementById('products-table');
    
    if (products.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="text-center">No data available</td></tr>';
        return;
    }
    
    let html = '';
    products.forEach(product => {
        html += `
            <tr>
                <td>${product.name}</td>
                <td>${product.count}</td>
                <td>${product.percentage}%</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

// Handle window resize for responsive chart
window.addEventListener('resize', function() {
    if (chart) {
        chart.resize();
    }
});
