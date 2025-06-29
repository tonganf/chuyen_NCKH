// Global variables
let chart = null;
let historicalData = null;
let predictionResults = null;
let autoFetchTimeout = null;
let currentData = null;

// DOM Elements
const elements = {
    startDate: document.getElementById('start-date'),
    endDate: document.getElementById('end-date'),
    modelCheckboxes: document.querySelectorAll('input[type="checkbox"][id^="model-"]'),
    fetchDataBtn: document.getElementById('fetch-data-btn'),
    predictBtn: document.getElementById('predict-btn'),
    exportBtn: document.getElementById('export-btn'),
    chartCanvas: document.getElementById('price-chart'),
    chartLoading: document.getElementById('chart-loading'),
    metricsContainer: document.getElementById('metrics-container'),
    statusText: document.getElementById('status-text'),
    lastUpdated: document.getElementById('last-updated'),
    loadingOverlay: document.getElementById('loading-overlay')
};

// Initialize date pickers with default values
const today = new Date();
const oneYearAgo = new Date(today);
oneYearAgo.setFullYear(today.getFullYear() - 1);

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Bitcoin Price Prediction App initialized');
    
    // Set default dates
    const startDateInput = document.getElementById('start-date');
    const endDateInput = document.getElementById('end-date');
    
    if (endDateInput) endDateInput.value = today.toISOString().split('T')[0];
    if (startDateInput) startDateInput.value = oneYearAgo.toISOString().split('T')[0];
    
    initializeChart();
    loadInitialData();
    
    // Event listeners
    const fetchDataBtn = document.getElementById('fetch-data-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    if (fetchDataBtn) {
        fetchDataBtn.addEventListener('click', loadData);
        console.log('✅ Fetch Data button connected');
    } else {
        console.error('❌ Fetch Data button not found');
    }
    
    if (predictBtn) {
        predictBtn.addEventListener('click', makePredictions);
        console.log('✅ Predict button connected');
    } else {
        console.error('❌ Predict button not found');
    }
});

function initializeChart() {
    const ctx = document.getElementById('price-chart');
    if (!ctx) {
        console.error('❌ Canvas element with id "price-chart" not found');
        return;
    }
    
    console.log('📊 Initializing chart...');
    
    // Destroy existing chart if it exists
    if (chart) {
        chart.destroy();
    }
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Historical Price',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.1)',
                borderWidth: 2,
                fill: false,
                tension: 0.1,
                pointRadius: 1,
                pointHoverRadius: 5
            }, {
                label: 'ARIMA Prediction',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                tension: 0.1,
                pointRadius: 2,
                pointHoverRadius: 5,
                spanGaps: true
            }, {
                label: 'SVM Prediction',
                data: [],
                borderColor: 'rgb(255, 205, 86)',
                backgroundColor: 'rgba(255, 205, 86, 0.1)',
                borderWidth: 2,
                borderDash: [10, 5],
                fill: false,
                tension: 0.1,
                pointRadius: 2,
                pointHoverRadius: 5,
                spanGaps: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Bitcoin Price History & Predictions'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            label += '$' + context.parsed.y.toLocaleString();
                            return label;
                        }
                    }
                }
            },
            elements: {
                point: {
                    radius: 2,
                    hoverRadius: 5
                }
            }
        }
    });
    
    console.log('✅ Chart initialized successfully');
}

function loadInitialData() {
    console.log('📅 Loading initial data...');
    loadData();
}

async function loadData() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    
    if (!startDate || !endDate) {
        alert('Please select both start and end dates');
        return;
    }
    
    console.log(`📅 Loading data from ${startDate} to ${endDate}`);
    showLoading(true);
    updateStatus('Loading historical data...');
    
    try {
        const response = await fetch(`/api/data?start_date=${startDate}&end_date=${endDate}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✅ Data loaded successfully:', data);
        console.log(`📊 Received ${data.dates?.length || 0} data points`);
        
        if (!data.dates || !data.prices || data.dates.length === 0) {
            throw new Error('No data received from API');
        }
        
        currentData = data;
        updateChart(data);
        clearPredictions();
        updateStatus(`Data loaded successfully (${data.dates.length} points)`);
        updateLastUpdated();
        
    } catch (error) {
        console.error('❌ Error loading data:', error);
        updateStatus('Failed to load data: ' + error.message);
        showError('Failed to load data. Please try again.');
    } finally {
        showLoading(false);
    }
}

async function makePredictions() {
    if (!currentData || !currentData.dates || currentData.dates.length === 0) {
        alert('Please load historical data first');
        return;
    }
    
    const horizon = 30; // Fixed horizon for now
    console.log(`🔮 Making predictions for ${horizon} days`);
    
    showLoading(true);
    updateStatus('Generating predictions...');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                horizon: horizon,
                end_date: currentData.dates[currentData.dates.length - 1]
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const predictions = await response.json();
        console.log('✅ Predictions received:', predictions);
        
        updateChartWithPredictions(predictions);
        updatePerformanceMetrics(predictions.performance_metrics || {});
        updatePredictionSummary(predictions);
        updateStatus('Predictions generated successfully');
        
    } catch (error) {
        console.error('❌ Error making predictions:', error);
        updateStatus('Failed to generate predictions: ' + error.message);
        alert('Failed to make predictions. Please try again.');
    } finally {
        showLoading(false);
    }
}

function updateChart(data) {
    if (!chart) {
        console.error('❌ Chart not initialized');
        return;
    }
    
    if (!data.dates || !data.prices) {
        console.error('❌ Invalid data for chart update:', data);
        return;
    }
    
    // Validate data integrity
    if (data.dates.length !== data.prices.length) {
        console.error(`❌ Data mismatch: ${data.dates.length} dates vs ${data.prices.length} prices`);
        return;
    }
    
    console.log('📊 Updating chart with historical data');
    console.log(`📊 Data points: ${data.dates.length}`);
    
    // Update chart labels and data
    chart.data.labels = data.dates;
    chart.data.datasets[0].data = data.prices;
    
    // Clear prediction data
    chart.data.datasets[1].data = [];
    chart.data.datasets[2].data = [];
    
    console.log('📊 Chart labels:', data.dates.slice(0, 3));
    console.log('📊 Chart data:', data.prices.slice(0, 3));
    
    // Force chart update
    chart.update('active');
    console.log('✅ Chart updated with historical data');
    console.log('📊 Final chart data:', {
        labelsCount: chart.data.labels.length,
        dataCount: chart.data.datasets[0].data.length,
        sampleLabels: chart.data.labels.slice(0, 3),
        sampleData: chart.data.datasets[0].data.slice(0, 3)
    });
}

function updateChartWithPredictions(predictions) {
    if (!chart || !predictions.predictions) {
        console.error('❌ Invalid predictions data');
        return;
    }
    
    console.log('📊 Updating chart with predictions');
    
    // Extend labels to include prediction dates
    let allDates = [...chart.data.labels];
    
    // Update ARIMA predictions
    if (predictions.predictions.arima) {
        const arima = predictions.predictions.arima;
        
        // Add prediction dates to labels if not already present
        arima.dates.forEach(date => {
            if (!allDates.includes(date)) {
                allDates.push(date);
            }
        });
        
        // Create full data array with nulls for historical period
        const arimaFullData = new Array(chart.data.labels.length).fill(null);
        arimaFullData.push(...arima.predictions);
        
        chart.data.datasets[1].data = arimaFullData;
        console.log('✅ ARIMA predictions added to chart');
    }
    
    // Update SVM predictions
    if (predictions.predictions.svm) {
        const svm = predictions.predictions.svm;
        
        // Add prediction dates to labels if not already present
        svm.dates.forEach(date => {
            if (!allDates.includes(date)) {
                allDates.push(date);
            }
        });
        
        // Create full data array with nulls for historical period
        const svmFullData = new Array(chart.data.labels.length).fill(null);
        svmFullData.push(...svm.predictions);
        
        chart.data.datasets[2].data = svmFullData;
        console.log('✅ SVM predictions added to chart');
    }
    
    // Update chart labels
    chart.data.labels = allDates;
    
    chart.update('active');
    console.log('✅ Chart updated with predictions');
}

function updatePerformanceMetrics(metrics) {
    console.log('📈 Updating performance metrics:', metrics);
    
    const metricsContainer = document.getElementById('performanceMetrics');
    if (!metricsContainer) {
        console.error('❌ Performance metrics container not found');
        return;
    }
    
    let metricsHTML = '<h3>📊 Performance Metrics</h3>';
    
    if (Object.keys(metrics).length === 0) {
        metricsHTML += '<p class="text-muted">No performance data available. Need more historical data for evaluation.</p>';
    } else {
        // ARIMA Metrics
        if (metrics.arima) {
            metricsHTML += `
                <div class="metric-group">
                    <h4>ARIMA Model</h4>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">$${metrics.arima.mae.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">RMSE:</span>
                            <span class="metric-value">$${metrics.arima.rmse.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">MAPE:</span>
                            <span class="metric-value">${metrics.arima.mape.toFixed(2)}%</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Directional Accuracy:</span>
                            <span class="metric-value">${metrics.arima.directional_accuracy.toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // SVM Metrics
        if (metrics.svm) {
            metricsHTML += `
                <div class="metric-group">
                    <h4>SVM Model</h4>
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <span class="metric-label">MAE:</span>
                            <span class="metric-value">$${metrics.svm.mae.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">RMSE:</span>
                            <span class="metric-value">$${metrics.svm.rmse.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">MAPE:</span>
                            <span class="metric-value">${metrics.svm.mape.toFixed(2)}%</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Directional Accuracy:</span>
                            <span class="metric-value">${metrics.svm.directional_accuracy.toFixed(1)}%</span>
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    metricsContainer.innerHTML = metricsHTML;
}

function updatePredictionSummary(predictions) {
    console.log('📋 Updating prediction summary');
    
    const summaryContainer = document.getElementById('predictionSummary');
    if (!summaryContainer) {
        console.error('❌ Prediction summary container not found');
        return;
    }
    
    let summaryHTML = '<h3>🔮 Prediction Summary</h3>';
    
    if (predictions.current_price) {
        summaryHTML += `<p><strong>Current Price:</strong> $${predictions.current_price.toLocaleString()}</p>`;
    }
    
    if (predictions.last_update) {
        summaryHTML += `<p><strong>Last Update:</strong> ${predictions.last_update}</p>`;
    }
    
    // ARIMA Summary
    if (predictions.predictions.arima) {
        const arima = predictions.predictions.arima;
        const lastPrediction = arima.predictions[arima.predictions.length - 1];
        const firstPrediction = arima.predictions[0];
        const trend = lastPrediction > firstPrediction ? 'Upward' : 'Downward';
        const trendEmoji = lastPrediction > firstPrediction ? '📈' : '📉';
        
        summaryHTML += `
            <div class="prediction-summary">
                <h4>ARIMA Prediction</h4>
                <p><strong>30-day outlook:</strong> ${trendEmoji} ${trend} trend</p>
                <p><strong>Predicted price (30 days):</strong> $${lastPrediction.toLocaleString()}</p>
                <p><strong>Expected change:</strong> ${((lastPrediction - predictions.current_price) / predictions.current_price * 100).toFixed(2)}%</p>
            </div>
        `;
    }
    
    // SVM Summary
    if (predictions.predictions.svm) {
        const svm = predictions.predictions.svm;
        const lastPrediction = svm.predictions[svm.predictions.length - 1];
        
        // Count up/down trends
        const upDays = svm.trends ? svm.trends.filter(t => t === 1).length : 0;
        const downDays = svm.trends ? svm.trends.filter(t => t === 0).length : 0;
        const totalDays = svm.trends ? svm.trends.length : 0;
        
        const overallTrend = upDays > downDays ? 'Bullish' : 'Bearish';
        const trendEmoji = upDays > downDays ? '🐂' : '🐻';
        
        summaryHTML += `
            <div class="prediction-summary">
                <h4>SVM Prediction</h4>
                <p><strong>30-day outlook:</strong> ${trendEmoji} ${overallTrend} (${upDays}↑ / ${downDays}↓ days)</p>
                <p><strong>Predicted price (30 days):</strong> $${lastPrediction.toLocaleString()}</p>
                <p><strong>Expected change:</strong> ${((lastPrediction - predictions.current_price) / predictions.current_price * 100).toFixed(2)}%</p>
                <p><strong>Bullish probability:</strong> ${(upDays / totalDays * 100).toFixed(1)}%</p>
            </div>
        `;
    }
    
    summaryContainer.innerHTML = summaryHTML;
}

function clearPredictions() {
    // Clear prediction data from chart
    if (chart) {
        chart.data.datasets[1].data = [];
        chart.data.datasets[2].data = [];
        chart.update('none');
    }
    
    // Clear metrics and summary
    const metricsContainer = document.getElementById('performanceMetrics');
    const summaryContainer = document.getElementById('predictionSummary');
    
    if (metricsContainer) {
        metricsContainer.innerHTML = '<h3>📊 Performance Metrics</h3><p class="text-muted">Make a prediction to see performance metrics.</p>';
    }
    
    if (summaryContainer) {
        summaryContainer.innerHTML = '<h3>🔮 Prediction Summary</h3><p class="text-muted">Make a prediction to see summary.</p>';
    }
}

function showLoading(show) {
    const loadingElement = document.getElementById('loading-overlay');
    if (loadingElement) {
        if (show) {
            loadingElement.classList.add('show');
        } else {
            loadingElement.classList.remove('show');
        }
    }
    
    // Disable buttons during loading
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = show;
    });
}

function updateStatus(message) {
    const statusElement = document.getElementById('status-text');
    if (statusElement) {
        statusElement.textContent = message;
    }
    console.log('📊 Status:', message);
}

// Update last updated timestamp
function updateLastUpdated() {
    const lastUpdatedElement = document.getElementById('last-updated');
    if (lastUpdatedElement) {
        lastUpdatedElement.textContent = new Date().toLocaleString();
    }
}

// Show error message
function showError(message) {
    updateStatus(`Error: ${message}`);
}

// Show status message with type
function showStatus(message, type = 'info') {
    const statusEl = document.getElementById('status-text');
    if (statusEl) {
        statusEl.textContent = message;
        statusEl.className = `status-value status-${type}`;
        
        // Clear status after 5 seconds
        setTimeout(() => {
            statusEl.textContent = 'Ready';
            statusEl.className = 'status-value';
        }, 5000);
    }
}

// Format date for file name
function formatDate(date) {
    return date.toISOString().split('T')[0];
} 