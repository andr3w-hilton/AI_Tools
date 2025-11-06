// Configuration
const API_BASE = 'https://api.digitalocean.com/v2';

// State management
let apiToken = null;
let dropletId = null;
let isLoading = false;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
});

function checkAuth() {
    const stored = localStorage.getItem('dropletAuth');
    if (stored) {
        try {
            const auth = JSON.parse(stored);
            apiToken = auth.token;
            dropletId = auth.dropletId;
            showControlView();
            loadDropletStatus();
        } catch (e) {
            logout();
        }
    } else {
        showLoginView();
    }
}

function login(event) {
    event.preventDefault();
    const token = document.getElementById('apiToken').value;
    const droplet = document.getElementById('dropletId').value;
    const password = document.getElementById('password').value;

    // Store encrypted credentials
    const auth = {
        token: token,
        dropletId: droplet,
        password: btoa(password) // Simple encoding, not real encryption
    };

    localStorage.setItem('dropletAuth', JSON.stringify(auth));
    apiToken = token;
    dropletId = droplet;

    showControlView();
    loadDropletStatus();
}

function logout() {
    localStorage.removeItem('dropletAuth');
    apiToken = null;
    dropletId = null;
    showLoginView();
}

function showLoginView() {
    document.getElementById('loginView').style.display = 'block';
    document.getElementById('controlView').style.display = 'none';
}

function showControlView() {
    document.getElementById('loginView').style.display = 'none';
    document.getElementById('controlView').style.display = 'block';
}

async function loadDropletStatus() {
    showLoading(true);
    hideError();

    try {
        const response = await fetch(`${API_BASE}/droplets/${dropletId}`, {
            headers: {
                'Authorization': `Bearer ${apiToken}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            if (response.status === 401) {
                throw new Error('Invalid API token. Please check your credentials.');
            }
            throw new Error(`API Error: ${response.status}`);
        }

        const data = await response.json();
        updateUI(data.droplet);
        showLoading(false);
        
    } catch (error) {
        console.error('Error loading droplet:', error);
        showError(error.message);
        showLoading(false);
    }
}

function updateUI(droplet) {
    const isActive = droplet.status === 'active';
    
    // Update droplet name
    document.getElementById('dropletName').textContent = droplet.name;
    
    // Update status
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    statusIndicator.className = `status-indicator ${isActive ? 'active' : 'off'}`;
    statusText.textContent = isActive ? 'Online' : 'Offline';
    
    // Update power button
    const powerButton = document.getElementById('powerButton');
    powerButton.className = `power-button ${isActive ? 'on' : 'off'}`;
    powerButton.textContent = isActive ? '⚡' : '🔌';
    powerButton.disabled = false;
    
    // Update info
    const publicIP = droplet.networks.v4.find(n => n.type === 'public');
    document.getElementById('ipAddress').textContent = publicIP ? publicIP.ip_address : 'N/A';
    document.getElementById('region').textContent = droplet.region.name;
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
}

async function toggleDroplet() {
    if (isLoading) return;
    
    const powerButton = document.getElementById('powerButton');
    powerButton.disabled = true;
    isLoading = true;
    
    try {
        // Get current status
        const statusResponse = await fetch(`${API_BASE}/droplets/${dropletId}`, {
            headers: {
                'Authorization': `Bearer ${apiToken}`,
                'Content-Type': 'application/json'
            }
        });
        
        const statusData = await statusResponse.json();
        const isActive = statusData.droplet.status === 'active';
        
        // Toggle power
        const action = isActive ? 'power_off' : 'power_on';
        const actionResponse = await fetch(`${API_BASE}/droplets/${dropletId}/actions`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ type: action })
        });
        
        if (!actionResponse.ok) {
            throw new Error(`Failed to ${action} droplet`);
        }
        
        // Show feedback
        const statusText = document.getElementById('statusText');
        statusText.textContent = isActive ? 'Shutting down...' : 'Starting up...';
        
        // Poll for status update
        setTimeout(() => {
            pollStatus();
        }, 3000);
        
    } catch (error) {
        console.error('Error toggling droplet:', error);
        showError(error.message);
        powerButton.disabled = false;
        isLoading = false;
    }
}

async function pollStatus() {
    let attempts = 0;
    const maxAttempts = 20; // 60 seconds max
    
    const poll = setInterval(async () => {
        attempts++;
        
        try {
            const response = await fetch(`${API_BASE}/droplets/${dropletId}`, {
                headers: {
                    'Authorization': `Bearer ${apiToken}`,
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            const status = data.droplet.status;
            
            // Check if status changed to stable state
            if (status === 'active' || status === 'off') {
                clearInterval(poll);
                updateUI(data.droplet);
                isLoading = false;
            }
            
            // Timeout after max attempts
            if (attempts >= maxAttempts) {
                clearInterval(poll);
                loadDropletStatus();
            }
            
        } catch (error) {
            clearInterval(poll);
            loadDropletStatus();
        }
    }, 3000); // Check every 3 seconds
}

function showLoading(show) {
    document.getElementById('loadingState').style.display = show ? 'block' : 'none';
    document.getElementById('controlState').style.display = show ? 'none' : 'block';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorState').style.display = 'block';
    document.getElementById('controlState').style.display = 'none';
    document.getElementById('loadingState').style.display = 'none';
}

function hideError() {
    document.getElementById('errorState').style.display = 'none';
}

// Auto-refresh every 30 seconds
setInterval(() => {
    if (apiToken && !isLoading) {
        loadDropletStatus();
    }
}, 30000);
