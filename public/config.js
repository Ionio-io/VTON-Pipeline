// ================================================================
// CONFIGURATION LOADER
// ================================================================
// This file loads API credentials from the server environment.
// Uses server.py to read .env file and provide config via /api/config

var FAL_KEY = 'YOUR_FAL_KEY';
var RP_EP = 'YOUR_RUNPOD_ENDPOINT_ID';
var RP_KEY = 'YOUR_RUNPOD_KEY';

// Load config from server endpoint
async function loadConfig() {
  try {
    console.log('[CONFIG] Fetching from /api/config...');
    const response = await fetch('/api/config', {
      method: 'GET',
      headers: { 'Accept': 'application/json' }
    });
    
    if (response.ok) {
      const config = await response.json();
      console.log('[CONFIG] Received:', { FAL_KEY: !!config.FAL_KEY, RP_EP: !!config.RP_EP, RP_KEY: !!config.RP_KEY });
      
      if (config.FAL_KEY) FAL_KEY = config.FAL_KEY;
      if (config.RP_EP) RP_EP = config.RP_EP;
      if (config.RP_KEY) RP_KEY = config.RP_KEY;
      
      console.log('[CONFIG] Configuration loaded successfully');
    } else {
      console.error('[CONFIG] Server returned status:', response.status);
    }
  } catch (error) {
    console.error('[CONFIG] Error loading configuration:', error);
    console.error('[CONFIG] Make sure you are running server.py, not python3 -m http.server');
  }
}

// Load config immediately
loadConfig();
