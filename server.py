#!/usr/bin/env python3
"""
Simple HTTP server for Virtual Try-On that loads environment variables from .env file
and provides them to the frontend.
"""

import os
import sys
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Load environment variables from .env file
def load_env_file():
    env_file = Path(__file__).parent / '.env'
    env_vars = {}
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
    
    return env_vars

# Load environment variables
ENV_VARS = load_env_file()

class ConfigHandler(SimpleHTTPRequestHandler):
    """Custom request handler"""
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        # Serve config endpoint
        if parsed_path.path == '/api/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            config = {
                'FAL_KEY': ENV_VARS.get('FAL_KEY', ''),
                'RP_EP': ENV_VARS.get('RP_EP', ''),
                'RP_KEY': ENV_VARS.get('RP_KEY', '')
            }
            self.wfile.write(json.dumps(config).encode())
            return
        
        # Handle CORS preflight requests
        if self.command == 'OPTIONS':
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.end_headers()
            return
        
        # Serve regular files
        super().do_GET()
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{self.log_date_time_string()}] {format % args}")

if __name__ == '__main__':
    # Change to the script directory
    os.chdir(Path(__file__).parent)
    
    port = int(os.environ.get('PORT', 8080))
    
    # Create server
    server_address = ('', port)
    httpd = HTTPServer(server_address, ConfigHandler)
    
    print(f"Virtual Try-On Server")
    print(f"====================")
    print(f"Port: {port}")
    print(f"URL: http://localhost:{port}/public/")
    print(f"\nEnvironment Variables Loaded:")
    print(f"  FAL_KEY: {'✓' if ENV_VARS.get('FAL_KEY') else '✗ NOT SET'}")
    print(f"  RP_EP: {'✓' if ENV_VARS.get('RP_EP') else '✗ (optional)'}")
    print(f"  RP_KEY: {'✓' if ENV_VARS.get('RP_KEY') else '✗ (optional)'}")
    print(f"\nPress Ctrl+C to stop the server\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
