from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import os
import socket

def get_ip_addresses():
    ip_list = []
    try:
        # Get all network interfaces
        interfaces = socket.getaddrinfo(host=socket.gethostname(), port=None, family=socket.AF_INET)
        # Extract unique IPs
        all_ips = set(item[4][0] for item in interfaces)
        # Filter out localhost
        ip_list = [ip for ip in all_ips if not ip.startswith('127.')]
    except Exception as e:
        print(f"Error getting IP addresses: {e}")
    return ip_list

# Create images directory if it doesn't exist
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)

class CameraHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # Get the content length from headers
        content_length = int(self.headers['Content-Length'])
        
        # Read the image data
        image_data = self.rfile.read(content_length)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # Send response back to the camera
        self.send_response(200)
        self.end_headers()
        
        # Print status
        print(f"Received and saved image: {filename} ({len(image_data)} bytes)")

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CameraHandler)
    
    # Print all available IP addresses
    print("\nAvailable IP addresses on this machine:")
    for ip in get_ip_addresses():
        print(f"http://{ip}:{port}")
    
    print(f"\nServer started on port {port}")
    print(f"Saving images to: {os.path.abspath(SAVE_DIR)}")
    print("Waiting for images... (Press Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    PORT = 8080  # Changed to 8080
    run_server(PORT)