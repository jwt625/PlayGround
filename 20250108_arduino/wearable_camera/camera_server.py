from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import os

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
    print(f"Server started on port {port}")
    print(f"Saving images to: {os.path.abspath(SAVE_DIR)}")
    print("Waiting for images... (Press Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    PORT = 8080
    run_server(PORT)