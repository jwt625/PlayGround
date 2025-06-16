import os
import io
import mimetypes
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json'  # Downloaded from Google Cloud Console
TOKEN_FILE = 'token.json'  # Will be created after first auth
EXPORT_ROOT = 'exported_drive'  # Local output folder

def authenticate():
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return creds

# Mimetypes for Google file exports
EXPORT_MIMETYPES = {
    'application/vnd.google-apps.document': ('text/html', '.html'),
    'application/vnd.google-apps.spreadsheet': ('text/html', '.html'),
    'application/vnd.google-apps.presentation': [
        ('application/pdf', '.pdf'),
        ('application/vnd.openxmlformats-officedocument.presentationml.presentation', '.pptx'),
    ]
}

def sanitize_filename(name):
    return "".join(c for c in name if c.isalnum() or c in " ._-").rstrip()

def download_file(service, file_id, file_name, local_path, mime_type):
    os.makedirs(local_path, exist_ok=True)
    full_path = os.path.join(local_path, sanitize_filename(file_name))

    if mime_type in EXPORT_MIMETYPES:
        exports = EXPORT_MIMETYPES[mime_type]
        if not isinstance(exports, list):
            exports = [exports]
        for export_mime, ext in exports:
            request = service.files().export(fileId=file_id, mimeType=export_mime)
            out_path = full_path + ext
            with io.FileIO(out_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
    elif mime_type == 'application/vnd.google-apps.folder':
        os.makedirs(os.path.join(local_path, file_name), exist_ok=True)
    else:
        request = service.files().get_media(fileId=file_id)
        ext = mimetypes.guess_extension(mime_type) or ''
        out_path = full_path + ext
        with io.FileIO(out_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

def process_folder(service, folder_id, local_path):
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        results = service.files().list(
            q=query,
            spaces='drive',
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
        for item in results.get('files', []):
            file_id = item['id']
            name = item['name']
            mime_type = item['mimeType']
            print(f"Processing: {name} ({mime_type})")
            target_path = os.path.join(local_path, sanitize_filename(name)) if mime_type == 'application/vnd.google-apps.folder' else local_path
            download_file(service, file_id, name, local_path, mime_type)
            if mime_type == 'application/vnd.google-apps.folder':
                process_folder(service, file_id, target_path)
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

def main():
    # Authenticate and get credentials
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    
    # Get root folder or specify a folder ID
    ROOT_FOLDER_ID = 'root'  # Use 'root' for entire drive or replace with specific folder ID
    
    os.makedirs(EXPORT_ROOT, exist_ok=True)
    process_folder(service, ROOT_FOLDER_ID, EXPORT_ROOT)

if __name__ == '__main__':
    main()
