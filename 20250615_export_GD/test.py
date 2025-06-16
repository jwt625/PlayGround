import os
import io
import mimetypes
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Replace with your path
EXPORT_ROOT = 'exported_drive'  # Local output folder

# Auth
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

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

def download_file(file_id, file_name, local_path, mime_type):
    os.makedirs(local_path, exist_ok=True)
    full_path = os.path.join(local_path, sanitize_filename(file_name))

    if mime_type in EXPORT_MIMETYPES:
        exports = EXPORT_MIMETYPES[mime_type]
        if not isinstance(exports, list):
            exports = [exports]
        for export_mime, ext in exports:
            request = drive_service.files().export(fileId=file_id, mimeType=export_mime)
            out_path = full_path + ext
            with io.FileIO(out_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
    elif mime_type == 'application/vnd.google-apps.folder':
        os.makedirs(os.path.join(local_path, file_name), exist_ok=True)
    else:
        request = drive_service.files().get_media(fileId=file_id)
        ext = mimetypes.guess_extension(mime_type) or ''
        out_path = full_path + ext
        with io.FileIO(out_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

def process_folder(folder_id, local_path):
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        results = drive_service.files().list(
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
            download_file(file_id, name, local_path, mime_type)
            if mime_type == 'application/vnd.google-apps.folder':
                process_folder(file_id, target_path)
        page_token = results.get('nextPageToken', None)
        if not page_token:
            break

# Entry point
ROOT_FOLDER_ID = 'your-root-folder-id'  # Replace with your folder ID
os.makedirs(EXPORT_ROOT, exist_ok=True)
process_folder(ROOT_FOLDER_ID, EXPORT_ROOT)
