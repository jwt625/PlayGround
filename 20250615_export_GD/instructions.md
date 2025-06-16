# Google Drive OAuth Setup Instructions

## Step 1: Create OAuth 2.0 Credentials

Since your Google Drive is managed by an organization, you'll need to create OAuth credentials through a personal Google account or request access from your organization.

### Option A: Using Personal Google Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing one)
3. Enable the Google Drive API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click on it and press "Enable"

4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
   - Choose "Desktop application"
   - Give it a name (e.g., "Drive Exporter")
   - Click "Create"

5. Download the credentials:
   - Click the download button next to your new OAuth client
   - Save the file as `credentials.json` in your project folder

### Option B: Request from Organization Admin

If you need to access organizational Drive data, ask your IT admin to:
1. Create an OAuth 2.0 client in your organization's Google Workspace
2. Grant you access to the Drive API
3. Provide you with the `credentials.json` file

## Step 2: Install Required Python Packages

```bash
pip install google-auth google-auth-oauthlib google-api-python-client
```

## Step 3: Run the Script

1. Place `credentials.json` in the same folder as `OAuth_test.py`
2. Run the script:
   ```bash
   python OAuth_test.py
   ```

3. First run will:
   - Open your web browser
   - Ask you to sign in to Google
   - Request permission to access your Drive (read-only)
   - Create a `token.json` file for future runs

## Step 4: Customize Export

- Change `ROOT_FOLDER_ID` in the script to export specific folders
- Modify `EXPORT_MIMETYPES` to change export formats
- Update `EXPORT_ROOT` to change output directory

## File Structure After Setup

```
your-project/
├── OAuth_test.py
├── credentials.json    # OAuth credentials from Google Cloud Console
├── token.json         # Auto-generated after first authentication
└── exported_drive/    # Your exported files will be here
```

## Troubleshooting

- **"Access blocked"**: Your org may restrict external OAuth apps
- **"Scope not approved"**: Request Drive API access from admin
- **"Invalid credentials"**: Re-download `credentials.json`
- **"Permission denied"**: Check if you have access to the Drive files

## Security Notes

- Keep `credentials.json` and `token.json` private
- The script only requests read-only access
- Tokens expire and will auto-refresh when needed