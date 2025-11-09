import express from 'express';
import multer from 'multer';
import cors from 'cors';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const execAsync = promisify(exec);

const app = express();
const PORT = 3001;

// Enable CORS
app.use(cors());
app.use(express.json());

// Serve static files from uploads directory
app.use('/files', express.static(path.join(__dirname, '../uploads')));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '../uploads');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, `upload-${uniqueSuffix}${path.extname(file.originalname)}`);
  }
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    if (path.extname(file.originalname).toLowerCase() === '.ply') {
      cb(null, true);
    } else {
      cb(new Error('Only .ply files are allowed'));
    }
  },
  limits: {
    fileSize: 500 * 1024 * 1024, // 500MB limit
  }
});

// Upload and convert endpoint
app.post('/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const inputPath = req.file.path;
    const outputPath = inputPath.replace('.ply', '.sog');

    console.log(`Processing ${inputPath}...`);

    // Try to convert PLY to SOG using splat-transform if available
    let convertedPath = inputPath;
    let conversionSucceeded = false;

    try {
      console.log(`Attempting conversion to ${outputPath}...`);
      const { stdout, stderr } = await execAsync(
        `splat-transform "${inputPath}" "${outputPath}" -w`,
        { maxBuffer: 50 * 1024 * 1024 } // 50MB buffer for large outputs
      );

      if (stderr) {
        console.error('Conversion stderr:', stderr);
      }
      if (stdout) {
        console.log('Conversion stdout:', stdout);
      }

      // Check if output file exists
      await fs.access(outputPath);
      convertedPath = outputPath;
      conversionSucceeded = true;
      console.log('Conversion successful!');

      // Clean up original PLY file after successful conversion
      await fs.unlink(inputPath);
    } catch (conversionError) {
      console.warn('Conversion failed or splat-transform not available:',
        conversionError instanceof Error ? conversionError.message : 'Unknown error');
      console.log('Serving original PLY file instead (Spark supports PLY natively)');
      // Keep the original PLY file
      convertedPath = inputPath;
      conversionSucceeded = false;
    }

    // Get file stats
    const stats = await fs.stat(convertedPath);
    const originalStats = await fs.stat(inputPath);

    // Return URL to the file (converted or original)
    const fileUrl = `http://localhost:${PORT}/files/${path.basename(convertedPath)}`;

    res.json({
      success: true,
      url: fileUrl,
      originalSize: originalStats.size,
      convertedSize: stats.size,
      compressionRatio: conversionSucceeded ? (originalStats.size / stats.size).toFixed(2) : '1.00',
      converted: conversionSucceeded,
      format: conversionSucceeded ? 'sog' : 'ply',
    });

    console.log(`Upload complete: ${fileUrl} (${conversionSucceeded ? 'converted to SOG' : 'serving as PLY'})`);
  } catch (error) {
    console.error('Upload/conversion error:', error);

    // Clean up files on error
    if (req.file) {
      try {
        await fs.unlink(req.file.path);
      } catch {}
    }

    res.status(500).json({
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// List uploaded files
app.get('/files', async (req, res) => {
  try {
    const uploadDir = path.join(__dirname, '../uploads');
    await fs.mkdir(uploadDir, { recursive: true });
    
    const files = await fs.readdir(uploadDir);
    const fileStats = await Promise.all(
      files.map(async (file) => {
        const filePath = path.join(uploadDir, file);
        const stats = await fs.stat(filePath);
        return {
          name: file,
          size: stats.size,
          created: stats.birthtime,
          url: `http://localhost:${PORT}/files/${file}`,
        };
      })
    );
    
    res.json({ files: fileStats });
  } catch (error) {
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to list files'
    });
  }
});

// Error handling middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Server error:', err);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`File server: http://localhost:${PORT}/files`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});

