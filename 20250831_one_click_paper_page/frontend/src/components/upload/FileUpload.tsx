import React, { useState, useRef, useEffect } from 'react';
import { FileProcessor, DragDropHandler, BatchFileUploader } from '../../lib/github/fileUpload';
import type { FileTypeDetection } from '../../types/github';

interface FileUploadProps {
  onFilesSelected: (files: File[]) => void;
  onOverleafUrl: (url: string) => void;
  maxFiles?: number;
  acceptedTypes?: string[];
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFilesSelected,
  onOverleafUrl,
  maxFiles = 5,
  acceptedTypes = ['.pdf', '.docx', '.tex', '.zip'],
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [overleafUrl, setOverleafUrl] = useState('');
  const [uploadMode, setUploadMode] = useState<'file' | 'overleaf'>('file');
  const [fileDetections, setFileDetections] = useState<FileTypeDetection[]>([]);
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const dragDropHandler = useRef<DragDropHandler | null>(null);

  useEffect(() => {
    if (dropZoneRef.current) {
      dragDropHandler.current = new DragDropHandler(
        dropZoneRef.current,
        handleFilesDropped
      );
    }

    return () => {
      dragDropHandler.current?.destroy();
    };
  }, []);

  const handleFilesDropped = (files: FileList) => {
    handleFileSelection(Array.from(files));
  };

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      handleFileSelection(Array.from(files));
    }
  };

  const handleFileSelection = (files: File[]) => {
    const validFiles: File[] = [];
    const detections: FileTypeDetection[] = [];

    files.forEach(file => {
      const validation = FileProcessor.validateFile(file);
      if (validation.valid) {
        validFiles.push(file);
        const detection = FileProcessor.detectFileType(file);
        detections.push(detection);
      } else {
        alert(`Error with file ${file.name}: ${validation.error}`);
      }
    });

    if (validFiles.length + selectedFiles.length > maxFiles) {
      alert(`Maximum ${maxFiles} files allowed`);
      return;
    }

    const newFiles = [...selectedFiles, ...validFiles];
    const newDetections = [...fileDetections, ...detections];
    
    setSelectedFiles(newFiles);
    setFileDetections(newDetections);
    onFilesSelected(newFiles);
  };

  const removeFile = (index: number) => {
    const newFiles = selectedFiles.filter((_, i) => i !== index);
    const newDetections = fileDetections.filter((_, i) => i !== index);
    
    setSelectedFiles(newFiles);
    setFileDetections(newDetections);
    onFilesSelected(newFiles);
  };

  const handleOverleafSubmit = () => {
    const detection = FileProcessor.validateOverleafUrl(overleafUrl);
    if (detection.confidence > 0.5) {
      onOverleafUrl(overleafUrl);
    } else {
      alert('Please enter a valid Overleaf project URL');
    }
  };

  const getFileTypeIcon = (detection: FileTypeDetection) => {
    switch (detection.type) {
      case 'pdf':
        return '📄';
      case 'docx':
        return '📝';
      case 'latex':
        return '📐';
      case 'zip':
        return '📦';
      case 'overleaf':
        return '🌐';
      default:
        return '📎';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Upload Your Paper</h2>
      
      {/* Upload Mode Selector */}
      <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
        <button
          className={`flex-1 py-2 px-4 rounded-md transition-colors ${
            uploadMode === 'file'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setUploadMode('file')}
        >
          📁 Upload Files
        </button>
        <button
          className={`flex-1 py-2 px-4 rounded-md transition-colors ${
            uploadMode === 'overleaf'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setUploadMode('overleaf')}
        >
          🌐 Overleaf Project
        </button>
      </div>

      {uploadMode === 'file' ? (
        <>
          {/* File Upload Area */}
          <div
            ref={dropZoneRef}
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <div className="mb-4">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                stroke="currentColor"
                fill="none"
                viewBox="0 0 48 48"
              >
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
            <p className="text-lg text-gray-600 mb-2">
              Drag and drop your files here, or{' '}
              <button
                className="text-blue-600 hover:text-blue-700 underline"
                onClick={() => fileInputRef.current?.click()}
              >
                browse
              </button>
            </p>
            <p className="text-sm text-gray-500">
              Supports PDF, DOCX, LaTeX (.tex), and ZIP files up to 50MB
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={acceptedTypes.join(',')}
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>

          {/* Selected Files List */}
          {selectedFiles.length > 0 && (
            <div className="mt-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                Selected Files ({selectedFiles.length})
              </h3>
              <div className="space-y-2">
                {selectedFiles.map((file, index) => {
                  const detection = fileDetections[index];
                  return (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                    >
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">
                          {getFileTypeIcon(detection)}
                        </span>
                        <div>
                          <p className="font-medium text-gray-800">{file.name}</p>
                          <p className="text-sm text-gray-500">
                            {(file.size / 1024 / 1024).toFixed(2)} MB •{' '}
                            <span className={getConfidenceColor(detection.confidence)}>
                              {detection.type.toUpperCase()}
                            </span>
                            {detection.confidence < 1.0 && (
                              <span className="text-gray-400">
                                {' '}({Math.round(detection.confidence * 100)}% confidence)
                              </span>
                            )}
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-red-500 hover:text-red-700 p-1"
                        title="Remove file"
                      >
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                          <path
                            fillRule="evenodd"
                            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                            clipRule="evenodd"
                          />
                        </svg>
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      ) : (
        /* Overleaf URL Input */
        <div className="space-y-4">
          <div>
            <label htmlFor="overleaf-url" className="block text-sm font-medium text-gray-700 mb-2">
              Overleaf Project URL
            </label>
            <input
              id="overleaf-url"
              type="url"
              value={overleafUrl}
              onChange={(e) => setOverleafUrl(e.target.value)}
              placeholder="https://www.overleaf.com/project/..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="mt-2 text-sm text-gray-500">
              Enter the URL of your Overleaf project. Make sure it's set to public or you have sharing enabled.
            </p>
          </div>
          <button
            onClick={handleOverleafSubmit}
            disabled={!overleafUrl.trim()}
            className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            Connect Overleaf Project
          </button>
        </div>
      )}
    </div>
  );
};
