/**
 * File upload and processing utilities for GitHub integration
 */

import type { FileTypeDetection, SupportedFileType, FileUpload } from '../../types/github';

export class FileProcessor {
  /**
   * Detect file type from uploaded file
   */
  static detectFileType(file: File): FileTypeDetection {
    const extension = file.name.toLowerCase().split('.').pop() || '';
    const mimeType = file.type.toLowerCase();

    // PDF detection
    if (extension === 'pdf' || mimeType === 'application/pdf') {
      return {
        type: 'pdf',
        confidence: 1.0,
        file,
      };
    }

    // DOCX detection
    if (
      extension === 'docx' ||
      mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ) {
      return {
        type: 'docx',
        confidence: 1.0,
        file,
      };
    }

    // LaTeX detection
    if (extension === 'tex' || extension === 'latex') {
      return {
        type: 'latex',
        confidence: 1.0,
        file,
      };
    }

    // ZIP detection (could be LaTeX project)
    if (extension === 'zip' || mimeType === 'application/zip') {
      return {
        type: 'zip',
        confidence: 0.8, // Lower confidence as it could be anything
        file,
      };
    }

    // Default to unknown
    return {
      type: 'pdf', // Default fallback
      confidence: 0.1,
      file,
    };
  }

  /**
   * Validate Overleaf URL
   */
  static validateOverleafUrl(url: string): FileTypeDetection {
    const overleafPattern = /^https:\/\/(www\.)?overleaf\.com\/project\/[a-f0-9]+/;
    const gitPattern = /^https:\/\/git\.overleaf\.com\/[a-f0-9]+/;

    if (overleafPattern.test(url) || gitPattern.test(url)) {
      return {
        type: 'overleaf',
        confidence: 1.0,
        overleaf_url: url,
      };
    }

    return {
      type: 'overleaf',
      confidence: 0.0,
      overleaf_url: url,
    };
  }

  /**
   * Convert file to base64 string
   */
  static async fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        // Remove data URL prefix (e.g., "data:application/pdf;base64,")
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  /**
   * Convert file to UTF-8 text (for text files)
   */
  static async fileToText(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsText(file, 'utf-8');
    });
  }

  /**
   * Prepare file for GitHub upload
   */
  static async prepareFileForUpload(
    file: File,
    targetPath: string
  ): Promise<FileUpload> {
    const detection = this.detectFileType(file);

    // For text files (LaTeX), upload as UTF-8
    if (detection.type === 'latex') {
      const content = await this.fileToText(file);
      return {
        path: targetPath,
        content,
        encoding: 'utf-8',
      };
    }

    // For binary files (PDF, DOCX, ZIP), upload as base64
    const content = await this.fileToBase64(file);
    return {
      path: targetPath,
      content,
      encoding: 'base64',
    };
  }

  /**
   * Generate appropriate filename for uploaded file
   */
  static generateTargetPath(file: File, detection: FileTypeDetection): string {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const extension = file.name.toLowerCase().split('.').pop() || 'bin';
    
    switch (detection.type) {
      case 'pdf':
        return `paper.pdf`;
      case 'docx':
        return `paper.docx`;
      case 'latex':
        return `main.tex`;
      case 'zip':
        return `paper-source.zip`;
      default:
        return `uploaded-${timestamp}.${extension}`;
    }
  }

  /**
   * Validate file size and type
   */
  static validateFile(file: File): { valid: boolean; error?: string } {
    const maxSize = 50 * 1024 * 1024; // 50MB limit
    const allowedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/zip',
      'text/plain',
      'text/x-tex',
      'application/x-tex',
    ];

    if (file.size > maxSize) {
      return {
        valid: false,
        error: 'File size exceeds 50MB limit',
      };
    }

    const detection = this.detectFileType(file);
    if (detection.confidence < 0.5) {
      return {
        valid: false,
        error: 'Unsupported file type. Please upload PDF, DOCX, LaTeX, or ZIP files.',
      };
    }

    return { valid: true };
  }
}

/**
 * File upload progress tracking
 */
export class UploadProgress {
  private callbacks: Array<(progress: number) => void> = [];

  onProgress(callback: (progress: number) => void): void {
    this.callbacks.push(callback);
  }

  updateProgress(progress: number): void {
    this.callbacks.forEach(callback => callback(progress));
  }

  reset(): void {
    this.callbacks = [];
  }
}

/**
 * Batch file upload utility
 */
export class BatchFileUploader {
  private files: File[] = [];
  private progress = new UploadProgress();

  addFile(file: File): void {
    const validation = FileProcessor.validateFile(file);
    if (!validation.valid) {
      throw new Error(validation.error);
    }
    this.files.push(file);
  }

  addFiles(files: FileList | File[]): void {
    const fileArray = Array.from(files);
    fileArray.forEach(file => this.addFile(file));
  }

  async prepareForUpload(): Promise<FileUpload[]> {
    const uploads: FileUpload[] = [];
    
    for (let i = 0; i < this.files.length; i++) {
      const file = this.files[i];
      const detection = FileProcessor.detectFileType(file);
      const targetPath = FileProcessor.generateTargetPath(file, detection);
      
      const upload = await FileProcessor.prepareFileForUpload(file, targetPath);
      uploads.push(upload);
      
      // Update progress
      const progress = ((i + 1) / this.files.length) * 100;
      this.progress.updateProgress(progress);
    }

    return uploads;
  }

  getProgress(): UploadProgress {
    return this.progress;
  }

  clear(): void {
    this.files = [];
    this.progress.reset();
  }

  getFileCount(): number {
    return this.files.length;
  }

  getFiles(): File[] {
    return [...this.files];
  }
}

/**
 * Drag and drop file handler
 */
export class DragDropHandler {
  private element: HTMLElement;
  private onFilesDropped: (files: FileList) => void;

  constructor(element: HTMLElement, onFilesDropped: (files: FileList) => void) {
    this.element = element;
    this.onFilesDropped = onFilesDropped;
    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    this.element.addEventListener('dragover', this.handleDragOver.bind(this));
    this.element.addEventListener('dragenter', this.handleDragEnter.bind(this));
    this.element.addEventListener('dragleave', this.handleDragLeave.bind(this));
    this.element.addEventListener('drop', this.handleDrop.bind(this));
  }

  private handleDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.element.classList.add('drag-over');
  }

  private handleDragEnter(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.element.classList.add('drag-enter');
  }

  private handleDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.element.classList.remove('drag-over', 'drag-enter');
  }

  private handleDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    
    this.element.classList.remove('drag-over', 'drag-enter');
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.onFilesDropped(files);
    }
  }

  destroy(): void {
    this.element.removeEventListener('dragover', this.handleDragOver);
    this.element.removeEventListener('dragenter', this.handleDragEnter);
    this.element.removeEventListener('dragleave', this.handleDragLeave);
    this.element.removeEventListener('drop', this.handleDrop);
  }
}
