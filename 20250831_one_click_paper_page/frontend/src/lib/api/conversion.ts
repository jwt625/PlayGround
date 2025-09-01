/**
 * API client for backend conversion endpoints
 */

export interface ConversionJobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface ConversionStatusResponse {
  job_id: string;
  status: "queued" | "processing" | "completed" | "failed" | "cancelled";
  phase:
    | "queued"
    | "preparing"
    | "analyzing"
    | "converting"
    | "processing"
    | "finalizing"
    | "completed";
  stage: string;
  message: string;
  error?: string;
}

export interface PaperMetadata {
  title?: string;
  authors?: string[];
  abstract?: string;
  keywords?: string[];
  doi?: string;
  arxiv_id?: string;
}

export interface ConversionResult {
  job_id: string;
  status: string;
  output_files: string[];
  metrics: {
    total_conversion_time: number;
    mode_used: string;
    quality_assessment: {
      has_good_text: boolean;
      recommended_mode: string;
      confidence: string;
      avg_chars_per_page: number;
      text_coverage: number;
    };
    model_load_time?: number;
    processing_time?: number;
  };
  markdown_length: number;
  image_count: number;
  html_file: string;
  markdown_file: string;
  metadata?: PaperMetadata;
}

export type ConversionMode = "auto" | "fast" | "quality";

const API_BASE = "http://localhost:8000/api";

export class ConversionAPI {
  /**
   * Upload a file and start conversion
   */
  static async uploadAndConvert(
    file: File,
    template: string,
    mode: ConversionMode = "auto",
    repositoryName?: string
  ): Promise<ConversionJobResponse> {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("template", template);
    formData.append("mode", mode);
    if (repositoryName) {
      formData.append("repository_name", repositoryName);
    }

    const response = await fetch(`${API_BASE}/convert/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  /**
   * Get conversion status
   */
  static async getStatus(jobId: string): Promise<ConversionStatusResponse> {
    const response = await fetch(`${API_BASE}/convert/status/${jobId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get status");
    }

    return response.json();
  }

  /**
   * Get conversion result
   */
  static async getResult(jobId: string): Promise<ConversionResult> {
    const response = await fetch(`${API_BASE}/convert/result/${jobId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to get result");
    }

    return response.json();
  }

  /**
   * Cancel conversion
   */
  static async cancel(jobId: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE}/convert/cancel/${jobId}`, {
      method: "DELETE",
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to cancel");
    }

    return response.json();
  }

  /**
   * Poll conversion status until completion
   */
  static async pollStatus(
    jobId: string,
    onProgress?: (status: ConversionStatusResponse) => void,
    intervalMs: number = 2000
  ): Promise<ConversionResult> {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getStatus(jobId);

          if (onProgress) {
            onProgress(status);
          }

          if (status.status === "completed") {
            const result = await this.getResult(jobId);
            resolve(result);
          } else if (status.status === "failed") {
            reject(new Error(status.error || "Conversion failed"));
          } else if (status.status === "cancelled") {
            reject(new Error("Conversion was cancelled"));
          } else {
            // Continue polling
            setTimeout(poll, intervalMs);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }
}

/**
 * React hook for conversion operations
 */
export function useConversion() {
  const [isConverting, setIsConverting] = useState(false);
  const [phase, setPhase] = useState<string>("queued");
  const [stage, setStage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ConversionResult | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const startConversion = async (
    file: File,
    template: string,
    mode: ConversionMode = "auto",
    repositoryName?: string
  ) => {
    setIsConverting(true);
    setPhase("preparing");
    setStage("Uploading file...");
    setError(null);
    setResult(null);

    try {
      // Start conversion
      const job = await ConversionAPI.uploadAndConvert(
        file,
        template,
        mode,
        repositoryName
      );
      setJobId(job.job_id);

      // Poll for completion
      const result = await ConversionAPI.pollStatus(job.job_id, status => {
        setPhase(status.phase);
        setStage(status.message);
      });

      setResult(result);
      setStage("Conversion completed!");
      setPhase("completed");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Conversion failed");
    } finally {
      setIsConverting(false);
    }
  };

  const reset = () => {
    setIsConverting(false);
    setPhase("queued");
    setStage("");
    setError(null);
    setResult(null);
    setJobId(null);
  };

  return {
    isConverting,
    phase,
    stage,
    error,
    result,
    jobId,
    startConversion,
    reset,
  };
}

// Import React for the hook
import { useState } from "react";
