import React from 'react';
import type { ConversionResult } from '../../lib/api/conversion';

interface ConversionProgressProps {
  isConverting: boolean;
  progress: number;
  stage: string;
  error: string | null;
  result: ConversionResult | null;
  onCancel?: () => void;
  onRetry?: () => void;
  onContinue?: (result: ConversionResult) => void;
}

export const ConversionProgress: React.FC<ConversionProgressProps> = ({
  isConverting,
  progress,
  stage,
  error,
  result,
  onCancel,
  onRetry,
  onContinue,
}) => {
  if (!isConverting && !error && !result) {
    return null;
  }

  return (
    <div className="conversion-progress">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-md mx-auto">
        {/* Header */}
        <div className="text-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            {isConverting ? 'Converting Document' : error ? 'Conversion Failed' : 'Conversion Complete'}
          </h3>
        </div>

        {/* Progress Bar */}
        {isConverting && (
          <div className="mb-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Progress</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Status Message */}
        <div className="mb-4">
          <p className="text-sm text-gray-600 text-center">
            {stage || 'Processing...'}
          </p>
        </div>

        {/* Loading Spinner */}
        {isConverting && (
          <div className="flex justify-center mb-4">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="mb-4">
            <div className="bg-red-50 border border-red-200 rounded-md p-3">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-red-800">{error}</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Success State */}
        {result && (
          <div className="mb-4">
            <div className="bg-green-50 border border-green-200 rounded-md p-3">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-green-800">
                    Document converted successfully!
                  </p>
                  <div className="mt-2 text-xs text-green-700">
                    <p>• Conversion time: {result.metrics.total_conversion_time.toFixed(1)}s</p>
                    <p>• Mode used: {result.metrics.mode_used}</p>
                    <p>• Content length: {result.markdown_length.toLocaleString()} characters</p>
                    <p>• Images extracted: {result.image_count}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex gap-3 justify-center">
          {isConverting && onCancel && (
            <button
              onClick={onCancel}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Cancel
            </button>
          )}

          {error && onRetry && (
            <button
              onClick={onRetry}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Try Again
            </button>
          )}

          {result && onContinue && (
            <button
              onClick={() => onContinue(result)}
              className="px-4 py-2 text-sm font-medium text-white bg-green-600 border border-transparent rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
            >
              Continue to Deployment
            </button>
          )}
        </div>

        {/* Conversion Details (for completed conversions) */}
        {result && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <details className="text-sm">
              <summary className="cursor-pointer text-gray-600 hover:text-gray-900">
                View conversion details
              </summary>
              <div className="mt-2 space-y-1 text-xs text-gray-500">
                <p><strong>Job ID:</strong> {result.job_id}</p>
                <p><strong>Output files:</strong> {result.output_files.length}</p>
                <p><strong>HTML file:</strong> {result.html_file}</p>
                <p><strong>Markdown file:</strong> {result.markdown_file}</p>
                <p><strong>Quality assessment:</strong></p>
                <ul className="ml-4 space-y-1">
                  <li>• Good text: {result.metrics.quality_assessment.has_good_text ? 'Yes' : 'No'}</li>
                  <li>• Confidence: {result.metrics.quality_assessment.confidence}</li>
                  <li>• Text coverage: {(result.metrics.quality_assessment.text_coverage * 100).toFixed(1)}%</li>
                  <li>• Avg chars/page: {result.metrics.quality_assessment.avg_chars_per_page.toFixed(0)}</li>
                </ul>
              </div>
            </details>
          </div>
        )}
      </div>
    </div>
  );
};
