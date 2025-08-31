import React from 'react';
import type { ConversionMode } from '../../lib/api/conversion';

interface ConversionModeSelectorProps {
  mode: ConversionMode;
  onModeChange: (mode: ConversionMode) => void;
}

export const ConversionModeSelector: React.FC<ConversionModeSelectorProps> = ({
  mode,
  onModeChange,
}) => {
  const modes: Array<{
    id: ConversionMode;
    name: string;
    description: string;
    time: string;
    icon: string;
  }> = [
    {
      id: 'auto',
      name: 'Smart Mode',
      description: 'Automatically chooses the best conversion method',
      time: '~40 seconds',
      icon: '🤖',
    },
    {
      id: 'fast',
      name: 'Fast Mode',
      description: 'Quick conversion for digital PDFs with good text',
      time: '~40 seconds',
      icon: '⚡',
    },
    {
      id: 'quality',
      name: 'Quality Mode',
      description: 'Full OCR processing for scanned documents',
      time: '~6 minutes',
      icon: '🔍',
    },
  ];

  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-3">Conversion Mode</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {modes.map((modeOption) => (
          <button
            key={modeOption.id}
            onClick={() => onModeChange(modeOption.id)}
            className={`p-4 rounded-lg border-2 text-left transition-all duration-200 ${
              mode === modeOption.id
                ? 'border-blue-500 bg-blue-50 text-blue-900'
                : 'border-gray-200 bg-white text-gray-700 hover:border-gray-300 hover:bg-gray-50'
            }`}
          >
            <div className="flex items-center mb-2">
              <span className="text-2xl mr-2">{modeOption.icon}</span>
              <span className="font-medium">{modeOption.name}</span>
            </div>
            <p className="text-sm text-gray-600 mb-1">{modeOption.description}</p>
            <p className="text-xs text-gray-500">{modeOption.time}</p>
          </button>
        ))}
      </div>
    </div>
  );
};
