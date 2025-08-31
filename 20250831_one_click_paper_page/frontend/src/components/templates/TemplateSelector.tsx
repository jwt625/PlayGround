import React, { useState } from 'react';
import type { PaperTemplate } from '../../types/github';
import { AVAILABLE_TEMPLATES } from '../../types/github';

interface TemplateSelectorProps {
  onTemplateSelected: (template: PaperTemplate) => void;
  selectedTemplate?: PaperTemplate;
}

export const TemplateSelector: React.FC<TemplateSelectorProps> = ({
  onTemplateSelected,
  selectedTemplate,
}) => {
  const [hoveredTemplate, setHoveredTemplate] = useState<string | null>(null);

  const handleTemplateClick = (template: PaperTemplate) => {
    onTemplateSelected(template);
  };

  const getTemplateIcon = (templateId: string) => {
    switch (templateId) {
      case 'academic-pages':
        return '🎓';
      case 'academic-project-page':
        return '📊';
      case 'al-folio':
        return '✨';
      default:
        return '📄';
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800 mb-2">Choose Your Template</h2>
        <p className="text-gray-600">
          Select a template that best fits your academic paper presentation needs
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {AVAILABLE_TEMPLATES.map((template) => {
          const isSelected = selectedTemplate?.id === template.id;
          const isHovered = hoveredTemplate === template.id;

          return (
            <div
              key={template.id}
              className={`relative bg-white rounded-lg shadow-lg border-2 transition-all duration-200 cursor-pointer ${
                isSelected
                  ? 'border-blue-500 ring-2 ring-blue-200'
                  : isHovered
                  ? 'border-gray-300 shadow-xl'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => handleTemplateClick(template)}
              onMouseEnter={() => setHoveredTemplate(template.id)}
              onMouseLeave={() => setHoveredTemplate(null)}
            >
              {/* Selection Indicator */}
              {isSelected && (
                <div className="absolute -top-2 -right-2 bg-blue-500 text-white rounded-full w-6 h-6 flex items-center justify-center">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
              )}

              <div className="p-6">
                {/* Template Header */}
                <div className="flex items-center mb-4">
                  <span className="text-3xl mr-3">{getTemplateIcon(template.id)}</span>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800">
                      {template.name}
                    </h3>
                    <p className="text-sm text-gray-500">
                      {template.id === 'academic-pages' && 'Jekyll-based'}
                      {template.id === 'academic-project-page' && 'JavaScript-based'}
                      {template.id === 'al-folio' && 'Jekyll-based'}
                    </p>
                  </div>
                </div>

                {/* Template Description */}
                <p className="text-gray-600 mb-4 text-sm leading-relaxed">
                  {template.description}
                </p>

                {/* Features List */}
                <div className="mb-4">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Features:</h4>
                  <ul className="space-y-1">
                    {template.features.slice(0, 3).map((feature, index) => (
                      <li key={index} className="flex items-center text-xs text-gray-600">
                        <svg
                          className="w-3 h-3 text-green-500 mr-2 flex-shrink-0"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        {feature}
                      </li>
                    ))}
                    {template.features.length > 3 && (
                      <li className="text-xs text-gray-500 ml-5">
                        +{template.features.length - 3} more features
                      </li>
                    )}
                  </ul>
                </div>

                {/* Template Actions */}
                <div className="flex items-center justify-between pt-4 border-t border-gray-100">
                  <a
                    href={template.repository_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                    onClick={(e) => e.stopPropagation()}
                  >
                    View Source →
                  </a>
                  {template.preview_url && (
                    <a
                      href={template.preview_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-gray-600 hover:text-gray-700 text-sm"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Preview
                    </a>
                  )}
                </div>
              </div>

              {/* Hover Overlay */}
              {isHovered && !isSelected && (
                <div className="absolute inset-0 bg-blue-50 bg-opacity-50 rounded-lg flex items-center justify-center">
                  <div className="bg-white px-4 py-2 rounded-md shadow-sm">
                    <span className="text-blue-600 font-medium">Click to Select</span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Selected Template Summary */}
      {selectedTemplate && (
        <div className="mt-8 p-6 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex items-center mb-3">
            <span className="text-2xl mr-3">{getTemplateIcon(selectedTemplate.id)}</span>
            <div>
              <h3 className="text-lg font-semibold text-blue-800">
                Selected: {selectedTemplate.name}
              </h3>
              <p className="text-blue-600 text-sm">{selectedTemplate.description}</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
            <div>
              <h4 className="font-medium text-blue-800 mb-2">All Features:</h4>
              <ul className="space-y-1">
                {selectedTemplate.features.map((feature, index) => (
                  <li key={index} className="flex items-center text-sm text-blue-700">
                    <svg
                      className="w-3 h-3 text-blue-500 mr-2 flex-shrink-0"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-blue-800 mb-2">Template Info:</h4>
              <div className="space-y-2 text-sm text-blue-700">
                <p>
                  <span className="font-medium">Type:</span>{' '}
                  {selectedTemplate.id === 'academic-pages' && 'Full Academic Site'}
                  {selectedTemplate.id === 'academic-project-page' && 'Project Showcase'}
                  {selectedTemplate.id === 'al-folio' && 'Minimal Portfolio'}
                </p>
                <p>
                  <span className="font-medium">Technology:</span>{' '}
                  {selectedTemplate.id.includes('academic-pages') || selectedTemplate.id.includes('al-folio') 
                    ? 'Jekyll (Ruby)' 
                    : 'JavaScript/HTML'}
                </p>
                <a
                  href={selectedTemplate.repository_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center text-blue-600 hover:text-blue-700 font-medium"
                >
                  View Documentation
                  <svg className="w-3 h-3 ml-1" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
