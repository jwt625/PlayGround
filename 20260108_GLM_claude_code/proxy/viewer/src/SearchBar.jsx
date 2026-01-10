function SearchBar({ searchQuery, onSearchChange, searchType, onSearchTypeChange, searchFields, onSearchFieldsChange }) {
  const handleFieldChange = (field) => {
    if (searchFields.includes(field)) {
      onSearchFieldsChange(searchFields.filter(f => f !== field))
    } else {
      onSearchFieldsChange([...searchFields, field])
    }
  }

  const fieldOptions = [
    { value: 'user_message', label: 'User Message' },
    { value: 'assistant_response', label: 'Assistant Response' },
    { value: 'request_body', label: 'Request Body (JSON)' },
    { value: 'response_body', label: 'Response Body (JSON)' },
    { value: 'tools', label: 'Tool Names' }
  ]

  return (
    <div className="search-bar">
      <div className="search-input-group">
        <input
          type="text"
          placeholder="Search logs..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="search-input"
        />
        <select
          value={searchType}
          onChange={(e) => onSearchTypeChange(e.target.value)}
          className="search-type-select"
        >
          <option value="all">Text</option>
          <option value="regex">Regex</option>
        </select>
        <button
          onClick={() => onSearchChange('')}
          className="clear-search-btn"
          title="Clear search"
          disabled={!searchQuery}
        >
          Clear
        </button>
      </div>

      <div className="search-fields">
        <span className="search-fields-label">Search in:</span>
        {fieldOptions.map(field => (
          <label key={field.value} className="checkbox-field">
            <input
              type="checkbox"
              value={field.value}
              checked={searchFields.includes(field.value)}
              onChange={() => handleFieldChange(field.value)}
            />
            {field.label}
          </label>
        ))}
      </div>
    </div>
  )
}

export default SearchBar