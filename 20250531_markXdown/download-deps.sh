#!/bin/bash

# Create lib directory if it doesn't exist
mkdir -p lib

# Download marked.js
curl -o lib/marked.min.js https://cdn.jsdelivr.net/npm/marked/marked.min.js

# Download MathJax
curl -o lib/mathjax.min.js https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js 