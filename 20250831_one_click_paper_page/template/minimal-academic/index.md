---
layout: default
title: "Academic Paper"
---

# {{ page.title }}

<div class="paper-meta">
  <p class="authors">{{ site.author }}</p>
  <p class="date">{{ site.time | date: "%B %Y" }}</p>
</div>

## Abstract

This is where the paper abstract will be displayed. The content will be automatically generated from the converted PDF.

## Paper Content

The main content of the paper will be inserted here during the deployment process.

---

<div class="paper-actions">
  <a href="paper.pdf" class="btn btn-primary" download>Download PDF</a>
  <a href="#citation" class="btn btn-secondary">Cite This Paper</a>
</div>

## Citation

```bibtex
@article{paper2024,
  title={{{ page.title }}},
  author={{{ site.author }}},
  year={2024},
  url={{{ site.url }}{{ site.baseurl }}}
}
```

<style>
.paper-meta {
  text-align: center;
  margin: 2rem 0;
  padding: 1rem;
  background-color: #f8f9fa;
  border-radius: 8px;
}

.paper-meta .authors {
  font-size: 1.2em;
  font-weight: 500;
  margin-bottom: 0.5rem;
}

.paper-meta .date {
  color: #6c757d;
  margin-bottom: 0;
}

.paper-actions {
  text-align: center;
  margin: 2rem 0;
}

.btn {
  display: inline-block;
  padding: 0.5rem 1rem;
  margin: 0 0.5rem;
  text-decoration: none;
  border-radius: 4px;
  font-weight: 500;
  transition: background-color 0.2s;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn-primary:hover {
  background-color: #0056b3;
  color: white;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-secondary:hover {
  background-color: #545b62;
  color: white;
}

/* Math formula styling */
.MathJax {
  font-size: 1.1em !important;
}

/* Table styling */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0;
}

table th,
table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: left;
}

table th {
  background-color: #f2f2f2;
  font-weight: bold;
}

/* Image styling */
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1rem auto;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Code block styling */
pre {
  background-color: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 4px;
  padding: 1rem;
  overflow-x: auto;
}

code {
  background-color: #f8f9fa;
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.9em;
}

pre code {
  background-color: transparent;
  padding: 0;
}
</style>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js">
</script>

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    tags: 'ams'
  },
  options: {
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  }
};
</script>
