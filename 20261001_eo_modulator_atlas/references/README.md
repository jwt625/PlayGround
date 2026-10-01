Local source cache, tracked in-repo since 2026-10-01 (DevLog-000 decision 5 revised; includes publisher-copyright PDFs for off-repo backup). Layout per source:

    references/<paper_id>/
      source.pdf | source.html      raw download
      source.json                   url, doi, sha256, retrieved_on (ISO), license, http status
      text.md                       extracted text, page-delimited ("<!-- page N -->"), metadata header
      figures/page_NN.png           150 dpi render of every page that has a figure caption
      figures/img_pNN_k.png         embedded raster images (>= 150 px)
      figures/figures.json          page/caption index
