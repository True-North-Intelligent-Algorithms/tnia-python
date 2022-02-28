---
layout: basic
---

## Converting notebooks to markdown

Can convert with jekyllnb example useage below

```
jupyter jekyllnb --site-dir . --page-dir notebooks_ --image-dir images notebooks/Deconvolution/extract_psf.ipynb
```

Note:  In the notebook need to go to edit->Edit Notebook Metadata and add the following if you want frontmatter in the markdown

```
 "jekyll": {
    "layout": "$layout_name$",
    "title": "Title"
  }
```

## Add to menu

Go to '_includes/menu' and add the new notebook to the Notebooks section menu