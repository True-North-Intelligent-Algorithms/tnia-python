## How to add a new documentation page and show it in this menu

The TNIA web page has built by re-using code from the ImageJ wiki [from here](https://github.com/imagej/imagej.github.io)

Instructions on how to add a page and show it on the sidebar menu.

1.  Optionally add a new folder under the 'docs' folder (or use existing folder)
2.  Add a markdown file with the new documentation.
3.  Next modify the file ```_includes/menu``` to add a link to the new page.  Note ```menu``` has NO extension (this can be confusing if grepping with *.* as the file type, as it will not find the ```menu``` file) 