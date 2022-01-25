/* Credit:
 * - https://stackoverflow.com/a/987376/1207769
 * - https://www.w3schools.com/howto/howto_js_copy_clipboard.asp
 */
function selectAndCopy(element) {
  if (document.body.createTextRange) { //ms
    var range = document.body.createTextRange();
    range.moveToElementText(element);
    range.select();
  }
  else if (window.getSelection) { //all others
    var selection = window.getSelection();
    var range = document.createRange();
    range.selectNodeContents(element);
    selection.removeAllRanges();
    selection.addRange(range);
  }
  document.execCommand("copy");
}

// Splice in a "Copy" button next to each code block.
document.querySelectorAll("pre").forEach(function(pre) {
  if (pre.childElementCount != 1) return;
  var code = pre.children[0];
  if (code.nodeName != "CODE") return;

  var copy = document.createElement("A");
  copy.innerHTML = "Copy";
  copy.style.backgroundColor = '#f7f7f7';
  copy.style.border = '1px solid black';
  copy.style.borderRadius = '3px';
  copy.style.color = '#586069';
  copy.style.cursor = 'pointer';
  copy.style.display = 'none';
  copy.style.font = 'bold 1em monospace';
  copy.style.padding = '0.4rem';
  copy.style.position = 'absolute';
  copy.style.right = '5px';
  copy.style.textDecoration = 'none';
  copy.style.top = '5px';
  copy.onclick = function(e) { selectAndCopy(code); }

  pre.onmouseover = function(e) { copy.style.display = 'inline'; }
  pre.onmouseout = function(e) { copy.style.display = 'none'; }
  pre.style.position = 'relative';
  pre.insertBefore(copy, code);
});
