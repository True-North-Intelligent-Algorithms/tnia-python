function searchInput() {
  return document.getElementById("search-box").querySelector("input");
}
function searchClear() {
  return document.getElementById("search-box").querySelectorAll("button")[1];
}
function searchResults() {
  return document.getElementById("search-results");
}
function searchHitItems() {
  return document.getElementsByClassName("search-result");
}
function searchPageLinks() {
  return document.getElementsByClassName("ais-Pagination-link");
}

function refreshSearchResultsVisibility() {
  var displayMode = searchInput().value == '' ? 'none' : 'block';
  searchResults().style.display = displayMode;
}

function focusSearchBar() {
  searchInput().focus();
  refreshSearchResultsVisibility();
}

function isSearchActive() {
  return searchResults().style.display == 'block' ||
         document.activeElement == searchInput();
}

function clearSearch() {
  searchInput().value = '';
  refreshSearchResultsVisibility();
}

var selectedHit = -1;

function visitSelected() {
  var items = searchHitItems();
  var hit = Math.max(0, selectedHit);
  var i = Math.min(hit, items.length - 1);
  window.location = items[i].parentElement.href;
}

function selectHit(step) {
  var items = searchHitItems();
  var oldIndex = selectedHit;

  // move the selected hit
  selectedHit += step;
  if (selectedHit < -1) {
    // trying to move up from the search box
    selectedHit = -1;
  }
  else if (selectedHit < 0) {
    // moved off the list, back to the search box
    focusSearchBar();
  }
  else if (selectedHit >= items.length) {
    // trying to move off the bottom of the list
    selectedHit = items.length - 1;
  }

  if (oldIndex == selectedHit) return;

  if (oldIndex >= 0 && oldIndex < items.length) {
    // deselect previously selected hit
    items[oldIndex].style.background = null;
  }
  if (selectedHit >= 0 && selectedHit < items.length) {
    // highlight the new selected hit
    items[selectedHit].style.background = '#7dd';
    items[selectedHit].focus();
  }
}

function selectPage(buttonIndex) {
  var pageButtons = searchPageLinks();
  if (buttonIndex < 0) {
    // count negative indices from the right
    buttonIndex += pageButtons.length;
  }
  pageButtons[buttonIndex].click();
  selectedHit = -1;
}

searchInput().oninput = refreshSearchResultsVisibility;
searchInput().onkeydown = function(e) {
  if (e.keyCode == 27) clearSearch(); // escape
  else if (e.keyCode == 13) visitSelected(); // enter
};

searchClear().onkeydown = function(e) {
  if (e.keyCode == 32) clearSearch(); // space bar
};
searchClear().onclick = function(e) { clearSearch(); }

document.addEventListener("keydown", function(e) {
  if (isSearchActive()) {
    if (e.keyCode == 40) selectHit(1); // down arrow
    else if (e.keyCode == 38) selectHit(-1); // up arrow
    else if (e.keyCode == 34) selectPage(-2); // page down: page + 1
    else if (e.keyCode == 33) selectPage(1); // page up: page - 1
    else if (e.keyCode == 35) selectPage(-1); // end: last page
    else if (e.keyCode == 36) selectPage(0); // home: first page
    else return;
  }
  else {
    if (e.keyCode == 76 && !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey) {
      // NB: Only do this without modifiers, so that e.g. ctrl+L still works.
      focusSearchBar(); // L
    }
    else return;
  }
  e.preventDefault();
}, false);
