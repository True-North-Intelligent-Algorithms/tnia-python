/* Cookie functions from: https://www.w3schools.com/js/js_cookies.asp */

function setCookie(cname, cvalue, exdays) {
  var d = new Date();
  d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
  var expires = "expires="+d.toUTCString();
  document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/;sameSite=lax";
}

function getCookie(cname) {
  var name = cname + "=";
  var ca = document.cookie.split(';');
  for (var i=0; i<ca.length; i++) {
    var c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) == 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

function loadDockState() {
  var docks = getCookie('docks');
  if (!docks) return;
  var tokens = docks.split('~');
  for (var i=0; i<tokens.length; i++) {
    var dockId, dockableIds;
    [dockId, dockableList] = tokens[i].split(':', 2);
    if (!dockId || !dockableList) continue;
    var dock = document.getElementById(dockId);
    if (!dock) continue;
    var dockableIds = dockableList.split(',');
    for (var j=0; j<dockableIds.length; j++) {
      var dockable = document.getElementById(dockableIds[j]);
      if (!dockable) continue;
      dock.appendChild(dockable);
    }
  }
}

function saveDockState() {
  var tokens = [];
  var docks = document.querySelectorAll('.dock');
  docks.forEach(function(dock) {
    var token = [];
    dock.querySelectorAll('.dockable').forEach(function(dockable) {
      token.push(dockable.id);
    });
    tokens.push(dock.id + ':' + token.join(','));
  });
  setCookie('docks', tokens.join('~'), 36500);
}

function grab(el) {
  el.style.opacity = '0.4';
  el.setAttribute('draggable', 'true');
}

function ungrab(el) {
  el.setAttribute('draggable', 'false');
  el.style.opacity = '1.0';
}

function dockMouseDown(e) {
  grab(e.target.parentNode);
}

function dockMouseUp(e) {
  ungrab(e.target.parentNode);
}

function dockDragStart(e) {
  draggedElement = e.target;
  /* NB: Work around bug in Chrome; see https://stackoverflow.com/a/20733870. */
  setTimeout(function() {
    overlays.forEach(function(overlay) {
      overlay.style.display = 'block';
      overlay.classList.add('drag-active');
    });
  }, 10);
}

function dockDragEnd(e) {
  overlays.forEach(function(overlay) {
    overlay.classList.remove('drag-active');
    overlay.classList.remove('drag-over');
    overlay.style.display = 'none';
  });
  ungrab(draggedElement);
  draggedElement = null;
  saveDockState();
}

function dockDragEnter(e) {
  this.classList.add('drag-over');
}

function dockDragOver(e) {
  var dock = document.getElementById(e.target.getAttribute('data-dock-target'));
  var dockItems = dock.querySelectorAll('.dockable');
  var placed = false;
  for (var i=0; i<dockItems.length; i++) {
    var dockItem = dockItems[i];
    var y = dockItem.offsetTop + dockItem.offsetHeight / 2;
    if (e.layerY < y) {
      // insert dragged element just prior to this one.
      dock.insertBefore(draggedElement, dockItem);
      placed = true;
      break;
    }
  }
  if (!placed) dock.appendChild(draggedElement);
  if (e.preventDefault) e.preventDefault();
  return false;
}

function dockDragLeave(e) {
  this.classList.remove('drag-over');
}

loadDockState();

let draggedElement = null;

document.querySelectorAll('.dockable').forEach(function(dockable) {
  /* Adapted from: https://jsfiddle.net/a6tgy9so/1/ */
  var handle = dockable.querySelector('.drag-handle');
  handle.addEventListener('mousedown', dockMouseDown, false);
  handle.addEventListener('mouseup', dockMouseUp, false);

  dockable.addEventListener('dragstart', dockDragStart, false);
  dockable.addEventListener('dragend', dockDragEnd, false);
});

let overlays = document.querySelectorAll('.dock-overlay');
overlays.forEach(function(overlay) {
  overlay.addEventListener('dragenter', dockDragEnter, false);
  overlay.addEventListener('dragover', dockDragOver, false);
  overlay.addEventListener('dragleave', dockDragLeave, false);
});
