const Cite = require('citation-js')

function fetchCitation(element) {
  let doi = element.getAttribute('data-citation-id');
  let style = element.getAttribute('data-citation-style');
  if (doi == null || style == null) return;
  Cite.async(doi).then(function (citation) {
    element.innerHTML = citation.format('bibliography', {
      format: 'html',
      template: style,
      lang: 'en-US'
    }).replace(/(https:\/\/doi\.org\/([^<]*))/, '<a href="$1">doi:$2</a>');
  })
}

document.querySelectorAll(".citation").forEach(function(element) {
  fetchCitation(element);
});
