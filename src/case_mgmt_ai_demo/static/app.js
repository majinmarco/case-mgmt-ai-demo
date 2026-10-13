// A citation opens its Reglamento PDF in the side panel, on the cited page, without leaving
// the case page. Without this script the link still opens the PDF on that page.
(function () {
  const viewer = document.getElementById("visor");
  if (!viewer) return;
  const holder = document.getElementById("visor-documento");
  const title = document.getElementById("visor-cita");
  const newTab = document.getElementById("visor-pestana");

  function open(href, citation) {
    // A new frame each time: changing only #page=N of a PDF already shown does not move it.
    const frame = document.createElement("iframe");
    frame.src = href;
    frame.title = "Documento citado: " + citation;
    holder.replaceChildren(frame);
    title.textContent = citation;
    newTab.href = href;
    viewer.hidden = false;
    viewer.scrollIntoView({ block: "nearest" }); // on a narrow screen the panel is below the page
  }

  // Delegated, so citations in a Recommendation HTMX swaps in work too.
  document.addEventListener("click", function (event) {
    const link = event.target.closest("a.cita");
    if (!link || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    open(link.href, link.textContent.trim());
  });

  document.getElementById("visor-cerrar").addEventListener("click", function () {
    viewer.hidden = true;
    holder.replaceChildren();
  });
})();

// An invalid new Permit Request comes back with status 422 and its errors under each field:
// HTMX swaps that form in, which it does not do for an error status on its own.
document.addEventListener("htmx:beforeSwap", function (event) {
  if (event.detail.xhr.status === 422 && event.detail.target.id === "solicitud-nueva") {
    event.detail.shouldSwap = true;
    event.detail.isError = false;
  }
});
