document.getElementById("menuBtn").addEventListener("click", () => {
  const dropdown = document.getElementById("dropdownMenu");
  dropdown.style.display = dropdown.style.display === "flex" ? "none" : "flex";
});
