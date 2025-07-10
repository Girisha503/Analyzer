// ✅ Toggle nav menu button
document.addEventListener("DOMContentLoaded", () => {
  const menuBtn = document.getElementById("menuBtn");
  const dropdown = document.getElementById("dropdownMenu");

  menuBtn.addEventListener("click", () => {
    dropdown.style.display = dropdown.style.display === "flex" ? "none" : "flex";
  });

  // ✅ Elements for analyzing complaints
  const analyzeBtn = document.querySelector(".input-section .btn");
  const textarea = document.querySelector(".input-section textarea");
  const resultBox = document.querySelector("#analysis .result-box");
  const responseBox = document.querySelector("#draft .result-box");

  analyzeBtn.addEventListener("click", async () => {
    const userText = textarea.value.trim();
    if (!userText) {
      alert("Please enter a complaint first!");
      return;
    }

    // ✅ Clear previous results
    resultBox.textContent = "Analyzing...";
    responseBox.textContent = "Generating response...";

    // ✅ Call backend /summarize
    try {
      const res = await fetch("http://127.0.0.1:8000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      resultBox.textContent = data.success ? data.summary : `Error: ${data.error || "Failed to summarize."}`;

    } catch (err) {
      console.error(err);
      resultBox.textContent = "Error connecting to backend for summary.";
    }

    // ✅ Call backend /respond
    try {
      const res2 = await fetch("http://127.0.0.1:8000/respond", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });

      if (!res2.ok) {
        throw new Error(`Server error: ${res2.status}`);
      }

      const data2 = await res2.json();
      responseBox.textContent = data2.success ? data2.response : "Could not generate response.";

    } catch (err) {
      console.error(err);
      responseBox.textContent = "Error connecting to backend for response.";
    }
  });
});
