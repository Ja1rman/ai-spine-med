"use strict";

// The header does not need the full RSNA index to show its count.
void (async () => {
  try {
    const response = await fetch("data/nav-counts.json", {
      cache: "no-store",
      credentials: "same-origin",
      referrerPolicy: "no-referrer",
    });
    if (!response.ok) return;
    const { cohort } = await response.json();
    if (Number.isSafeInteger(cohort) && cohort >= 0) {
      document.getElementById("cohort-total").textContent = String(cohort);
    }
  } catch {
    // Older exports can omit the count and still provide working navigation.
  }
})();
