const API_BASE = "http://127.0.0.1:8000";

const emailText = document.getElementById("emailText");
const btnCheck = document.getElementById("btnCheck");
const btnHam = document.getElementById("btnHam");
const btnSpam = document.getElementById("btnSpam");
const resultLine = document.getElementById("resultLine");
const confBar = document.getElementById("confBar");
const errBox = document.getElementById("errBox");

function setError(msg = "") {
  errBox.textContent = msg;
}

function setLoading(btn, isLoading, loadingText, normalText) {
  btn.disabled = isLoading;
  btn.textContent = isLoading ? loadingText : normalText;
}

function resetResultUI() {
  resultLine.textContent = "Result: —";
  resultLine.style.color = "#e7eefc";
  confBar.style.width = "0%";
  confBar.style.backgroundColor = "#AB0B4B";
}

function setResult(label, spamProb) {
  const pct = Math.round(spamProb * 10000) / 100; // 2dp
  const isSpam = label.toLowerCase() === "spam";

  resultLine.innerHTML = `Result: <strong>${label.toUpperCase()}</strong> — ${pct}% spam confidence`;

  // Green for HAM, Red/Wine for SPAM
  resultLine.style.color = isSpam ? "#AB0B4B" : "#2ecc71";
  confBar.style.backgroundColor = isSpam ? "#AB0B4B" : "#2ecc71";

  confBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

async function fetchSample(type) {
  setError("");
  const targetBtn = type === "ham" ? btnHam : btnSpam;

  setLoading(targetBtn, true, "Loading...", type === "ham" ? "Random HAM" : "Random SPAM");

  try {
    const res = await fetch(`${API_BASE}/sample?label=${type}`);
    const data = await res.json();

    if (!res.ok) {
      setError(data.error || "Failed to fetch sample.");
      return;
    }

    emailText.value = data.text || "";
    resetResultUI();
  } catch (e) {
    setError("Backend not reachable. Start it on http://127.0.0.1:8000");
  } finally {
    setLoading(btnHam, false, "", "Random HAM");
    setLoading(btnSpam, false, "", "Random SPAM");
  }
}

btnHam.addEventListener("click", () => fetchSample("ham"));
btnSpam.addEventListener("click", () => fetchSample("spam"));

btnCheck.addEventListener("click", async () => {
  setError("");
  const text = (emailText.value || "").trim();
  if (!text) return setError("Paste an email first.");

  setLoading(btnCheck, true, "Checking...", "Check");

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    if (!res.ok) return setError(data.error || "Prediction failed.");

    setResult(data.label, data.spam_probability);
  } catch (e) {
    setError("Backend not reachable. Start it on http://127.0.0.1:8000");
  } finally {
    setLoading(btnCheck, false, "", "Check");
  }
});