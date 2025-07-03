function updateBodyLength() {
  var length = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--variable-collection-length'));
  var Ln = parseFloat(document.getElementById('lnInput').value);
  if (isNaN(length) || isNaN(Ln)) {
      document.getElementById('lengthInput').value = '';
  } else {
      var bodyLength = length - Ln;
      document.getElementById('lengthInput').value = bodyLength.toFixed(2);
  }
}

document.getElementById('lnInput').addEventListener('input', updateBodyLength);
updateBodyLength();

const dropdownBtn = document.getElementById("dropdownBtn");
const dropdownList = document.getElementById("dropdownList");
const body = document.body;

dropdownBtn.addEventListener("click", () => {
  dropdownList.style.display = dropdownList.style.display === "block" ? "none" : "block";
});

dropdownList.querySelectorAll("a").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const mode = item.dataset.mode;
    body.setAttribute("data-variable-collection-mode", mode);
    dropdownBtn.textContent = `Model: ${mode}`;
    dropdownList.style.display = "none";
    updateDisplayedValues();
    updateFieldLocking(mode);
  });
});

document.addEventListener("click", e => {
  if (!e.target.closest(".dropdown")) dropdownList.style.display = "none";
});

function updateDisplayedValues() {
  const cssVars = getComputedStyle(body);
  const getVar = name => parseFloat(cssVars.getPropertyValue(name))?.toFixed(2) || "0.00";

  const values = {
    length: getVar("--variable-collection-length"),
    diameter: getVar("--variable-collection-diameter"),
    LW: getVar("--variable-collection-LW"),
    LF: getVar("--variable-collection-LF"),
    CTW: getVar("--variable-collection-CTW"),
    CRW: getVar("--variable-collection-CRW"),
    CTF: getVar("--variable-collection-CTF"),
    CRF: getVar("--variable-collection-CRF"),
    BW: getVar("--variable-collection-BW"),
    BF: getVar("--variable-collection-BF"),
  };

  document.querySelectorAll(".text-wrapper-3")[0].textContent = values.length;
  document.querySelectorAll(".text-wrapper-3")[1].textContent = values.LF;
  document.querySelectorAll(".text-wrapper-3")[2].textContent = values.LW;
  document.querySelector(".text-wrapper-6").textContent = values.diameter;
  document.querySelectorAll(".text-wrapper-8")[0].textContent = values.CTW;
  document.querySelectorAll(".text-wrapper-6")[1].textContent = values.CRW;
  document.querySelectorAll(".text-wrapper-8")[1].textContent = values.CTF;
  document.querySelectorAll(".text-wrapper-8")[2].textContent = values.CRF;
  document.querySelectorAll(".text-wrapper-8")[3].textContent = values.BW;
  document.querySelectorAll(".text-wrapper-8")[4].textContent = values.BF;
}

updateDisplayedValues();

document.querySelector('.overlap-3').addEventListener('click', async () => {
  try {
    const mach = parseFloat(document.getElementById('machInput').value);
    const aoa = parseFloat(document.getElementById('aoaInput').value);
    const ln = parseFloat(document.getElementById('lnInput').value);
    const swept = parseFloat(document.getElementById('sweptInput').value);
    const lln = parseFloat(document.getElementById('lengthInput').value);
    const mode = document.getElementById('dropdownBtn').textContent.split(": ")[1];

    if (isNaN(mach) || isNaN(aoa) || isNaN(ln)|| isNaN(swept)|| isNaN(lln)) {
      alert("Vui lòng nhập đủ cả 5 thông số!");
      return;
    }

    const response = await fetch("https://fern-elegant-dentist.glitch.me/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({mode, mach, aoa, ln, swept, lln})
    });

    const data = await response.json();

    if (!response.ok) {
      alert(data.error || "Lỗi không xác định từ server");
      return;
    }

    document.querySelector(".text-wrapper-18").textContent = data.cl;
    document.querySelector(".text-wrapper-16").textContent = data.cd;

    if (data.warning) {
      alert(data.warning);
    }
  } catch (err) {
    console.error("Lỗi:", err);
    alert("Không kết nối được với server: " + err.message);
  }
});

function updateFieldLocking(mode) {
  const lnInput = document.getElementById("lnInput");
  const sweptInput = document.getElementById("sweptInput");

  const isNASA = mode === "NASA";

  lnInput.readOnly = !isNASA;
  sweptInput.readOnly = !isNASA;

  if (!isNASA) {
    machInput.value = 0;
    aoaInput.value = 0;
    lnInput.value = 19.32;
    sweptInput.value = 16.54;
  }
}
