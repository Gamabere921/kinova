const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const loader = document.getElementById("loader");
const resultDiv = document.getElementById("result");
const top1 = document.getElementById("top1");
const materialMenu = document.getElementById("materialMenu");
const materialInfo = document.getElementById("materialInfo");

let chart = null;

const MATERIAL_INFO = {
  glass: { title: "Glass", desc: "Recyclable multiple times without losing quality.", tips: "Separate by color, avoid ceramics." },
  paper: { title: "Paper", desc: "Cellulose-based, recyclable but degrades.", tips: "Avoid greasy or wet paper." },
  cardboard: { title: "Cardboard", desc: "Widely recycled in packaging.", tips: "Keep dry and flattened." },
  plastic: { title: "Plastic", desc: "Complex recycling depending on polymer.", tips: "Separate PET from others." },
  metal: { title: "Metal", desc: "Efficient recycling, no property loss.", tips: "Rinse cans before recycling." },
  trash: { title: "Trash", desc: "Non-recyclable waste.", tips: "Reduce as much as possible." }
};

// Sidebar click
materialMenu.addEventListener("click", e => {
  if (e.target.tagName === "LI") {
    const mat = e.target.getAttribute("data-material");
    const info = MATERIAL_INFO[mat];
    if (info) {
      materialInfo.innerHTML = `
        <h3>${info.title}</h3>
        <p>${info.desc}</p>
        <p><strong>Tips:</strong> ${info.tips}</p>
      `;
      materialInfo.classList.add("fade-in");
      setTimeout(() => materialInfo.classList.remove("fade-in"), 600);
    }
  }
});

// Preview de imagen
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }
});

// Predict
predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  loader.classList.remove("hidden");
  resultDiv.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    const data = await res.json();

    loader.classList.add("hidden");
    resultDiv.classList.remove("hidden");

    top1.innerHTML = `Top prediction: <strong>${data.top1.class}</strong> (${(data.top1.prob*100).toFixed(2)}%)`;

    // Gráfico
    const labels = Object.keys(data.all_probs);
    const probs = Object.values(data.all_probs).map(p => (p * 100).toFixed(2));

    if (chart) chart.destroy();
    chart = new Chart(document.getElementById("chart"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "Probability (%)",
          data: probs,
          backgroundColor: "#76c7f0"
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, max: 100 }
        }
      }
    });

    resultDiv.classList.add("fade-in");
    setTimeout(() => resultDiv.classList.remove("fade-in"), 600);

  } catch (err) {
    loader.classList.add("hidden");
    alert("Error analyzing the image");
    console.error(err);
  }
});
