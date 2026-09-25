document.addEventListener("DOMContentLoaded", () => {
    // --- State ---
    let currentInferenceData = null;
    let currentOriginalB64 = null;
    let currentGradcamB64 = null;
    let currentSegB64 = null;

    // --- DOM Elements ---
    const dropzoneBox = document.getElementById("dropzoneBox");
    const fileInput = document.getElementById("fileInput");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const imageViewerBox = document.getElementById("imageViewerBox");
    const splitGrid = document.getElementById("splitGrid");
    const singleGrid = document.getElementById("singleGrid");
    const singleImg = document.getElementById("singleImg");
    const imgOriginal = document.getElementById("imgOriginal");
    const imgGradcam = document.getElementById("imgGradcam");
    const imgSeg = document.getElementById("imgSeg");
    const viewControls = document.getElementById("viewControls");
    const viewButtons = document.querySelectorAll(".view-btn");
    const inferenceLoading = document.getElementById("inferenceLoading");
    const btnClear = document.getElementById("btnClear");
    const btnOpenPdfModal = document.getElementById("btnOpenPdfModal");
    const btnConsultCopilot = document.getElementById("btnConsultCopilot");

    const samplesContainer = document.getElementById("samplesContainer");
    const emptyState = document.getElementById("emptyState");
    const resultsContent = document.getElementById("resultsContent");
    const statusBadge = document.getElementById("statusBadge");

    const diagBanner = document.getElementById("diagBanner");
    const diagTitle = document.getElementById("diagTitle");
    const diagDesc = document.getElementById("diagDesc");
    const diagIcon = document.getElementById("diagIcon");
    const confValue = document.getElementById("confValue");

    const probAllTxt = document.getElementById("probAllTxt");
    const probAllBar = document.getElementById("probAllBar");
    const probNormTxt = document.getElementById("probNormTxt");
    const probNormBar = document.getElementById("probNormBar");

    const valNcRatio = document.getElementById("valNcRatio");
    const lblNcStatus = document.getElementById("lblNcStatus");
    const valCircularity = document.getElementById("valCircularity");
    const valNucleusArea = document.getElementById("valNucleusArea");
    const valCytoArea = document.getElementById("valCytoArea");
    const actionText = document.getElementById("actionText");

    // Modal
    const pdfModal = document.getElementById("pdfModal");
    const btnClosePdfModal = document.getElementById("btnClosePdfModal");
    const btnCancelPdf = document.getElementById("btnCancelPdf");
    const pdfForm = document.getElementById("pdfForm");

    // Quick Demos
    const btnDemoNormal = document.getElementById("btnDemoNormal");
    const btnDemoAll = document.getElementById("btnDemoAll");

    if (btnDemoNormal) {
        btnDemoNormal.addEventListener("click", () => {
            triggerSample("data/samples/normal/normal_sample_001.jpg");
        });
    }

    if (btnDemoAll) {
        btnDemoAll.addEventListener("click", () => {
            triggerSample("data/samples/all_blast/all_blast_sample_001.jpg");
        });
    }

    // Load Samples
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                samplesContainer.innerHTML = "";
                data.samples.slice(0, 8).forEach((sample, idx) => {
                    const chip = document.createElement("div");
                    chip.className = `sample-chip ${sample.is_leukemia ? "is-all" : "is-normal"}`;
                    chip.innerHTML = `
                        <img src="${sample.thumbnail}" alt="${sample.category}">
                        <div>
                            <span class="chip-title">${sample.is_leukemia ? "ALL Blast Case #" + (idx + 1) : "Normal Case #" + (idx + 1)}</span>
                            <span class="chip-tag">${sample.category}</span>
                        </div>
                    `;
                    chip.addEventListener("click", () => {
                        triggerSample(sample.path);
                    });
                    samplesContainer.appendChild(chip);
                });
            } else {
                samplesContainer.innerHTML = "<span style='font-size:0.8rem; color:var(--text-muted);'>No sample images loaded.</span>";
            }
        } catch (e) {
            console.error("Failed to fetch samples:", e);
        }
    }
    loadSamples();

    // Drag and Drop
    dropzoneBox.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzoneBox.classList.add("dragover");
    });

    dropzoneBox.addEventListener("dragleave", () => {
        dropzoneBox.classList.remove("dragover");
    });

    dropzoneBox.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzoneBox.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    // Pipeline Step Progress
    function setPipelineStep(step) {
        for (let i = 1; i <= 5; i++) {
            const el = document.getElementById(`pStep${i}`);
            if (!el) continue;
            if (i < step) {
                el.className = "p-step done";
            } else if (i === step) {
                el.className = "p-step active";
            } else {
                el.className = "p-step";
            }
        }
    }

    async function triggerSample(path) {
        const formData = new FormData();
        formData.append("sample_path", path);
        runInference(formData);
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);
        runInference(formData);
    }

    async function runInference(formData) {
        inferenceLoading.style.display = "flex";
        setPipelineStep(2);

        try {
            setPipelineStep(3);
            const res = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const err = await res.json();
                alert(`Analysis Error: ${err.detail || 'Inference failed'}`);
                inferenceLoading.style.display = "none";
                return;
            }

            setPipelineStep(4);
            const json = await res.json();
            
            setTimeout(() => {
                setPipelineStep(5);
                renderResults(json.data);
                inferenceLoading.style.display = "none";
            }, 300);

        } catch (err) {
            console.error(err);
            alert("Connection error while communicating with pathology engine.");
            inferenceLoading.style.display = "none";
        }
    }

    function renderResults(data) {
        currentInferenceData = data;
        const pred = data.prediction;
        const morph = data.morphology;
        const images = data.images;

        // Save last report to localStorage so Copilot page can read it!
        try {
            localStorage.setItem("hemavision_last_report", JSON.stringify(data));
        } catch (e) {}

        currentOriginalB64 = images.original_base64;
        currentGradcamB64 = images.gradcam_base64;
        currentSegB64 = images.segmentation_base64;

        dropzoneContent.style.display = "none";
        imageViewerBox.style.display = "block";
        viewControls.style.display = "flex";
        btnClear.style.display = "inline-flex";
        btnOpenPdfModal.style.display = "inline-flex";
        btnConsultCopilot.style.display = "inline-flex";

        imgOriginal.src = currentOriginalB64;
        imgGradcam.src = currentGradcamB64;
        imgSeg.src = currentSegB64;

        switchView("split");

        emptyState.style.display = "none";
        resultsContent.style.display = "block";

        const isALL = pred.is_leukemia;
        diagBanner.className = `diag-banner ${isALL ? "is-leukemia" : "is-normal"}`;
        diagTitle.innerText = pred.label;
        diagIcon.innerText = isALL ? "🚨" : "🛡️";
        diagDesc.innerText = isALL
            ? "Malignant lymphoblasts with severe nuclear enlargement and chromatin hyperchromasia."
            : "Mature lymphocytic profile with physiological nuclear-to-cytoplasmic volume.";
        confValue.innerText = `${(pred.confidence * 100).toFixed(1)}%`;

        statusBadge.className = "badge-clinical";
        statusBadge.innerText = isALL ? "Positive for Leukemia" : "Negative / Benign";
        if (isALL) {
            statusBadge.style.background = "#fee2e2";
            statusBadge.style.color = "#b91c1c";
            statusBadge.style.borderColor = "#fecaca";
        } else {
            statusBadge.style.background = "#f0fdf4";
            statusBadge.style.color = "#16a34a";
            statusBadge.style.borderColor = "#bbf7d0";
        }

        const pAll = (pred.probabilities["Acute Lymphoblastic Leukemia (ALL)"] * 100).toFixed(1);
        const pNorm = (pred.probabilities["Normal"] * 100).toFixed(1);

        probAllTxt.innerText = `${pAll}%`;
        probAllBar.style.width = `${pAll}%`;
        probNormTxt.innerText = `${pNorm}%`;
        probNormBar.style.width = `${pNorm}%`;

        valNcRatio.innerText = morph.nc_ratio.toFixed(3);
        lblNcStatus.innerText = morph.morphology_status;
        valCircularity.innerText = morph.nuclear_circularity.toFixed(3);
        valNucleusArea.innerText = `${morph.nucleus_area_pixels.toLocaleString()} px`;
        valCytoArea.innerText = `${morph.cytoplasm_area_pixels.toLocaleString()} px`;

        actionText.innerText = data.clinical_decision.action;
    }

    // View Switching
    viewButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            viewButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            switchView(btn.getAttribute("data-view"));
        });
    });

    function switchView(mode) {
        if (mode === "split") {
            splitGrid.style.display = "grid";
            singleGrid.style.display = "none";
        } else {
            splitGrid.style.display = "none";
            singleGrid.style.display = "block";
            if (mode === "original") singleImg.src = currentOriginalB64;
            else if (mode === "gradcam") singleImg.src = currentGradcamB64;
            else if (mode === "seg") singleImg.src = currentSegB64;
        }
    }

    // Reset View
    btnClear.addEventListener("click", () => {
        currentInferenceData = null;
        dropzoneContent.style.display = "block";
        imageViewerBox.style.display = "none";
        viewControls.style.display = "none";
        btnClear.style.display = "none";
        btnOpenPdfModal.style.display = "none";
        btnConsultCopilot.style.display = "none";
        emptyState.style.display = "block";
        resultsContent.style.display = "none";
        statusBadge.className = "badge-clinical";
        statusBadge.innerText = "Awaiting Specimen";
        statusBadge.style = "";
        fileInput.value = "";
        setPipelineStep(1);
    });

    // Consult Copilot Button
    btnConsultCopilot.addEventListener("click", () => {
        window.location.href = "/copilot";
    });

    // PDF Report Modal
    btnOpenPdfModal.addEventListener("click", () => {
        pdfModal.style.display = "flex";
    });

    btnClosePdfModal.addEventListener("click", () => {
        pdfModal.style.display = "none";
    });

    btnCancelPdf.addEventListener("click", () => {
        pdfModal.style.display = "none";
    });

    pdfForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const pId = document.getElementById("pdfPatientId").value;
        const pName = document.getElementById("pdfPatientName").value;
        const notes = document.getElementById("pdfNotes").value;

        const downloadBtn = document.getElementById("btnDownloadPdf");
        downloadBtn.disabled = true;
        downloadBtn.innerText = "Generating PDF...";

        try {
            const formData = new FormData();
            formData.append("patient_id", pId);
            formData.append("patient_name", pName);
            formData.append("clinical_notes", notes);

            const res = await fetch("/api/generate-report", {
                method: "POST",
                body: formData
            });

            if (!res.ok) throw new Error("Failed to generate report");

            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = url;
            a.download = `Clinical_Report_${pId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            pdfModal.style.display = "none";
        } catch (err) {
            console.error(err);
            alert("Error downloading certified PDF report.");
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.innerText = "Download PDF Report";
        }
    });
});
