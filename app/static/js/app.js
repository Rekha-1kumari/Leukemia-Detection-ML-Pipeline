// Global Navigation Helper
window.switchTabByName = function(tabName) {
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");
    navTabs.forEach(t => t.classList.remove("active"));
    tabContents.forEach(c => c.classList.remove("active"));

    const targetTab = document.querySelector(`.nav-tab[data-tab="${tabName}"]`);
    const targetContent = document.getElementById(`tab-${tabName}`);
    if (targetTab) targetTab.classList.add("active");
    if (targetContent) targetContent.classList.add("active");
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

document.addEventListener("DOMContentLoaded", () => {
    // --- Navigation Tabs ---
    const navTabs = document.querySelectorAll(".nav-tab");
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.getAttribute("data-tab");
            window.switchTabByName(target);
        });
    });

    // --- State Variables ---
    let currentInferenceData = null;
    let currentOriginalB64 = null;
    let currentGradcamB64 = null;
    let currentSegB64 = null;

    // --- DOM Elements ---
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const imageViewerContainer = document.getElementById("imageViewerContainer");
    const splitView = document.getElementById("splitView");
    const singleView = document.getElementById("singleView");
    const singleViewImg = document.getElementById("singleViewImg");
    const imgOriginal = document.getElementById("imgOriginal");
    const imgGradcam = document.getElementById("imgGradcam");
    const imgSeg = document.getElementById("imgSeg");
    const viewControls = document.getElementById("viewControls");
    const viewButtons = document.querySelectorAll(".view-btn");
    const inferenceLoading = document.getElementById("inferenceLoading");
    const resetBtn = document.getElementById("resetBtn");
    const openReportModalBtn = document.getElementById("openReportModalBtn");
    const askCopilotBtn = document.getElementById("askCopilotBtn");

    const sampleCardsContainer = document.getElementById("sampleCardsContainer");
    const resultsEmpty = document.getElementById("resultsEmpty");
    const diagnosticBody = document.getElementById("diagnosticBody");
    const riskChip = document.getElementById("riskChip");
    const diagnosisBanner = document.getElementById("diagnosisBanner");
    const bannerTitle = document.getElementById("bannerTitle");
    const bannerDesc = document.getElementById("bannerDesc");
    const bannerIcon = document.getElementById("bannerIcon");
    const confVal = document.getElementById("confVal");

    const probAllText = document.getElementById("probAllText");
    const probAllBar = document.getElementById("probAllBar");
    const probNormalText = document.getElementById("probNormalText");
    const probNormalBar = document.getElementById("probNormalBar");

    const morphNCRatio = document.getElementById("morphNCRatio");
    const morphNCStatus = document.getElementById("morphNCStatus");
    const morphCircularity = document.getElementById("morphCircularity");
    const morphNucleusArea = document.getElementById("morphNucleusArea");
    const morphCytoArea = document.getElementById("morphCytoArea");
    const xaiExplanation = document.getElementById("xaiExplanation");
    const actionText = document.getElementById("actionText");

    // Modal Elements
    const reportModal = document.getElementById("reportModal");
    const closeReportModalBtn = document.getElementById("closeReportModalBtn");
    const cancelReportBtn = document.getElementById("cancelReportBtn");
    const reportForm = document.getElementById("reportForm");

    // --- Quick Demo Buttons ---
    const quickNormalBtn = document.getElementById("quickNormalBtn");
    const quickAllBtn = document.getElementById("quickAllBtn");

    if (quickNormalBtn) {
        quickNormalBtn.addEventListener("click", () => {
            triggerSampleInference("data/samples/normal/normal_sample_001.jpg");
        });
    }

    if (quickAllBtn) {
        quickAllBtn.addEventListener("click", () => {
            triggerSampleInference("data/samples/all_blast/all_blast_sample_001.jpg");
        });
    }

    // --- Load Sample Blood Smears ---
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                sampleCardsContainer.innerHTML = "";
                data.samples.slice(0, 8).forEach((sample, idx) => {
                    const card = document.createElement("div");
                    card.className = `sample-card-item ${sample.is_leukemia ? "is-all" : "is-normal"}`;
                    card.innerHTML = `
                        <img src="${sample.thumbnail}" alt="${sample.category}">
                        <div class="sample-meta">
                            <span class="sample-title">${sample.is_leukemia ? "ALL Blast Case #" + (idx + 1) : "Normal Case #" + (idx + 1)}</span>
                            <span class="sample-tag">${sample.category}</span>
                        </div>
                    `;
                    card.addEventListener("click", () => {
                        triggerSampleInference(sample.path);
                    });
                    sampleCardsContainer.appendChild(card);
                });
            } else {
                sampleCardsContainer.innerHTML = "<span style='font-size:0.8rem; color:var(--text-muted);'>No sample images loaded.</span>";
            }
        } catch (e) {
            console.error("Failed to fetch samples:", e);
        }
    }
    loadSamples();

    // --- Drag & Drop ---
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    // --- Pipeline Visual Step Animator ---
    function updatePipelineSteps(activeStep) {
        for (let i = 1; i <= 5; i++) {
            const el = document.getElementById(`pipeStep${i}`);
            if (!el) continue;
            if (i < activeStep) {
                el.className = "pipeline-step step-complete";
            } else if (i === activeStep) {
                el.className = "pipeline-step step-active";
            } else {
                el.className = "pipeline-step";
            }
        }
    }

    // --- Inference Trigger ---
    async function triggerSampleInference(samplePath) {
        const formData = new FormData();
        formData.append("sample_path", samplePath);
        executeInference(formData);
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);
        executeInference(formData);
    }

    async function executeInference(formData) {
        inferenceLoading.style.display = "flex";
        updatePipelineSteps(2);

        try {
            updatePipelineSteps(3);
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

            updatePipelineSteps(4);
            const json = await res.json();
            
            setTimeout(() => {
                updatePipelineSteps(5);
                renderPredictionResults(json.data);
                inferenceLoading.style.display = "none";
            }, 300);

        } catch (err) {
            console.error(err);
            alert("Network error while connecting to HemaVision AI server.");
            inferenceLoading.style.display = "none";
        }
    }

    // --- Render Results ---
    function renderPredictionResults(data) {
        currentInferenceData = data;
        const pred = data.prediction;
        const morph = data.morphology;
        const images = data.images;

        currentOriginalB64 = images.original_base64;
        currentGradcamB64 = images.gradcam_base64;
        currentSegB64 = images.segmentation_base64;

        dropzoneContent.style.display = "none";
        imageViewerContainer.style.display = "block";
        viewControls.style.display = "flex";
        resetBtn.style.display = "inline-flex";
        openReportModalBtn.style.display = "inline-flex";
        if (askCopilotBtn) askCopilotBtn.style.display = "inline-flex";

        imgOriginal.src = currentOriginalB64;
        imgGradcam.src = currentGradcamB64;
        imgSeg.src = currentSegB64;

        switchView("split");

        resultsEmpty.style.display = "none";
        diagnosticBody.style.display = "block";

        const isALL = pred.is_leukemia;
        diagnosisBanner.className = `diagnosis-banner ${isALL ? "is-leukemia" : "is-normal"}`;
        bannerTitle.innerText = pred.label;
        bannerIcon.innerText = isALL ? "🚨" : "🛡️";
        bannerDesc.innerText = isALL
            ? "High concentration of lymphoblasts with abnormal nuclear enlargement and hyperchromatin."
            : "Mature lymphocytic profile with normal nuclear-to-cytoplasm ratio and smooth nuclear membrane.";
        confVal.innerText = `${(pred.confidence * 100).toFixed(1)}%`;

        riskChip.className = `status-chip ${isALL ? "chip-danger" : "chip-success"}`;
        riskChip.innerText = isALL ? "HIGH RISK / LEUKEMIA" : "NORMAL / BENIGN";

        const pAll = (pred.probabilities["Acute Lymphoblastic Leukemia (ALL)"] * 100).toFixed(1);
        const pNorm = (pred.probabilities["Normal"] * 100).toFixed(1);

        probAllText.innerText = `${pAll}%`;
        probAllBar.style.width = `${pAll}%`;
        probNormalText.innerText = `${pNorm}%`;
        probNormalBar.style.width = `${pNorm}%`;

        morphNCRatio.innerText = morph.nc_ratio.toFixed(3);
        morphNCStatus.innerText = morph.morphology_status;
        morphCircularity.innerText = morph.nuclear_circularity.toFixed(3);
        morphNucleusArea.innerText = `${morph.nucleus_area_pixels.toLocaleString()} px`;
        morphCytoArea.innerText = `${morph.cytoplasm_area_pixels.toLocaleString()} px`;

        if (isALL) {
            xaiExplanation.innerHTML = `
                The Grad-CAM saliency highlights intensive gradient concentration across the <b>atypical, enlarged lymphoblast nucleus</b>. The model focal attention aligns with aberrant nuclear chromatin clumping, confirming true cytological pathology rather than background artifacts.
            `;
        } else {
            xaiExplanation.innerHTML = `
                The Grad-CAM heat map demonstrates balanced, distributed activations across the <b>compact, mature lymphocyte nucleus</b> with regular chromatin dispersion, corroborating benign resting morphology.
            `;
        }

        actionText.innerText = data.clinical_decision.action;
    }

    // --- View Mode Controls ---
    viewButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            viewButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            switchView(btn.getAttribute("data-view"));
        });
    });

    function switchView(mode) {
        if (mode === "split") {
            splitView.style.display = "grid";
            singleView.style.display = "none";
        } else {
            splitView.style.display = "none";
            singleView.style.display = "block";
            if (mode === "original") singleViewImg.src = currentOriginalB64;
            else if (mode === "gradcam") singleViewImg.src = currentGradcamB64;
            else if (mode === "seg") singleViewImg.src = currentSegB64;
        }
    }

    // --- Reset View ---
    resetBtn.addEventListener("click", () => {
        currentInferenceData = null;
        dropzoneContent.style.display = "block";
        imageViewerContainer.style.display = "none";
        viewControls.style.display = "none";
        resetBtn.style.display = "none";
        openReportModalBtn.style.display = "none";
        if (askCopilotBtn) askCopilotBtn.style.display = "none";
        resultsEmpty.style.display = "block";
        diagnosticBody.style.display = "none";
        riskChip.className = "status-chip";
        riskChip.innerText = "Awaiting Smear";
        fileInput.value = "";
        updatePipelineSteps(1);
    });

    // --- PDF Report Modal & Download ---
    openReportModalBtn.addEventListener("click", () => {
        reportModal.style.display = "flex";
    });

    closeReportModalBtn.addEventListener("click", () => {
        reportModal.style.display = "none";
    });

    cancelReportBtn.addEventListener("click", () => {
        reportModal.style.display = "none";
    });

    reportForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const pId = document.getElementById("reportPatientId").value;
        const pName = document.getElementById("reportPatientName").value;
        const notes = document.getElementById("reportNotes").value;

        const downloadBtn = document.getElementById("downloadPdfBtn");
        downloadBtn.disabled = true;
        downloadBtn.innerHTML = "Generating PDF...";

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
            reportModal.style.display = "none";
        } catch (err) {
            console.error(err);
            alert("Error downloading PDF clinical report.");
        } finally {
            downloadBtn.disabled = false;
            downloadBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                Download PDF Report
            `;
        }
    });

    // --- Interactive Symptoms Self-Assessment ---
    const symptomChecks = document.querySelectorAll(".symptom-check");
    const symptomsFeedback = document.getElementById("symptomsFeedback");
    const feedbackBadge = document.getElementById("feedbackBadge");
    const feedbackText = document.getElementById("feedbackText");

    symptomChecks.forEach(cb => {
        cb.addEventListener("change", () => {
            let totalWeight = 0;
            let checkedCount = 0;
            symptomChecks.forEach(c => {
                if (c.checked) {
                    totalWeight += parseInt(c.getAttribute("data-weight") || 1);
                    checkedCount++;
                }
            });

            if (checkedCount === 0) {
                symptomsFeedback.style.display = "none";
            } else {
                symptomsFeedback.style.display = "block";
                if (totalWeight >= 4) {
                    feedbackBadge.innerText = "Medical Consultation Strongly Advised";
                    feedbackBadge.style.color = "#e2847a";
                    feedbackText.innerText = "You have selected multiple hallmark indicators of bone marrow suppression (e.g. bleeding/petechiae with fatigue or fever). We strongly recommend visiting a primary care physician or hematologist for a routine Complete Blood Count (CBC) and physical examination.";
                } else {
                    feedbackBadge.innerText = "Mild / Non-Specific Symptoms";
                    feedbackBadge.style.color = "var(--accent-sage-light)";
                    feedbackText.innerText = "These symptoms can arise from everyday benign viral infections or minor nutritional deficiencies. However, if symptoms persist for more than 2 weeks, a simple CBC blood test provides quick peace of mind.";
                }
            }
        });
    });

    // --- Treatment Tabs Switcher ---
    const treatButtons = document.querySelectorAll(".treat-btn");
    treatButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const treat = btn.getAttribute("data-treat");
            treatButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            document.querySelectorAll(".treat-view").forEach(v => v.style.display = "none");
            const targetView = document.getElementById(`view-${treat}`);
            if (targetView) targetView.style.display = "block";

            // If switching to CAR-T simulation, trigger animation loop
            if (treat === "cart") {
                initCartSimulation();
            }
        });
    });

    // ==================== CAR-T CELL CANVAS SIMULATION ====================
    const canvas = document.getElementById("cartCanvas");
    let ctx2d = null;
    let simRunning = true;
    let animationId = null;

    let blasts = [];
    let cartCells = [];
    let particles = [];

    function initCartSimulation() {
        if (!canvas) return;
        ctx2d = canvas.getContext("2d");
        
        // Populate initial cellular micro-environment
        blasts = [];
        cartCells = [];
        particles = [];

        // 12 Leukemic Blast Cells
        for (let i = 0; i < 12; i++) {
            blasts.push({
                x: 100 + Math.random() * (canvas.width - 200),
                y: 60 + Math.random() * (canvas.height - 120),
                radius: 18 + Math.random() * 6,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                health: 100,
                lysing: false,
                dead: false,
                lyseTimer: 0
            });
        }

        // 6 Engineered CAR-T Cells
        for (let j = 0; j < 6; j++) {
            cartCells.push({
                x: 40 + Math.random() * (canvas.width - 80),
                y: 40 + Math.random() * (canvas.height - 80),
                radius: 12,
                vx: (Math.random() - 0.5) * 1.4,
                vy: (Math.random() - 0.5) * 1.4,
                target: null,
                docked: false,
                dockTimer: 0
            });
        }

        if (animationId) cancelAnimationFrame(animationId);
        runCartLoop();
    }

    function runCartLoop() {
        if (!ctx2d) return;

        // Clear canvas with deep dark medical backdrop
        ctx2d.fillStyle = "#0a110e";
        ctx2d.fillRect(0, 0, canvas.width, canvas.height);

        // Draw faint capillary / tissue grid lines
        ctx2d.strokeStyle = "rgba(132, 169, 140, 0.05)";
        ctx2d.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx2d.beginPath();
            ctx2d.moveTo(x, 0);
            ctx2d.lineTo(x, canvas.height);
            ctx2d.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx2d.beginPath();
            ctx2d.moveTo(0, y);
            ctx2d.lineTo(canvas.width, y);
            ctx2d.stroke();
        }

        // 1. Update & Draw Blast Cells
        blasts.forEach(b => {
            if (!b.dead) {
                b.x += b.vx;
                b.y += b.vy;

                // Bounce off canvas edges
                if (b.x < b.radius || b.x > canvas.width - b.radius) b.vx *= -1;
                if (b.y < b.radius || b.y > canvas.height - b.radius) b.vy *= -1;

                if (b.lysing) {
                    b.health -= 0.6;
                    b.lyseTimer++;
                    // Spawn lysis particles
                    if (Math.random() < 0.3) {
                        particles.push({
                            x: b.x + (Math.random() - 0.5) * b.radius,
                            y: b.y + (Math.random() - 0.5) * b.radius,
                            vx: (Math.random() - 0.5) * 1.2,
                            vy: (Math.random() - 0.5) * 1.2,
                            life: 30,
                            color: "rgba(224, 109, 83, 0.6)"
                        });
                    }
                    if (b.health <= 0) {
                        b.dead = true;
                        b.lysing = false;
                    }
                }

                // Render Blast Cell
                ctx2d.beginPath();
                ctx2d.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
                ctx2d.fillStyle = b.lysing ? "rgba(200, 90, 72, 0.4)" : "rgba(224, 109, 83, 0.8)";
                ctx2d.fill();
                ctx2d.strokeStyle = "#e06d53";
                ctx2d.lineWidth = 1.5;
                ctx2d.stroke();

                // Draw Enlarged Malignant Nucleus (High N:C ratio visual!)
                ctx2d.beginPath();
                ctx2d.arc(b.x, b.y, b.radius * 0.82, 0, Math.PI * 2);
                ctx2d.fillStyle = "rgba(100, 40, 30, 0.85)";
                ctx2d.fill();

                // Draw CD19 surface receptors (tiny spikes around perimeter)
                if (!b.lysing) {
                    for (let a = 0; a < Math.PI * 2; a += Math.PI / 4) {
                        const px = b.x + Math.cos(a) * (b.radius + 3);
                        const py = b.y + Math.sin(a) * (b.radius + 3);
                        ctx2d.beginPath();
                        ctx2d.arc(px, py, 2, 0, Math.PI * 2);
                        ctx2d.fillStyle = "#ff9a80";
                        ctx2d.fill();
                    }
                }
            } else {
                // Dead / Lysed Cell Fragment
                ctx2d.beginPath();
                ctx2d.arc(b.x, b.y, b.radius * 0.6, 0, Math.PI * 2);
                ctx2d.fillStyle = "rgba(74, 85, 80, 0.3)";
                ctx2d.fill();
                ctx2d.strokeStyle = "rgba(107, 124, 114, 0.4)";
                ctx2d.setLineDash([3, 3]);
                ctx2d.stroke();
                ctx2d.setLineDash([]);
            }
        });

        // 2. Update & Draw CAR-T Cells
        cartCells.forEach(c => {
            // Find closest living blast cell
            if (!c.docked) {
                let closest = null;
                let minDist = 9999;
                blasts.forEach(b => {
                    if (!b.dead && !b.lysing) {
                        const d = Math.hypot(b.x - c.x, b.y - c.y);
                        if (d < minDist) {
                            minDist = d;
                            closest = b;
                        }
                    }
                });

                if (closest) {
                    const angle = Math.atan2(closest.y - c.y, closest.x - c.x);
                    c.vx = Math.cos(angle) * 1.6;
                    c.vy = Math.sin(angle) * 1.6;

                    // Check for binding / docking
                    if (minDist < c.radius + closest.radius + 4) {
                        c.docked = true;
                        c.target = closest;
                        closest.lysing = true;
                    }
                }

                c.x += c.vx;
                c.y += c.vy;
            } else {
                // Docked to blast cell: releasing perforin/granzyme
                if (c.target && !c.target.dead) {
                    c.dockTimer++;
                    // Draw perforin connection beam
                    ctx2d.beginPath();
                    ctx2d.moveTo(c.x, c.y);
                    ctx2d.lineTo(c.target.x, c.target.y);
                    ctx2d.strokeStyle = "rgba(116, 198, 157, 0.8)";
                    ctx2d.lineWidth = 2.5;
                    ctx2d.stroke();

                    if (c.dockTimer > 120 || c.target.dead) {
                        c.docked = false;
                        c.target = null;
                        c.dockTimer = 0;
                        c.vx = (Math.random() - 0.5) * 1.5;
                        c.vy = (Math.random() - 0.5) * 1.5;
                    }
                } else {
                    c.docked = false;
                    c.target = null;
                }
            }

            // Render CAR-T Cell
            ctx2d.beginPath();
            ctx2d.arc(c.x, c.y, c.radius, 0, Math.PI * 2);
            ctx2d.fillStyle = "#52b788";
            ctx2d.fill();
            ctx2d.strokeStyle = "#84a98c";
            ctx2d.lineWidth = 1.5;
            ctx2d.stroke();

            // Chimeric Antigen Receptor (CAR) anchor points
            for (let a = 0; a < Math.PI * 2; a += Math.PI / 3) {
                const rx = c.x + Math.cos(a) * (c.radius + 4);
                const ry = c.y + Math.sin(a) * (c.radius + 4);
                ctx2d.fillStyle = "#a3c4ab";
                ctx2d.fillRect(rx - 1.5, ry - 1.5, 3, 3);
            }
        });

        // 3. Render Particles
        particles.forEach((p, idx) => {
            p.x += p.vx;
            p.y += p.vy;
            p.life--;
            ctx2d.beginPath();
            ctx2d.arc(p.x, p.y, 2, 0, Math.PI * 2);
            ctx2d.fillStyle = p.color;
            ctx2d.fill();
            if (p.life <= 0) particles.splice(idx, 1);
        });

        if (simRunning) {
            animationId = requestAnimationFrame(runCartLoop);
        }
    }

    // Controls for CAR-T Canvas
    const cartToggleBtn = document.getElementById("cartToggleBtn");
    const cartResetBtn = document.getElementById("cartResetBtn");

    if (cartToggleBtn) {
        cartToggleBtn.addEventListener("click", () => {
            simRunning = !simRunning;
            cartToggleBtn.innerText = simRunning ? "Pause Simulation" : "Resume Simulation";
            if (simRunning) runCartLoop();
        });
    }

    if (cartResetBtn) {
        cartResetBtn.addEventListener("click", () => {
            initCartSimulation();
        });
    }

    // Auto-init CAR-T canvas on page load
    initCartSimulation();

    // ==================== GEMINI AI COPILOT CHATBOT ====================
    const chatInput = document.getElementById("chatInput");
    const sendChatBtn = document.getElementById("sendChatBtn");
    const chatMessages = document.getElementById("chatMessages");
    const clearChatBtn = document.getElementById("clearChatBtn");

    window.askCopilotCurrentReport = function() {
        window.switchTabByName("copilot");
        if (currentInferenceData) {
            const pred = currentInferenceData.prediction.label;
            const conf = (currentInferenceData.prediction.confidence * 100).toFixed(1);
            const nc = currentInferenceData.morphology.nc_ratio;
            const prompt = `Please explain the current report in detail: Diagnosis is ${pred} (Confidence: ${conf}%) with an N:C ratio of ${nc}. What does this mean for the patient, and what should be the immediate next clinical steps?`;
            sendChatMessage(prompt);
        } else {
            sendChatMessage("Please explain what findings typically appear in a peripheral blood smear for suspected leukemia.");
        }
    };

    window.sendQuickPrompt = function(promptText) {
        sendChatMessage(promptText);
    };

    async function sendChatMessage(text) {
        if (!text || text.trim() === "") return;

        // Get selected persona mode
        const mode = document.querySelector('input[name="chatMode"]:checked')?.value || "patient";

        // Append User Message to Chat UI
        appendChatMessage("user", text);
        chatInput.value = "";

        // Append Loading Spinner
        const loadingId = "loading-" + Date.now();
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "chat-msg msg-ai";
        loadingDiv.id = loadingId;
        loadingDiv.innerHTML = `
            <div class="msg-avatar">🤖</div>
            <div class="msg-content">
                <span style="font-size:0.8rem; color:var(--text-muted);">Consulting Google Gemini 2.5 Flash...</span>
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: jsonStringifySafe({
                    message: text,
                    mode: mode,
                    context: currentInferenceData || null
                })
            });

            const data = await res.json();
            const loaderEl = document.getElementById(loadingId);
            if (loaderEl) loaderEl.remove();

            appendChatMessage("ai", data.response || "No response received from clinical core.");

        } catch (err) {
            console.error(err);
            const loaderEl = document.getElementById(loadingId);
            if (loaderEl) loaderEl.remove();
            appendChatMessage("ai", "I encountered a communication error with the Gemini clinical engine. Please ensure network connectivity.");
        }
    }

    function jsonStringifySafe(obj) {
        return JSON.stringify(obj, (k, v) => (k === "_raw_pil" ? undefined : v));
    }

    function appendChatMessage(sender, text) {
        const msgDiv = document.createElement("div");
        msgDiv.className = `chat-msg ${sender === "user" ? "msg-user" : "msg-ai"}`;
        
        // Simple Markdown parsing for formatting
        let formatted = text
            .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
            .replace(/\*(.*?)\*/g, '<i>$1</i>')
            .replace(/### (.*?)\n/g, '<h4 style="margin: 8px 0 4px; color:#eaf1ec;">$1</h4>')
            .replace(/## (.*?)\n/g, '<h3 style="margin: 10px 0 4px; color:#eaf1ec;">$1</h3>')
            .replace(/\n\n/g, '<br/><br/>')
            .replace(/\n- (.*?)/g, '<br/>• $1');

        msgDiv.innerHTML = `
            <div class="msg-avatar">${sender === "user" ? "👤" : "🤖"}</div>
            <div class="msg-content">${formatted}</div>
        `;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    if (sendChatBtn && chatInput) {
        sendChatBtn.addEventListener("click", () => {
            sendChatMessage(chatInput.value);
        });

        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage(chatInput.value);
            }
        });
    }

    if (clearChatBtn) {
        clearChatBtn.addEventListener("click", () => {
            chatMessages.innerHTML = `
                <div class="chat-msg msg-ai">
                    <div class="msg-avatar">🤖</div>
                    <div class="msg-content">
                        <p>Chat history cleared. How may I assist your clinical cytological research today?</p>
                    </div>
                </div>
            `;
        });
    }

    // --- Viva FAQ Accordion ---
    document.querySelectorAll(".faq-question").forEach(q => {
        q.addEventListener("click", () => {
            const item = q.parentElement;
            item.classList.toggle("open");
        });
    });
});
