document.addEventListener("DOMContentLoaded", () => {
    // Sub-navigation switcher
    const navButtons = document.querySelectorAll(".t-nav-btn");
    const sections = document.querySelectorAll(".treat-section");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const treat = btn.getAttribute("data-treat");
            navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            sections.forEach(s => s.style.display = "none");
            const targetSection = document.getElementById(`view-${treat}`);
            if (targetSection) targetSection.style.display = "block";

            if (treat === "cart") {
                initCartSimulation();
            }
        });
    });

    // ==================== CAR-T CELL SIMULATION ====================
    const canvas = document.getElementById("cartCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let simRunning = true;
    let animId = null;
    let blasts = [];
    let cartCells = [];
    let particles = [];

    function initCartSimulation() {
        blasts = [];
        cartCells = [];
        particles = [];

        // 12 Leukemic Blast Cells (CD19+)
        for (let i = 0; i < 12; i++) {
            blasts.push({
                x: 100 + Math.random() * (canvas.width - 200),
                y: 60 + Math.random() * (canvas.height - 120),
                radius: 17 + Math.random() * 5,
                vx: (Math.random() - 0.5) * 0.4,
                vy: (Math.random() - 0.5) * 0.4,
                health: 100,
                lysing: false,
                dead: false
            });
        }

        // 6 Reprogrammed CAR-T Cells
        for (let j = 0; j < 6; j++) {
            cartCells.push({
                x: 40 + Math.random() * (canvas.width - 80),
                y: 40 + Math.random() * (canvas.height - 80),
                radius: 11,
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5,
                target: null,
                docked: false,
                dockTimer: 0
            });
        }

        if (animId) cancelAnimationFrame(animId);
        simLoop();
    }

    function simLoop() {
        // Deep slate microscopic background
        ctx.fillStyle = "#09101d";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Grid lines (capillary mesh)
        ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }

        // 1. Draw Blast Cells
        blasts.forEach(b => {
            if (!b.dead) {
                b.x += b.vx;
                b.y += b.vy;

                if (b.x < b.radius || b.x > canvas.width - b.radius) b.vx *= -1;
                if (b.y < b.radius || b.y > canvas.height - b.radius) b.vy *= -1;

                if (b.lysing) {
                    b.health -= 0.6;
                    if (Math.random() < 0.25) {
                        particles.push({
                            x: b.x + (Math.random() - 0.5) * b.radius,
                            y: b.y + (Math.random() - 0.5) * b.radius,
                            vx: (Math.random() - 0.5) * 1.4,
                            vy: (Math.random() - 0.5) * 1.4,
                            life: 25,
                            color: "rgba(239, 68, 68, 0.6)"
                        });
                    }
                    if (b.health <= 0) {
                        b.dead = true;
                        b.lysing = false;
                    }
                }

                // Cell Body
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
                ctx.fillStyle = b.lysing ? "rgba(239, 68, 68, 0.4)" : "#ef4444";
                ctx.fill();
                ctx.strokeStyle = "#dc2626";
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Enlarged Nucleus (High N:C ratio)
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.radius * 0.82, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(127, 29, 29, 0.85)";
                ctx.fill();

                // Surface CD19 receptors
                if (!b.lysing) {
                    for (let a = 0; a < Math.PI * 2; a += Math.PI / 4) {
                        const px = b.x + Math.cos(a) * (b.radius + 3);
                        const py = b.y + Math.sin(a) * (b.radius + 3);
                        ctx.beginPath();
                        ctx.arc(px, py, 2, 0, Math.PI * 2);
                        ctx.fillStyle = "#fca5a5";
                        ctx.fill();
                    }
                }
            } else {
                // Fragmented dead cell
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.radius * 0.5, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(100, 116, 139, 0.25)";
                ctx.fill();
                ctx.strokeStyle = "rgba(100, 116, 139, 0.4)";
                ctx.setLineDash([2, 2]);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        });

        // 2. Draw CAR-T Cells
        cartCells.forEach(c => {
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
                    c.vx = Math.cos(angle) * 1.5;
                    c.vy = Math.sin(angle) * 1.5;

                    if (minDist < c.radius + closest.radius + 3) {
                        c.docked = true;
                        c.target = closest;
                        closest.lysing = true;
                    }
                }

                c.x += c.vx;
                c.y += c.vy;
            } else {
                if (c.target && !c.target.dead) {
                    c.dockTimer++;
                    // Perforin beam
                    ctx.beginPath();
                    ctx.moveTo(c.x, c.y);
                    ctx.lineTo(c.target.x, c.target.y);
                    ctx.strokeStyle = "rgba(52, 211, 153, 0.85)";
                    ctx.lineWidth = 2.5;
                    ctx.stroke();

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

            // Render CAR-T
            ctx.beginPath();
            ctx.arc(c.x, c.y, c.radius, 0, Math.PI * 2);
            ctx.fillStyle = "#10b981";
            ctx.fill();
            ctx.strokeStyle = "#059669";
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Receptors
            for (let a = 0; a < Math.PI * 2; a += Math.PI / 3) {
                const rx = c.x + Math.cos(a) * (c.radius + 4);
                const ry = c.y + Math.sin(a) * (c.radius + 4);
                ctx.fillStyle = "#a7f3d0";
                ctx.fillRect(rx - 1.5, ry - 1.5, 3, 3);
            }
        });

        // 3. Render Particles
        particles.forEach((p, idx) => {
            p.x += p.vx;
            p.y += p.vy;
            p.life--;
            ctx.beginPath();
            ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.fill();
            if (p.life <= 0) particles.splice(idx, 1);
        });

        if (simRunning) {
            animId = requestAnimationFrame(simLoop);
        }
    }

    const btnToggleCart = document.getElementById("btnToggleCart");
    const btnResetCart = document.getElementById("btnResetCart");

    if (btnToggleCart) {
        btnToggleCart.addEventListener("click", () => {
            simRunning = !simRunning;
            btnToggleCart.innerText = simRunning ? "Pause Simulation" : "Resume Simulation";
            if (simRunning) simLoop();
        });
    }

    if (btnResetCart) {
        btnResetCart.addEventListener("click", () => {
            initCartSimulation();
        });
    }

    initCartSimulation();
});
