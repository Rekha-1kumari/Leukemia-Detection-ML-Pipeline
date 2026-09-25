document.addEventListener("DOMContentLoaded", () => {
    const chatInputText = document.getElementById("chatInputText");
    const btnSendChat = document.getElementById("btnSendChat");
    const chatHistory = document.getElementById("chatHistory");
    const btnClearChat = document.getElementById("btnClearChat");

    // Load active patient report from localStorage if user analyzed an image
    let activeReportContext = null;
    try {
        const raw = localStorage.getItem("hemavision_last_report");
        if (raw) activeReportContext = JSON.parse(raw);
    } catch (e) {}

    window.sendPrompt = function(promptText) {
        handleUserMessage(promptText);
    };

    async function handleUserMessage(text) {
        if (!text || text.trim() === "") return;

        const mode = document.querySelector('input[name="chatMode"]:checked')?.value || "patient";

        // Display user message in UI
        addMessageBubble("user", text);
        if (chatInputText) chatInputText.value = "";

        // Add loading indicator
        const loadingId = "load-" + Date.now();
        const loadDiv = document.createElement("div");
        loadDiv.className = "chat-bubble";
        loadDiv.id = loadingId;
        loadDiv.innerHTML = `
            <div class="chat-avatar">🤖</div>
            <div class="chat-content" style="color: var(--text-muted); font-size: 0.84rem;">
                Consulting Google Gemini 2.5 Flash...
            </div>
        `;
        chatHistory.appendChild(loadDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        try {
            const res = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: text,
                    mode: mode,
                    context: activeReportContext
                })
            });

            const data = await res.json();
            const loaderEl = document.getElementById(loadingId);
            if (loaderEl) loaderEl.remove();

            addMessageBubble("ai", data.response || "No response received from clinical core.");

        } catch (err) {
            console.error(err);
            const loaderEl = document.getElementById(loadingId);
            if (loaderEl) loaderEl.remove();
            addMessageBubble("ai", "I encountered a communication error with the clinical reasoning engine. Please verify network connectivity.");
        }
    }

    function addMessageBubble(sender, text) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${sender}`;

        let formatted = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/### (.*?)\n/g, '<h4 style="margin: 12px 0 4px; color: var(--text-primary); font-size: 0.98rem; font-weight: 700;">$1</h4>')
            .replace(/## (.*?)\n/g, '<h3 style="margin: 14px 0 6px; color: var(--text-primary); font-size: 1.08rem; font-weight: 800;">$1</h3>')
            .replace(/\n- (.*?)/g, '<br/>• $1')
            .replace(/\n([0-9]+)\. (.*?)/g, '<br/><strong>$1.</strong> $2')
            .replace(/\n\n/g, '<br/><br/>');

        bubble.innerHTML = `
            <div class="chat-avatar">${sender === "user" ? "👤" : "🤖"}</div>
            <div class="chat-content">${formatted}</div>
        `;
        chatHistory.appendChild(bubble);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    if (btnSendChat && chatInputText) {
        btnSendChat.addEventListener("click", () => {
            handleUserMessage(chatInputText.value);
        });

        chatInputText.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleUserMessage(chatInputText.value);
            }
        });
    }

    if (btnClearChat) {
        btnClearChat.addEventListener("click", () => {
            chatHistory.innerHTML = `
                <div class="chat-bubble">
                    <div class="chat-avatar">🤖</div>
                    <div class="chat-content">
                        <p>Chat history cleared. How may I assist your hematology research today?</p>
                    </div>
                </div>
            `;
        });
    }
});
