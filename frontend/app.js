const API_BASE = '';

const messagesEl   = document.getElementById('messages');
const emptyState   = document.getElementById('empty-state');
const input        = document.getElementById('user-input');
const sendBtn      = document.getElementById('send-btn');
const sessionLabel = document.getElementById('session-label');
const newSessionBtn = document.getElementById('new-session-btn');
const toast        = document.getElementById('toast');

let sessionId = null;
let isWaiting = false;

/* ── toast ── */
let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

/* ── session ── */
async function initSession() {
  sessionLabel.textContent = 'connecting…';
  try {
    const res = await fetch(`${API_BASE}/new_session`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    sessionId = data.session_id;
    sessionLabel.textContent = `session: ${sessionId.slice(0, 8)}…`;
    clearMessages();
  } catch {
    sessionLabel.textContent = 'connection failed';
    showToast('⚠ Could not reach the API server.');
  }
}

function clearMessages() {
  messagesEl.innerHTML = '';
  messagesEl.appendChild(emptyState);
  emptyState.style.display = 'flex';
}

/* ── messages ── */
function hideEmpty() {
  emptyState.style.display = 'none';
}

function appendMessage(role, text) {
  hideEmpty();
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;

  const label = document.createElement('span');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'you' : 'model';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function addTypingIndicator() {
  hideEmpty();
  const wrap = document.createElement('div');
  wrap.className = 'message bot';
  wrap.id = 'typing';

  const label = document.createElement('span');
  label.className = 'msg-label';
  label.textContent = 'model';

  const bubble = document.createElement('div');
  bubble.className = 'bubble typing-bubble';
  bubble.innerHTML = '<span></span><span></span><span></span>';

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTypingIndicator() {
  document.getElementById('typing')?.remove();
}

/* ── send ── */
async function sendMessage() {
  const text = input.value.trim();
  if (!text || isWaiting || !sessionId) return;

  isWaiting = true;
  sendBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  appendMessage('user', text);
  addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: text })
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    removeTypingIndicator();
    appendMessage('bot', data.response);
  } catch (err) {
    removeTypingIndicator();
    showToast(`⚠ ${err.message || 'Request failed.'}`);
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

/* ── auto-resize textarea ── */
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);
newSessionBtn.addEventListener('click', initSession);

/* ── boot ── */
initSession();
