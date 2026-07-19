// ── Markdown renderer ──────────────────────────────────────────────────────

function renderMarkdown(text) {
    if (!text) return '';
    var html = text;

    // Escape HTML first (except what we'll generate)
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // Code blocks (must be before inline code)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
        return '<pre><code>' + code.replace(/\n$/, '') + '</code></pre>';
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headings
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Horizontal rule
    html = html.replace(/^---$/gm, '<hr>');

    // Unordered lists — group consecutive li lines
    html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Blockquotes
    html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

    // Paragraphs: double newlines
    html = html.replace(/\n\n+/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Clean up empty paragraphs and newlines inside blocks
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/\n/g, '<br>');
    html = html.replace(/<p><br>/g, '<p>');
    html = html.replace(/<br><\/p>/g, '</p>');
    html = html.replace(/<\/ul>\n<br>/g, '</ul>');
    html = html.replace(/<\/pre>\n<br>/g, '</pre>');
    html = html.replace(/<br><\/li>/g, '</li>');
    html = html.replace(/<h([1-4])>/g, '<br><h$1>');

    // Remove leading <br>
    html = html.replace(/^<br>/, '');

    return html;
}

// ── AppState ──────────────────────────────────────────────────────────────

var AppState = {
    sessionId: null,        // Currently active session ID
    activeTab: 'documents',
    courseChoices: [],
    chatAbortController: null,
    pendingFiles: [],
    sessions: [],           // List of sessions for the current course
};

// ── Toast ─────────────────────────────────────────────────────────────────

function showToast(message, type) {
    type = type || 'info';
    var container = document.getElementById('toast-container');
    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function () {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(function () { toast.remove(); }, 300);
    }, 4000);
}

// ── API ───────────────────────────────────────────────────────────────────

var API = {
    async uploadFiles(formData) {
        var resp = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData,
        });
        if (!resp.ok) {
            var err = await resp.json().catch(function () { return { detail: 'Upload failed' }; });
            throw new Error(err.detail || 'Upload failed');
        }
        return resp.json();
    },

    async pollTask(taskId) {
        var resp = await fetch('/api/documents/tasks/' + taskId);
        if (!resp.ok) throw new Error('Task not found');
        return resp.json();
    },

    async getFiles() {
        var resp = await fetch('/api/documents/files');
        return resp.json();
    },

    async getCourses() {
        var resp = await fetch('/api/documents/courses');
        return resp.json();
    },

    async clearAll() {
        var resp = await fetch('/api/documents/clear', { method: 'POST' });
        return resp.json();
    },

    async renameCourse(currentName, newName) {
        var resp = await fetch('/api/documents/courses/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ current_name: currentName, new_name: newName }),
        });
        return resp.json();
    },

    async renameSection(courseName, currentSection, newSection) {
        var resp = await fetch('/api/documents/sections/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                course_name: courseName,
                current_section: currentSection,
                new_section: newSection,
            }),
        });
        return resp.json();
    },

    // ── Session API ──────────────────────────────────────────────────

    async listSessions(courseName) {
        var url = '/api/sessions?course_name=' + encodeURIComponent(courseName || '');
        var resp = await fetch(url);
        if (!resp.ok) throw new Error('Failed to list sessions');
        return resp.json();
    },

    async createSession(courseName) {
        var resp = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ course_name: courseName || '' }),
        });
        if (!resp.ok) throw new Error('Failed to create session');
        return resp.json();
    },

    async deleteSession(sessionId) {
        var resp = await fetch('/api/sessions/' + encodeURIComponent(sessionId), {
            method: 'DELETE',
        });
        if (!resp.ok) throw new Error('Failed to delete session');
        return resp.json();
    },

    async getSessionTurns(sessionId) {
        var resp = await fetch('/api/sessions/' + encodeURIComponent(sessionId) + '/turns');
        if (!resp.ok) throw new Error('Failed to load session turns');
        return resp.json();
    },

    async clearChat(sessionId) {
        var resp = await fetch('/api/chat/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId }),
        });
        return resp.json();
    },

    streamChat(body, handlers) {
        var controller = new AbortController();
        AppState.chatAbortController = controller;
        fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: controller.signal,
        }).then(function (response) {
            if (!response.ok) {
                handlers.onError('Server error: ' + response.status);
                return;
            }
            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';

            function read() {
                reader.read().then(function (result) {
                    if (result.done) { return; }
                    buffer += decoder.decode(result.value, { stream: true });
                    var lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    var eventType = null;
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];
                        if (line.startsWith('event: ')) {
                            eventType = line.slice(7).trim();
                        } else if (line.startsWith('data: ') && eventType) {
                            try {
                                var data = JSON.parse(line.slice(6));
                                dispatchEvent(eventType, data, handlers);
                            } catch (e) {
                                console.warn('Failed to parse SSE data:', e);
                            }
                            eventType = null;
                        }
                    }
                    read();
                }).catch(function (err) {
                    if (err.name !== 'AbortError') {
                        handlers.onError('Stream error: ' + err.message);
                    }
                });
            }
            read();
        }).catch(function (err) {
            if (err.name !== 'AbortError') {
                handlers.onError('Connection error: ' + err.message);
            }
        });
    },
};

function dispatchEvent(type, data, handlers) {
    switch (type) {
        case 'messages':
            if (handlers.onMessages) handlers.onMessages(data);
            break;
        case 'done':
            if (handlers.onDone) handlers.onDone();
            break;
        case 'error':
            if (handlers.onError) handlers.onError(data.message);
            break;
    }
}

// ── Documents Tab ─────────────────────────────────────────────────────────

var DocumentsTab = {
    fileInput: null,
    uploadZone: null,
    filePreviewList: null,
    uploadPlaceholder: null,

    init: function () {
        this.fileInput = document.getElementById('file-input');
        this.uploadZone = document.getElementById('upload-zone');
        this.filePreviewList = document.getElementById('file-preview-list');
        this.uploadPlaceholder = document.getElementById('upload-placeholder');

        this.setupUploadZone();
        document.getElementById('add-btn').addEventListener('click', this.handleUpload.bind(this));
        document.getElementById('refresh-btn').addEventListener('click', this.refreshAll.bind(this));
        document.getElementById('clear-btn').addEventListener('click', this.handleClear.bind(this));
        document.getElementById('rename-course-btn').addEventListener('click', this.handleRenameCourse.bind(this));
        document.getElementById('rename-section-btn').addEventListener('click', this.handleRenameSection.bind(this));
    },

    setupUploadZone: function () {
        var self = this;
        this.uploadZone.addEventListener('click', function () { self.fileInput.click(); });
        this.fileInput.addEventListener('change', function () { self.onFilesSelected(); });

        this.uploadZone.addEventListener('dragover', function (e) {
            e.preventDefault();
            self.uploadZone.classList.add('drag-over');
        });
        this.uploadZone.addEventListener('dragleave', function () {
            self.uploadZone.classList.remove('drag-over');
        });
        this.uploadZone.addEventListener('drop', function (e) {
            e.preventDefault();
            self.uploadZone.classList.remove('drag-over');
            var dt = e.dataTransfer;
            if (dt && dt.files.length > 0) {
                self.addFiles(dt.files);
            }
        });
    },

    onFilesSelected: function () {
        if (this.fileInput.files.length > 0) {
            this.addFiles(this.fileInput.files);
            this.fileInput.value = '';
        }
    },

    addFiles: function (fileList) {
        for (var i = 0; i < fileList.length; i++) {
            AppState.pendingFiles.push(fileList[i]);
        }
        this.renderFilePreview();
    },

    renderFilePreview: function () {
        if (AppState.pendingFiles.length === 0) {
            this.filePreviewList.classList.add('hidden');
            this.uploadPlaceholder.style.display = '';
            return;
        }
        this.filePreviewList.classList.remove('hidden');
        this.uploadPlaceholder.style.display = 'none';
        var html = '';
        for (var i = 0; i < AppState.pendingFiles.length; i++) {
            html += '<li>' + AppState.pendingFiles[i].name +
                ' <span class="remove-file" data-index="' + i + '">&times;</span></li>';
        }
        this.filePreviewList.innerHTML = html;

        var self = this;
        var removeBtns = this.filePreviewList.querySelectorAll('.remove-file');
        removeBtns.forEach(function (btn) {
            btn.addEventListener('click', function (e) {
                e.stopPropagation();
                var idx = parseInt(this.getAttribute('data-index'));
                AppState.pendingFiles.splice(idx, 1);
                self.renderFilePreview();
            });
        });
    },

    handleUpload: async function () {
        if (AppState.pendingFiles.length === 0) {
            showToast('Please select files to upload', 'warning');
            return;
        }
        var courseNames = document.getElementById('course-names-input').value.trim();

        var formData = new FormData();
        for (var i = 0; i < AppState.pendingFiles.length; i++) {
            formData.append('files', AppState.pendingFiles[i]);
        }
        formData.append('course_names', courseNames);

        var addBtn = document.getElementById('add-btn');
        addBtn.disabled = true;
        addBtn.textContent = 'Uploading...';

        try {
            var result = await API.uploadFiles(formData);
            AppState.pendingFiles = [];
            this.renderFilePreview();
            await this.pollProgress(result.task_id);
        } catch (e) {
            showToast(e.message, 'error');
        } finally {
            addBtn.disabled = false;
            addBtn.textContent = 'Add Documents';
        }
    },

    pollProgress: async function (taskId) {
        var progressWrap = document.getElementById('upload-progress-wrap');
        var progressBar = document.getElementById('upload-progress-bar');
        var progressLabel = document.getElementById('upload-progress-label');

        progressWrap.classList.remove('hidden');

        while (true) {
            try {
                var task = await API.pollTask(taskId);
                progressBar.style.width = (task.progress * 100) + '%';
                progressLabel.textContent = task.description || '';

                if (task.status === 'completed') {
                    var r = task.result;
                    showToast('Done: ' + r.added + ' added, ' + r.skipped + ' skipped, ' + r.failed + ' failed', 'success');
                    break;
                } else if (task.status === 'failed') {
                    showToast('Upload failed: ' + (task.error || 'unknown error'), 'error');
                    break;
                }
            } catch (e) {
                showToast('Progress check failed: ' + e.message, 'error');
                break;
            }
            await new Promise(function (r) { setTimeout(r, 500); });
        }

        progressWrap.classList.add('hidden');
        await this.refreshAll();
    },

    refreshAll: async function () {
        try {
            var results = await Promise.all([API.getFiles(), API.getCourses()]);
            var files = results[0];
            var courses = results[1];
            document.getElementById('file-list').textContent = files.files.length > 0 ? files.files.join('\n') : 'No documents.';
            document.getElementById('course-list').textContent = courses.formatted || 'No courses.';
            AppState.courseChoices = courses.choices || [];
            this.updateCourseDropdowns();
        } catch (e) {
            showToast('Failed to refresh: ' + e.message, 'error');
        }
    },

    handleClear: async function () {
        if (!confirm('Delete all documents, vectors, and courses? This cannot be undone.')) return;
        try {
            await API.clearAll();
            showToast('All documents cleared', 'success');
            await this.refreshAll();
        } catch (e) {
            showToast('Clear failed: ' + e.message, 'error');
        }
    },

    handleRenameCourse: async function () {
        var currentName = document.getElementById('edit-course-dropdown').value;
        var newName = document.getElementById('new-course-name').value.trim();
        if (!currentName) { showToast('Select a course first', 'warning'); return; }
        if (!newName) { showToast('Enter a new course name', 'warning'); return; }
        try {
            var r = await API.renameCourse(currentName, newName);
            if (r.success) {
                showToast('Course renamed', 'success');
                document.getElementById('new-course-name').value = '';
                await this.refreshAll();
            } else {
                showToast(r.error || 'Rename failed', 'error');
            }
        } catch (e) {
            showToast('Rename failed: ' + e.message, 'error');
        }
    },

    handleRenameSection: async function () {
        var courseName = document.getElementById('edit-course-dropdown').value;
        var currentSection = document.getElementById('section-current-name').value.trim();
        var newSection = document.getElementById('section-new-name').value.trim();
        if (!courseName) { showToast('Select a course first', 'warning'); return; }
        if (!currentSection) { showToast('Enter the current section name', 'warning'); return; }
        if (!newSection) { showToast('Enter a new section name', 'warning'); return; }
        try {
            var r = await API.renameSection(courseName, currentSection, newSection);
            if (r.success) {
                showToast('Section renamed', 'success');
                document.getElementById('section-current-name').value = '';
                document.getElementById('section-new-name').value = '';
                await this.refreshAll();
            } else {
                showToast(r.error || 'Rename failed', 'error');
            }
        } catch (e) {
            showToast('Rename failed: ' + e.message, 'error');
        }
    },

    updateCourseDropdowns: function () {
        var dropdowns = [
            document.getElementById('edit-course-dropdown'),
            document.getElementById('chat-course-dropdown'),
        ];
        for (var d = 0; d < dropdowns.length; d++) {
            var select = dropdowns[d];
            if (!select) continue;
            var currentVal = select.value;
            var html = '';
            if (select.id === 'chat-course-dropdown') {
                html += '<option value="">All courses</option>';
            } else {
                html += '<option value="">-- Select course --</option>';
            }
            for (var i = 0; i < AppState.courseChoices.length; i++) {
                var c = AppState.courseChoices[i];
                html += '<option value="' + c + '">' + c + '</option>';
            }
            select.innerHTML = html;
            select.value = currentVal;
        }
    },
};

// ── Session manager ───────────────────────────────────────────────────────

var SessionManager = {
    init: function () {
        document.getElementById('chat-course-dropdown').addEventListener('change', this.onCourseChange.bind(this));
        document.getElementById('chat-session-dropdown').addEventListener('change', this.onSessionChange.bind(this));
        document.getElementById('chat-new-session-btn').addEventListener('click', this.handleNewSession.bind(this));
        document.getElementById('chat-delete-session-btn').addEventListener('click', this.handleDeleteSession.bind(this));
    },

    getCurrentCourse: function () {
        return document.getElementById('chat-course-dropdown').value || '';
    },

    onCourseChange: async function () {
        await this.loadSessions();
    },

    onSessionChange: async function () {
        var sessionId = document.getElementById('chat-session-dropdown').value;
        if (!sessionId) return;
        AppState.sessionId = sessionId;
        await this.loadSessionHistory(sessionId);
    },

    loadSessions: async function () {
        var courseName = this.getCurrentCourse();
        try {
            var result = await API.listSessions(courseName);
            AppState.sessions = result.sessions || [];

            if (AppState.sessions.length === 0) {
                // Auto-create a session for this course
                await this.autoCreateSession(courseName);
                return;
            }

            this.renderSessionDropdown();

            // Select the most recent session (first in list)
            var firstId = AppState.sessions[0].id;
            document.getElementById('chat-session-dropdown').value = firstId;
            AppState.sessionId = firstId;
            await this.loadSessionHistory(firstId);
        } catch (e) {
            showToast('Failed to load sessions: ' + e.message, 'error');
        }
    },

    autoCreateSession: async function (courseName) {
        try {
            var result = await API.createSession(courseName);
            AppState.sessionId = result.session_id;
            // Reload sessions to get the full list
            var listResult = await API.listSessions(courseName);
            AppState.sessions = listResult.sessions || [];
            this.renderSessionDropdown();
            // Ensure the new session is selected
            document.getElementById('chat-session-dropdown').value = AppState.sessionId;
            ChatTab.clearMessages();
        } catch (e) {
            showToast('Failed to create session: ' + e.message, 'error');
        }
    },

    renderSessionDropdown: function () {
        var select = document.getElementById('chat-session-dropdown');
        var currentVal = select.value;
        var html = '';
        for (var i = 0; i < AppState.sessions.length; i++) {
            var s = AppState.sessions[i];
            var label = s.title || 'New Session';
            html += '<option value="' + s.id + '">' + escapeHtml(label) + '</option>';
        }
        if (!html) {
            html = '<option value="">No sessions</option>';
        }
        select.innerHTML = html;
        // Restore selection if the current session still exists
        if (currentVal && AppState.sessions.some(function (s) { return s.id === currentVal; })) {
            select.value = currentVal;
        }
    },

    refreshDropdownOnly: async function () {
        var courseName = this.getCurrentCourse();
        try {
            var result = await API.listSessions(courseName);
            AppState.sessions = result.sessions || [];
            this.renderSessionDropdown();
        } catch (e) {
            console.warn('Failed to refresh session dropdown:', e);
        }
    },

    loadSessionHistory: async function (sessionId) {
        try {
            var result = await API.getSessionTurns(sessionId);
            var turns = result.turns || [];
            ChatTab.clearMessages();
            for (var i = 0; i < turns.length; i++) {
                var t = turns[i];
                ChatTab.addMessage('user', t.user_original);
                ChatTab.addMessage('assistant', t.assistant_final);
            }
        } catch (e) {
            // Non-critical: chat still works without history
            console.warn('Failed to load session history:', e);
        }
    },

    handleNewSession: async function () {
        var courseName = this.getCurrentCourse();
        try {
            var result = await API.createSession(courseName);
            AppState.sessionId = result.session_id;
            // Reload sessions
            var listResult = await API.listSessions(courseName);
            AppState.sessions = listResult.sessions || [];
            this.renderSessionDropdown();
            document.getElementById('chat-session-dropdown').value = AppState.sessionId;
            ChatTab.clearMessages();
            showToast('New session created', 'info');
        } catch (e) {
            showToast('Failed to create session: ' + e.message, 'error');
        }
    },

    handleDeleteSession: async function () {
        var sessionId = AppState.sessionId;
        if (!sessionId) { showToast('No session to delete', 'warning'); return; }
        if (AppState.sessions.length <= 1) {
            showToast('Cannot delete the only session', 'warning');
            return;
        }
        if (!confirm('Delete this session and all its messages?')) return;

        try {
            await API.deleteSession(sessionId);
            // Reload sessions
            var courseName = this.getCurrentCourse();
            var listResult = await API.listSessions(courseName);
            AppState.sessions = listResult.sessions || [];

            if (AppState.sessions.length === 0) {
                await this.autoCreateSession(courseName);
            } else {
                this.renderSessionDropdown();
                AppState.sessionId = AppState.sessions[0].id;
                document.getElementById('chat-session-dropdown').value = AppState.sessionId;
                await this.loadSessionHistory(AppState.sessionId);
            }
            showToast('Session deleted', 'info');
        } catch (e) {
            showToast('Failed to delete session: ' + e.message, 'error');
        }
    },
};

function formatDate(isoStr) {
    if (!isoStr) return '';
    try {
        var d = new Date(isoStr);
        var month = ('0' + (d.getMonth() + 1)).slice(-2);
        var day = ('0' + d.getDate()).slice(-2);
        var hours = ('0' + d.getHours()).slice(-2);
        var mins = ('0' + d.getMinutes()).slice(-2);
        return d.getFullYear() + '-' + month + '-' + day + ' ' + hours + ':' + mins;
    } catch (e) {
        return isoStr;
    }
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Chat Tab ───────────────────────────────────────────────────────────────

var ChatTab = {
    loading: false,
    done: false,
    msgEls: [],

    init: function () {
        document.getElementById('chat-form').addEventListener('submit', this.handleSend.bind(this));
        document.getElementById('chat-clear-btn').addEventListener('click', this.handleClear.bind(this));
    },

    handleSend: function (e) {
        e.preventDefault();
        if (this.loading) return;
        if (!AppState.sessionId) {
            showToast('No session selected', 'warning');
            return;
        }
        var input = document.getElementById('chat-input');
        var message = input.value.trim();
        if (!message) return;

        input.value = '';
        this.loading = true;
        this.done = false;
        var sendBtn = document.getElementById('chat-send-btn');
        sendBtn.disabled = true;

        var placeholder = document.getElementById('chat-placeholder');
        if (placeholder) placeholder.style.display = 'none';

        this.addMessage('user', message);
        this.msgEls = [];

        var self = this;
        var courseDropdown = document.getElementById('chat-course-dropdown');

        API.streamChat({
            message: message,
            history: this.getHistory(),
            course_name: courseDropdown.value || null,
            session_id: AppState.sessionId,
        }, {
            onMessages: function (messages) { self.renderMessages(messages); },
            onDone: function () { self.handleDone(); },
            onError: function (msg) { self.handleError(msg); },
        });
    },

    renderMessages: function (messages) {
        var container = document.getElementById('chat-messages');
        // Remove message elements beyond the current message count
        while (this.msgEls.length > messages.length) {
            var extra = this.msgEls.pop();
            if (extra && extra.parentNode) extra.remove();
        }
        // Update or create elements
        for (var i = 0; i < messages.length; i++) {
            var msg = messages[i];
            var md = msg.metadata || {};
            var content = msg.content || '';
            var el = this.msgEls[i];

            if (md.node && (md.node === 'recognize_intent' || md.node === 'rewrite_query' ||
                md.node === 'plan_rag_tasks' || md.node === 'summarize_history')) {
                // System node card
                if (!el || !el.classList.contains('system-node')) {
                    if (el && el.parentNode) el.remove();
                    el = this.makeSystemNodeCard(md.title || md.node);
                    this.msgEls[i] = el;
                }
                el.querySelector('.node-content').innerHTML = renderMarkdown(content);

            } else if (md.node === 'clarification') {
                // Clarification message
                if (!el || !el.classList.contains('clarification')) {
                    if (el && el.parentNode) el.remove();
                    el = this.makeBubble('clarification');
                    this.msgEls[i] = el;
                }
                el.querySelector('.message-content').textContent = content;

            } else if (md.title && md.title.indexOf('🛠') === 0) {
                // Tool call/result card
                if (!el || !el.classList.contains('tool')) {
                    if (el && el.parentNode) el.remove();
                    el = this.makeToolCard(md.title);
                    this.msgEls[i] = el;
                }
                if (content.indexOf('Running') === 0) {
                    // Tool call - still running
                    var status = el.querySelector('.tool-status');
                    if (status) { status.textContent = 'running...'; status.className = 'tool-status'; }
                    el.querySelector('.tool-content').textContent = 'Loading...';
                } else {
                    // Tool result
                    var status = el.querySelector('.tool-status');
                    if (status) { status.textContent = '(done)'; status.className = 'tool-status tool-done'; }
                    el.querySelector('.tool-content').textContent = content;
                }

            } else if (msg.role === 'assistant') {
                // Plain assistant message (streaming answer)
                if (!el || el.classList.contains('system-node') || el.classList.contains('tool') ||
                    el.classList.contains('clarification')) {
                    if (el && el.parentNode) el.remove();
                    el = this.makeBubble('assistant');
                    this.msgEls[i] = el;
                }
                if (!this.done) {
                    // Raw text while streaming
                    el.querySelector('.message-content').textContent = content;
                }
            }
        }
        this.scrollToBottom();
    },

    handleDone: function () {
        this.done = true;
        // Re-render the last assistant message with markdown
        for (var i = this.msgEls.length - 1; i >= 0; i--) {
            var el = this.msgEls[i];
            if (el && el.classList.contains('assistant')) {
                var contentEl = el.querySelector('.message-content');
                if (contentEl) {
                    contentEl.innerHTML = renderMarkdown(contentEl.textContent || '');
                }
                break;
            }
        }
        this.loading = false;
        document.getElementById('chat-send-btn').disabled = false;
        document.getElementById('chat-input').focus();

        SessionManager.refreshDropdownOnly();
        var self = this;
        setTimeout(function () {
            SessionManager.refreshDropdownOnly();
        }, 3000);
    },

    handleError: function (msg) {
        this.loading = false;
        this.done = true;
        document.getElementById('chat-send-btn').disabled = false;
        this.addMessage('assistant', '❌ ' + msg);
        showToast(msg, 'error');
    },

    makeBubble: function (role) {
        var div = document.createElement('div');
        div.className = 'message ' + role;
        var span = document.createElement('span');
        span.className = 'message-content';
        div.appendChild(span);
        document.getElementById('chat-messages').appendChild(div);
        return div;
    },

    makeSystemNodeCard: function (title) {
        var div = document.createElement('div');
        div.className = 'message system-node';
        var details = document.createElement('details');
        details.open = true;
        var summary = document.createElement('summary');
        summary.textContent = title;
        var nodeContent = document.createElement('div');
        nodeContent.className = 'node-content';
        details.appendChild(summary);
        details.appendChild(nodeContent);
        div.appendChild(details);
        document.getElementById('chat-messages').appendChild(div);
        return div;
    },

    makeToolCard: function (title) {
        var div = document.createElement('div');
        div.className = 'message tool';
        div.innerHTML =
            '<details open>' +
            '<summary>' + title + ' <span class="tool-status">running...</span></summary>' +
            '<div class="tool-content">Loading...</div>' +
            '</details>';
        document.getElementById('chat-messages').appendChild(div);
        return div;
    },

    addMessage: function (role, content) {
        var el = this.makeBubble(role);
        if (role === 'assistant' && content) {
            el.querySelector('.message-content').innerHTML = renderMarkdown(content);
        } else {
            el.querySelector('.message-content').textContent = content;
        }
        return el;
    },

    clearMessages: function () {
        var container = document.getElementById('chat-messages');
        container.innerHTML =
            '<div class="chat-placeholder" id="chat-placeholder">' +
            '<strong>Ask me anything!</strong>' +
            '<em>I\'ll search, reason, and act to give you the best answer :)</em>' +
            '</div>';
        this.msgEls = [];
        this.done = false;
    },

    scrollToBottom: function () {
        var container = document.getElementById('chat-messages');
        container.scrollTop = container.scrollHeight;
    },

    getHistory: function () {
        var messages = document.getElementById('chat-messages').querySelectorAll(
            '.message.user, .message.assistant, .message.clarification'
        );
        var history = [];
        messages.forEach(function (msg) {
            var role = msg.classList.contains('user') ? 'user' : 'assistant';
            var contentEl = msg.querySelector('.message-content');
            if (contentEl) {
                history.push({ role: role, content: contentEl.textContent });
            }
        });
        return history;
    },

    handleClear: async function () {
        if (!AppState.sessionId) return;
        try {
            await API.clearChat(AppState.sessionId);
            this.clearMessages();
            showToast('Session cleared', 'info');
        } catch (e) {
            showToast('Clear session failed: ' + e.message, 'error');
        }
    },
};

// ── Init ───────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function () {
    var tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            var tabName = this.getAttribute('data-tab');
            tabBtns.forEach(function (b) { b.classList.remove('active'); });
            this.classList.add('active');
            document.querySelectorAll('.tab-panel').forEach(function (p) { p.classList.remove('active'); });
            document.getElementById('tab-' + tabName).classList.add('active');
            AppState.activeTab = tabName;
        });
    });

    DocumentsTab.init();
    SessionManager.init();
    ChatTab.init();

    // Load courses, then sessions for the initially selected course
    DocumentsTab.refreshAll().then(function () {
        return SessionManager.loadSessions();
    });
});
