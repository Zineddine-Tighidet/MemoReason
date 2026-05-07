/**
 * API client for the annotation backend v2.
 * All document endpoints now use theme-based URLs; user comes from session cookie.
 */
const API = {
    base: '/api/v1',

    async _sleep(ms) {
        await new Promise((resolve) => setTimeout(resolve, ms));
    },

    async _fetch(path, options = {}, attempt = 0) {
        const url = this.base + path;
        const method = String(options.method || 'GET').toUpperCase();
        const canRetry = method === 'GET' && attempt < 1;
        let res;
        try {
            res = await fetch(url, {
                credentials: 'include',
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options,
            });
        } catch (err) {
            if (canRetry) {
                await this._sleep(350);
                return this._fetch(path, options, attempt + 1);
            }
            throw err;
        }
        if (res.status === 401 && canRetry) {
            await this._sleep(350);
            return this._fetch(path, options, attempt + 1);
        }
        if (res.status === 401) {
            window.location.href = '/login';
            throw new Error('Session expired');
        }
        if (canRetry && res.status >= 500) {
            await this._sleep(350);
            return this._fetch(path, options, attempt + 1);
        }
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        return res.json();
    },

    // Auth
    login(username, password) {
        return this._fetch('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });
    },
    logout() {
        return this._fetch('/auth/logout', { method: 'POST' });
    },
    me() {
        return this._fetch('/auth/me');
    },
    createUser(username, password, role) {
        return this._fetch('/auth/users', {
            method: 'POST',
            body: JSON.stringify({ username, password, role }),
        });
    },
    listUsers() {
        return this._fetch('/auth/users');
    },

    // Themes & documents
    listThemes() {
        return this._fetch('/themes');
    },
    listThemeDocuments(theme) {
        return this._fetch(`/themes/${encodeURIComponent(theme)}/documents`);
    },

    // Documents (theme-based, user from session)
    loadDocument(theme, docId, { reviewTarget = null } = {}) {
        const params = new URLSearchParams();
        if (reviewTarget) params.set('review_target', String(reviewTarget));
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}${suffix}`);
    },
    loadEditorBootstrap(theme, docId, { reviewTarget = null } = {}) {
        const params = new URLSearchParams();
        if (reviewTarget) params.set('review_target', String(reviewTarget));
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/bootstrap${suffix}`);
    },
    loadReferenceBootstrap(theme, docId, { reviewTarget = null, referenceReviewer = null } = {}) {
        const params = new URLSearchParams();
        if (reviewTarget) params.set('review_target', String(reviewTarget));
        if (referenceReviewer) params.set('reference_reviewer', String(referenceReviewer));
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/reference-bootstrap${suffix}`);
    },
    loadSourceDocument(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/source`);
    },
    saveDocument(theme, docId, data, { reviewTarget = null } = {}) {
        const params = new URLSearchParams();
        if (reviewTarget) params.set('review_target', String(reviewTarget));
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}${suffix}`, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    },
    loadReviewDocument(reviewType, theme, docId) {
        return this._fetch(`/review-campaigns/${encodeURIComponent(reviewType)}/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}`);
    },
    saveReviewDocument(reviewType, theme, docId, data) {
        return this._fetch(`/review-campaigns/${encodeURIComponent(reviewType)}/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}`, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    },
    finishReviewDocument(reviewType, theme, docId) {
        return this._fetch(`/review-campaigns/${encodeURIComponent(reviewType)}/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/finish`, {
            method: 'POST',
        });
    },
    validateDocument(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/validate`, { method: 'POST' });
    },
    extractEntities(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/entities`);
    },
    loadImplicitRules(theme, docId, docData, { resetExclusions = true } = {}) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/implicit-rules`, {
            method: 'POST',
            body: JSON.stringify({
                doc_data: docData || {},
                reset_exclusions: resetExclusions,
            }),
        });
    },
    finishDocument(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/finish`, { method: 'POST' });
    },
    reviewDocument(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/review`, { method: 'POST' });
    },
    validateStatus(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/validate-status`, { method: 'POST' });
    },
    unvalidateStatus(theme, docId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/unvalidate`, { method: 'POST' });
    },

    // Questions
    addQuestion(theme, docId, question) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/questions`, {
            method: 'POST',
            body: JSON.stringify(question),
        });
    },
    updateQuestion(theme, docId, questionId, question) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/questions/${encodeURIComponent(questionId)}`, {
            method: 'PUT',
            body: JSON.stringify(question),
        });
    },
    deleteQuestion(theme, docId, questionId) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/questions/${encodeURIComponent(questionId)}`, {
            method: 'DELETE',
        });
    },

    // Rules
    updateRules(theme, docId, rules) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/rules`, {
            method: 'PUT',
            body: JSON.stringify(rules),
        });
    },

    // Fictional generation
    generateFictional(theme, docId, seed = 42) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/generate-fictional`, {
            method: 'POST',
            body: JSON.stringify({ seed }),
        });
    },
    generateFictionalBatch(theme, docId, {
        seed = 23,
        targetVersionCount = 10,
        maxAttempts = 60,
        seedStride = 1000003,
    } = {}) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/generate-fictional-batch`, {
            method: 'POST',
            body: JSON.stringify({
                seed,
                target_version_count: targetVersionCount,
                max_attempts: maxAttempts,
                seed_stride: seedStride,
            }),
        });
    },
    getGroqPlaygroundModels() {
        return this._fetch('/playground/groq/models');
    },
    runGroqPlayground(theme, docId, payload) {
        return this._fetch(`/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/playground/groq/run`, {
            method: 'POST',
            body: JSON.stringify(payload || {}),
        });
    },

    // Taxonomy & Progress
    getTaxonomy() {
        return this._fetch('/taxonomy');
    },
    getProgress() {
        return this._fetch('/progress');
    },
    getDashboardBootstrap(scope = 'documents') {
        const q = new URLSearchParams({ scope: String(scope || 'documents') });
        return this._fetch(`/workflow/dashboard-bootstrap?${q.toString()}`);
    },
    getMyQueue() {
        return this._fetch('/workflow/my-queue');
    },
    assignRandomMyQueueTask() {
        return this._fetch('/workflow/my-queue/assign-random', { method: 'POST' });
    },
    getMyReviewQueue(reviewType) {
        return this._fetch(`/workflow/my-review-queues/${encodeURIComponent(reviewType)}`);
    },
    assignRandomMyReviewQueueTask(reviewType) {
        return this._fetch(`/workflow/my-review-queues/${encodeURIComponent(reviewType)}/assign-random`, { method: 'POST' });
    },
    getMyAgreements() {
        return this._fetch('/workflow/my-agreements');
    },
    getMyResolutionFeedback() {
        return this._fetch('/workflow/my-resolution-feedback');
    },
    getMyReviewResolutionFeedback(reviewType) {
        return this._fetch(`/workflow/my-review-resolution-feedback/${encodeURIComponent(reviewType)}`);
    },
    submitMyResolutionFeedbackDecision(theme, docId, responseStatus) {
        return this._fetch(`/workflow/my-resolution-feedback/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/decision`, {
            method: 'POST',
            body: JSON.stringify({ response_status: responseStatus }),
        });
    },
    submitMyReviewResolutionFeedbackDecision(reviewType, theme, docId, responseStatus) {
        return this._fetch(
            `/workflow/my-review-resolution-feedback/${encodeURIComponent(reviewType)}/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/decision`,
            {
                method: 'POST',
                body: JSON.stringify({ response_status: responseStatus }),
            }
        );
    },
    getMyAgreementPacket(theme, docId) {
        return this._fetch(`/workflow/my-agreements/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/packet`);
    },
    resolveMyAgreement(theme, docId, finalVariant = null) {
        return this._fetch(`/workflow/my-agreements/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/resolve`, {
            method: 'POST',
            body: JSON.stringify(finalVariant ? { final_variant: finalVariant } : {}),
        });
    },
    getWorkflowMonitor() {
        return this._fetch('/workflow/admin/monitor');
    },
    getAdminSubmissions() {
        return this._fetch('/workflow/admin/submissions');
    },
    getAdminSubmissionSummary(theme, docId) {
        return this._fetch(`/workflow/admin/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/summary`);
    },
    getAdminReviewSubmissionSummary(reviewType, theme, docId) {
        return this._fetch(
            `/workflow/admin/review-campaigns/${encodeURIComponent(reviewType)}/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/summary`
        );
    },
    getAdminSubmissionContent(theme, docId, variant) {
        const q = new URLSearchParams({ variant: String(variant || '') }).toString();
        return this._fetch(`/workflow/admin/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}?${q}`);
    },
    getAdminReviewSubmissionContent(reviewType, theme, docId, variant) {
        const q = new URLSearchParams({ variant: String(variant || '') }).toString();
        return this._fetch(
            `/workflow/admin/review-campaigns/${encodeURIComponent(reviewType)}/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}?${q}`
        );
    },
    setAdminFinalSubmission(theme, docId, sourceVariant) {
        return this._fetch(`/workflow/admin/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/final`, {
            method: 'POST',
            body: JSON.stringify({ source_variant: sourceVariant }),
        });
    },
    setAdminFinalFromEditor(theme, docId, document, sourceLabel = 'admin') {
        return this._fetch(`/workflow/admin/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/final-from-editor`, {
            method: 'POST',
            body: JSON.stringify({ document, source_label: sourceLabel }),
        });
    },
    setAdminReviewFinalFromEditor(reviewType, theme, docId, document, sourceLabel = 'admin') {
        return this._fetch(
            `/workflow/admin/review-campaigns/${encodeURIComponent(reviewType)}/submissions/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/final-from-editor`,
            {
                method: 'POST',
                body: JSON.stringify({ document, source_label: sourceLabel }),
            }
        );
    },
    completeAdminReviewFromEditor(reviewType, theme, docId, document, sourceLabel = 'admin_completed') {
        return this._fetch(
            `/workflow/admin/review-campaigns/${encodeURIComponent(reviewType)}/documents/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/complete-from-editor`,
            {
                method: 'POST',
                body: JSON.stringify({ document, source_label: sourceLabel }),
            }
        );
    },
    resolveReviewAgreementPacket(reviewType, campaignId, theme, docId, finalVariant = null, completionMode = '') {
        const body = {};
        if (finalVariant) body.final_variant = finalVariant;
        if (completionMode) body.completion_mode = completionMode;
        return this._fetch(
            `/workflow/admin/review-campaigns/${encodeURIComponent(reviewType)}/agreements/${encodeURIComponent(campaignId)}/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/resolve`,
            {
                method: 'POST',
                body: JSON.stringify(body),
            }
        );
    },
    createWorkflowRun(payload) {
        return this._fetch('/workflow/admin/runs', {
            method: 'POST',
            body: JSON.stringify(payload || {}),
        });
    },
    getReviewCampaigns() {
        return this._fetch('/workflow/admin/review-campaigns');
    },
    createReviewCampaign(payload) {
        return this._fetch('/workflow/admin/review-campaigns', {
            method: 'POST',
            body: JSON.stringify(payload || {}),
        });
    },
    completeReviewCampaign(campaignId) {
        return this._fetch(`/workflow/admin/review-campaigns/${encodeURIComponent(campaignId)}/complete`, {
            method: 'POST',
        });
    },
    completeWorkflowRun(runId) {
        return this._fetch(`/workflow/admin/runs/${encodeURIComponent(runId)}/complete`, {
            method: 'POST',
        });
    },
    getAgreementPacket(runId, theme, docId) {
        return this._fetch(
            `/workflow/admin/runs/${encodeURIComponent(runId)}/agreements/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}`
        );
    },
    resolveAgreementPacket(runId, theme, docId, finalVariant = null) {
        return this._fetch(
            `/workflow/admin/runs/${encodeURIComponent(runId)}/agreements/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/resolve`,
            {
                method: 'POST',
                body: JSON.stringify(finalVariant ? { final_variant: finalVariant } : {}),
            }
        );
    },

    // History
    getDocumentHistory(theme, docId, { reviewType = null } = {}) {
        const params = new URLSearchParams();
        if (reviewType) params.set('review_type', String(reviewType));
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/history/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}${suffix}`);
    },
    getHistorySnapshot(theme, docId, historyId) {
        return this._fetch(`/history/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/snapshot/${historyId}`);
    },
    getAnnotationVersion(theme, docId, versionKey, { reviewType = null } = {}) {
        const params = new URLSearchParams();
        if (reviewType) params.set('review_type', String(reviewType));
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return this._fetch(`/history/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/versions/${encodeURIComponent(versionKey)}${suffix}`);
    },
    getAnnotationsMetadata(theme, docId) {
        return this._fetch(`/history/${encodeURIComponent(theme)}/${encodeURIComponent(docId)}/annotations/metadata`);
    },
    getRecentHistory(scope = 'documents') {
        const normalized = String(scope || 'documents').trim().toLowerCase();
        const params = new URLSearchParams();
        params.set('scope', normalized);
        return this._fetch(`/history/recent?${params.toString()}`);
    },

    // Extended Taxonomy Management (power users only)
    getExtendedTaxonomy() {
        return this._fetch('/taxonomy/extended');
    },
    updateExtendedTaxonomy(taxonomyData) {
        return this._fetch('/taxonomy/extended', {
            method: 'PUT',
            body: JSON.stringify(taxonomyData),
        });
    },
    exportTaxonomyMarkdown() {
        return this._fetch('/taxonomy/export/markdown');
    },
    exportTaxonomyReadme() {
        // Backward-compatible alias.
        return this._fetch('/taxonomy/export/readme');
    },
    updateTaxonomyDoc() {
        return this._fetch('/taxonomy/update-doc', {
            method: 'POST',
        });
    },
    updateReadme() {
        // Backward-compatible alias.
        return this._fetch('/taxonomy/update-readme', {
            method: 'POST',
        });
    },
};
