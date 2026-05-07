/**
 * Main editor Alpine.js application (v2).
 * Single-page layout: document + entity sidebar on left, Q/R/History on right.
 */

function editorApp(theme, docId) {
    const isPowerUser = (window.CURRENT_USER_ROLE || 'regular_user') === 'power_user';
    const currentUsername = String(window.CURRENT_USER_NAME || '').trim();
    const defaultGroqSystemPrompt = 'Answer strictly from the provided document and not from prior knowledge. If the document does not determine the answer, reply exactly with Cannot be determined. Keep each answer short and standalone.';
    const qaCoverageExemptionsField = 'qa_coverage_exemptions';
    const qaRequiredQuestionTypes = ['extractive', 'arithmetic', 'inference', 'temporal'];
    const qaRequiredAnswerTypes = ['variant', 'invariant', 'refusal'];
    const refusalAnswerLiteral = 'Cannot be determined';
    const inlineAnnotationRegex = /\[([^\]]+);\s*([^\]]+)\]/g;
    const urlParams = new URLSearchParams(window.location.search || '');
    const agreementModeRequested = urlParams.get('agreement_mode') === '1';
    const contestVariantRequested = ['reviewer_a', 'reviewer_b'].includes(urlParams.get('contest_variant'))
        ? urlParams.get('contest_variant')
        : '';
    const reviewTargetRequested = ['rules', 'questions'].includes(urlParams.get('review_target'))
        ? urlParams.get('review_target')
        : '';
    const dashboardSectionRequested = ['documents', 'rules', 'questions'].includes(urlParams.get('dashboard_section'))
        ? urlParams.get('dashboard_section')
        : '';
    const referenceModeRequested = urlParams.get('reference_mode') === '1';
    const referenceReviewerRequested = String(urlParams.get('reference_reviewer') || '').trim();
    return {
        theme,
        doc_id: docId,
        isPowerUser,
        reviewTarget: reviewTargetRequested,
        dashboardReturnSection: dashboardSectionRequested || reviewTargetRequested || 'documents',
        referenceMode: referenceModeRequested,
        referenceReviewer: referenceReviewerRequested,
        docData: null,
        taxonomy: {},
        loading: true,
        loadingImplicitRules: false,
        saving: false,
        dirty: false,
        sourceView: false,
        showEntityReferences: false,
        sourceEditorText: '',
        showShortcuts: false,
        showGroqPlaygroundModal: false,
        agreementClickTimer: null,
        // Global document navigator
        allDocuments: [],
        currentDocGlobalIndex: -1,
        navigationLoading: true,

        // Entity sidebar
        entityTypes: [],
        entityGroups: {},
        expandedEntity: null,
        paintEntity: null,
        highlightedType: null,
        lockedHighlight: null,  // For click-to-lock highlighting

        // Annotation popup
        popup: {
            show: false,
            x: 0, y: 0,
            selectedText: '',
            rawStart: -1,
            rawEnd: -1,
            entityType: '',
            entityId: '',
            attribute: '',
            relationshipTarget: '',
            editing: false,
            editRef: null,
            resizingSpan: false,
        },

        // New entity dialog
        newEntityDialog: { show: false, type: '' },

        // Rule composer state
        ruleInputContent: '',
        ruleCommentAliasCacheKey: '',
        ruleCommentAliasEntries: [],

        // Resize mode state
        resizeMode: { active: false },

        // Undo stack
        undoStack: [],
        redoStack: [],

        // Right panel collapse state
        panelState: { questions: true, rules: true, agreement: true, history: true },

        groqPlayground: {
            initialized: false,
            configured: true,
            loadingModels: false,
            running: false,
            error: '',
            documentSource: 'current',
            fictionalVersionIndex: 0,
            availableModels: [],
            selectedModels: [],
            modelFilter: '',
            questions: [],
            systemPrompt: defaultGroqSystemPrompt,
            temperature: 0,
            seed: 23,
            maxTokens: 512,
            showAdvanced: false,
            results: [],
            lastRunQuestions: [],
            lastDocumentCharCount: 0,
            lastCompletedAt: '',
        },

        // Question editing: tracks which question is in edit mode
        editingQuestion: null,
        editingRule: null,
        qaInsertionTarget: {
            questionIdx: null,
            field: 'question',
        },

        // History
        historyEntries: [],
        annotationVersions: [],
        openVersionDetails: {},
        currentStatus: 'draft',
        lastEditor: null,
        historySnapshot: null,
        showHistoryModal: false,
        annotationMetadata: {annotations: {}, questions: {}, rules: {}, has_history: false},
            agreementWorkspace: {
                active: isPowerUser && agreementModeRequested,
                loading: false,
                finalizing: false,
                error: '',
                contestVariant: contestVariantRequested,
                comparison: {
                    mode: 'annotator_agreement',
                    sideAKey: 'reviewer_a',
                    sideBKey: 'reviewer_b',
                    contestVariant: '',
                },
                disableCurrentInference: false,
                awaitingReviewerAcceptance: false,
                resolveMode: true,
                inferResolutionFromCurrent: false,
                runId: null,
                runName: '',
                status: '',
                loadedVariant: 'editor',
            versions: {
                source: {
                    key: 'source',
                    label: 'Opus',
                    username: '',
                    available: false,
                    path: '',
                    document_to_annotate: '',
                    editable_document: null,
                },
                reviewer_a: {
                    key: 'reviewer_a',
                    label: 'Reviewer 1',
                    username: '',
                    available: false,
                    path: '',
                    document_to_annotate: '',
                    editable_document: null,
                },
                reviewer_b: {
                    key: 'reviewer_b',
                    label: 'Reviewer 2',
                    username: '',
                    available: false,
                    path: '',
                    document_to_annotate: '',
                    editable_document: null,
                },
                final: {
                    key: 'final',
                    label: 'Final',
                    username: '',
                    available: false,
                    path: '',
                    document_to_annotate: '',
                    editable_document: null,
                },
            },
            merge: {
                plainText: '',
                html: '',
                inlineHtml: '',
                inlineSegments: [],
                decisions: {},
                manualResolutions: {},
                manualTextResolutions: {},
                baseToCurrentMapper: null,
                inlineStats: {
                    agreed: 0,
                    total: 0,
                    resolved: 0,
                    remaining: 0,
                },
                selectedConflictId: '',
                warning: '',
                agreedCount: 0,
                conflictCount: 0,
                conflicts: [],
            },
            structured: {
                warning: '',
                decisions: {},
                manualResolutions: {},
                questionConflicts: [],
                ruleConflicts: [],
                stats: {
                    question: { total: 0, resolved: 0, remaining: 0 },
                    rule: { total: 0, resolved: 0, remaining: 0 },
                },
            },
            conflictModal: {
                show: false,
                conflictIds: [],
                conflictId: '',
                choice: '',
                manualRefs: '',
                x: 24,
                y: 96,
            },
            questionConflictModal: {
                show: false,
                conflictId: '',
                x: 24,
                y: 96,
            },
        },

        get currentStatusLabel() {
            if (this.referenceMode) {
                if (this.reviewTarget === 'rules') return 'Rules Reference';
                if (this.reviewTarget === 'questions') return 'Questions Reference';
                return 'Reference';
            }
            const rawStatus = String(this.currentStatus || 'draft')
                .replace(/_/g, ' ')
                .replace(/\b\w/g, (ch) => ch.toUpperCase());
            if (this.reviewTarget === 'rules') return `Rules Review ${rawStatus}`;
            if (this.reviewTarget === 'questions') return `Questions Review ${rawStatus}`;
            return `Doc Annotation ${rawStatus}`;
        },
        get reviewTargetLabel() {
            if (this.referenceMode) {
                if (this.reviewTarget === 'rules') return 'Rules Reference';
                if (this.reviewTarget === 'questions') return 'Questions Reference';
                return 'Reference';
            }
            if (this.reviewTarget === 'rules') return 'Rules Review';
            if (this.reviewTarget === 'questions') return 'Questions Review';
            return 'Doc Annotation';
        },
        get canEditReviewTargetWorkspace() {
            if (this.referenceMode) return false;
            if (!this.reviewTarget) return true;
            if (!isPowerUser) return true;
            if (this.agreementWorkspace.active) return true;
            return this.reviewTarget === 'rules' || this.reviewTarget === 'questions';
        },
        get canEditQuestions() {
            return this.canEditReviewTargetWorkspace && (!this.reviewTarget || this.reviewTarget === 'questions');
        },
        get canEditRules() {
            return this.canEditReviewTargetWorkspace && (!this.reviewTarget || this.reviewTarget === 'rules');
        },
        get canEditDocumentAnnotations() {
            return !this.reviewTarget || (
                this.reviewTarget === 'rules'
                && isPowerUser
                && this.agreementWorkspace.active
            );
        },
        get canSaveWorkspace() {
            return !this.reviewTarget || this.canEditReviewTargetWorkspace;
        },
        get canFinishWorkspace() {
            if (this.reviewTarget && isPowerUser && !this.agreementWorkspace.active) {
                return this.reviewTarget === 'rules' || this.reviewTarget === 'questions';
            }
            return !this.reviewTarget || this.canEditReviewTargetWorkspace;
        },
        get showFinishWorkspaceAction() {
            if ((this.reviewTarget === 'rules' || this.reviewTarget === 'questions') && isPowerUser && !this.agreementWorkspace.active) {
                return true;
            }
            return this.canFinishWorkspace && this.currentStatus !== 'completed' && this.currentStatus !== 'validated';
        },
        get showQuestionsPanel() {
            return this.reviewTarget === 'questions';
        },
        get showRulesPanel() {
            if (this.referenceMode && this.reviewTarget === 'questions') return false;
            return this.reviewTarget === 'rules' || this.reviewTarget === 'questions';
        },
        get finishButtonLabel() {
            if (this.reviewTarget === 'rules') return isPowerUser ? '✓ Finish Rules Review' : '✓ Submit Rules Review';
            if (this.reviewTarget === 'questions') return isPowerUser ? '✓ Finish Questions Review' : '✓ Submit Questions Review';
            return isPowerUser ? '✓ Finish' : '✓ Submit Annotations';
        },
        get finishButtonBusyLabel() {
            if (this.reviewTarget === 'rules') return isPowerUser ? '⏳ Finishing Rules Review...' : '⏳ Submitting Rules Review...';
            if (this.reviewTarget === 'questions') return isPowerUser ? '⏳ Finishing Questions Review...' : '⏳ Submitting Questions Review...';
            return '⏳ Saving...';
        },

        // Computed
        get renderedHtml() {
            return renderAnnotatedHtml(
                this.docData?.document_to_annotate || '',
                isPowerUser ? (this.docData?.annotated_by || null) : null,
                this.documentRenderOptions()
            );
        },
        documentRenderOptions() {
            return {
                displayMode: this.showEntityReferences ? 'ref' : 'text',
                showControls: !this.showEntityReferences,
            };
        },
        renderMainDocumentHtml() {
            const raw = this.docData?.document_to_annotate || '';
            const agreementActive = this.agreementWorkspace.active && this.agreementWorkspace.resolveMode && !this.sourceView;
            const annotatedBy = isPowerUser ? (this.docData?.annotated_by || null) : null;
            const renderOptions = this.documentRenderOptions();
            if (this.showEntityReferences && !agreementActive) {
                return renderAnnotatedHtml(raw, annotatedBy, renderOptions);
            }
            if (!agreementActive) {
                return renderAnnotatedHtml(raw, annotatedBy, renderOptions);
            }

            const unresolved = this.getAgreementUnresolvedConflicts();
            if (!Array.isArray(unresolved) || unresolved.length === 0) {
                return renderAnnotatedHtml(raw, annotatedBy, renderOptions);
            }

            const parsedCurrent = this._parseAgreementAnnotatedText(raw || '');
            const plain = String(parsedCurrent?.plainText || '');
            const plainAnnotations = Array.isArray(parsedCurrent?.annotations) ? parsedCurrent.annotations : [];
            const rawAnnotations = parseAnnotations(raw || '');
            if (!plain) return '';
            if (rawAnnotations.length !== plainAnnotations.length) {
                return renderAnnotatedHtml(raw, annotatedBy, renderOptions);
            }

            const selectedConflictId = String(this.agreementWorkspace.merge.selectedConflictId || '');
            const overlapIdsForRange = (start, end) => {
                const ids = [];
                for (const conflict of unresolved) {
                    const cStart = this._conflictCurrentStart(conflict);
                    const cEnd = this._conflictCurrentEnd(conflict);
                    if (Number(start) < cEnd && Number(end) > cStart) {
                        ids.push(conflict.id);
                    }
                }
                return ids;
            };

            const renderPlainWithConflictMarks = (start, end) => {
                if (end <= start) return '';
                const cutPoints = new Set([start, end]);
                for (const conflict of unresolved) {
                    const cStart = this._conflictCurrentStart(conflict);
                    const cEnd = this._conflictCurrentEnd(conflict);
                    if (start < cEnd && end > cStart) {
                        cutPoints.add(Math.max(start, cStart));
                        cutPoints.add(Math.min(end, cEnd));
                    }
                }
                const points = Array.from(cutPoints)
                    .filter((value) => Number.isFinite(value))
                    .sort((a, b) => a - b);

                let out = '';
                for (let idx = 0; idx < points.length - 1; idx++) {
                    const segStart = points[idx];
                    const segEnd = points[idx + 1];
                    if (segEnd <= segStart) continue;
                    const text = plain.substring(segStart, segEnd);
                    if (!text) continue;
                    const ids = overlapIdsForRange(segStart, segEnd);
                    if (ids.length === 0) {
                        out += this._escapeAgreementHtml(text);
                        continue;
                    }
                    const classes = ['agreement-conflict-mark', 'agreement-conflict-plain'];
                    if (selectedConflictId && ids.includes(selectedConflictId)) {
                        classes.push('selected');
                    }
                    const attrs = [
                        `class="${classes.join(' ')}"`,
                        `data-agreement-conflict-id="${this._escapeAgreementHtml(ids[0])}"`,
                        `data-agreement-conflict-ids="${this._escapeAgreementHtml(ids.join(','))}"`,
                    ];
                    if (ids.length > 1) {
                        attrs.push(`data-agreement-conflict-count="${ids.length}"`);
                    }
                    out += `<span ${attrs.join(' ')}>${this._escapeAgreementHtml(text)}</span>`;
                }
                return out;
            };

            const count = Math.min(rawAnnotations.length, plainAnnotations.length);
            const mergedAnnotations = [];
            for (let idx = 0; idx < count; idx++) {
                const rawAnn = rawAnnotations[idx];
                const plainAnn = plainAnnotations[idx];
                mergedAnnotations.push({
                    rawStart: Number(rawAnn.start),
                    rawEnd: Number(rawAnn.end),
                    plainStart: Number(plainAnn.start),
                    plainEnd: Number(plainAnn.end),
                    text: String(rawAnn.text || plainAnn.text || ''),
                    ref: String(rawAnn.ref || plainAnn.ref || ''),
                    entityId: String(rawAnn.entityId || plainAnn.entityId || ''),
                    entityType: String(rawAnn.entityType || plainAnn.entityType || ''),
                });
            }
            mergedAnnotations.sort((a, b) => a.plainStart - b.plainStart);

            let html = '';
            let cursor = 0;
            for (const ann of mergedAnnotations) {
                html += renderPlainWithConflictMarks(cursor, ann.plainStart);

                const overlapIds = overlapIdsForRange(ann.plainStart, ann.plainEnd);
                const typeClass = `ann-${ann.entityType}`;
                const classes = ['ann', typeClass];
                if (overlapIds.length > 0) {
                    classes.push('agreement-conflict-mark');
                    if (selectedConflictId && overlapIds.includes(selectedConflictId)) {
                        classes.push('selected');
                    }
                }
                const attrs = [
                    `class="${classes.join(' ')}"`,
                    `data-ref="${this._escapeAgreementHtml(ann.ref)}"`,
                    `data-entity-id="${this._escapeAgreementHtml(ann.entityId)}"`,
                    `data-entity-type="${this._escapeAgreementHtml(ann.entityType)}"`,
                    `data-start="${ann.rawStart}"`,
                    `data-end="${ann.rawEnd}"`,
                ];
                if (annotatedBy) {
                    attrs.push(`data-annotated-by="${this._escapeAgreementHtml(annotatedBy)}"`);
                }
                if (overlapIds.length > 0) {
                    attrs.push(`data-agreement-conflict-id="${this._escapeAgreementHtml(overlapIds[0])}"`);
                    attrs.push(`data-agreement-conflict-ids="${this._escapeAgreementHtml(overlapIds.join(','))}"`);
                    if (overlapIds.length > 1) {
                        attrs.push(`data-agreement-conflict-count="${overlapIds.length}"`);
                    }
                }

                html += `<span ${attrs.join(' ')}>`;
                html += '<span class="resize-handle left" data-side="left"></span>';
                html += '<button class="ann-delete-btn" title="Delete this annotation">×</button>';
                html += this._escapeAgreementHtml(ann.text);
                html += '<span class="resize-handle right" data-side="right"></span>';
                html += '</span>';

                cursor = ann.plainEnd;
            }

            html += renderPlainWithConflictMarks(cursor, plain.length);
            return html;
        },

        setEntityReferenceView(enabled) {
            const nextValue = !!enabled;
            if (
                nextValue
                && this.agreementWorkspace.active
                && this.agreementWorkspace.resolveMode
            ) {
                showToast('Turn off Resolve Conflicts to show entity references inline', 'info');
                return;
            }
            if (this.showEntityReferences === nextValue) return;
            this.showEntityReferences = nextValue;
            if (nextValue) {
                this.sourceView = false;
                this.paintEntity = null;
                if (this.newEntityDialog?.show) {
                    this.newEntityDialog.show = false;
                }
                if (this.popup?.show) {
                    this.popup.show = false;
                }
                const selection = window.getSelection ? window.getSelection() : null;
                if (selection && typeof selection.removeAllRanges === 'function') {
                    selection.removeAllRanges();
                }
            }
            this.$nextTick(() => {
                this.attachAnnotationHoverListeners();
                this.enhanceTooltipsWithMetadata();
            });
        },

        toggleEntityReferenceView() {
            this.setEntityReferenceView(!this.showEntityReferences);
        },

        _splitRuleString(ruleText) {
            const text = String(ruleText || '').trim();
            if (!text) return { expression: '', explanation: '' };
            const hashIndex = text.indexOf('#');
            if (hashIndex < 0) {
                return { expression: text, explanation: '' };
            }
            return {
                expression: text.slice(0, hashIndex).trim(),
                explanation: text.slice(hashIndex + 1).trim(),
            };
        },

        _composeRuleString(expressionText, explanationText) {
            const expression = String(expressionText || '').trim();
            const explanation = String(explanationText || '').trim();
            if (!expression) return '';
            if (!explanation) return expression;
            return `${expression} # ${explanation}`;
        },

        _renderRuleExpressionWithEntityRefs(expressionText) {
            const text = String(expressionText || '').trim();
            if (!text) return '';
            if (typeof window.renderAnswerExpression === 'function') {
                return window.renderAnswerExpression(text);
            }
            return this._escapeAgreementHtml(text);
        },

        renderRuleDisplayHtml(ruleText) {
            const parts = this._splitRuleString(ruleText);
            const expressionHtml = parts.expression
                ? this._renderRuleExpressionWithEntityRefs(parts.expression)
                : '';
            const explanationBodyHtml = this._renderRuleCommentWithEntityMentions(parts.explanation, false);
            const explanationLine = parts.explanation
                ? `<div class="rule-explanation-line"><strong>Explanation:</strong><em>${explanationBodyHtml}</em></div>`
                : '';

            return `<div class="rule-expression-line">${expressionHtml}</div>${explanationLine}`;
        },

        explicitRuleCount() {
            return Array.isArray(this.docData?.rules) ? this.docData.rules.length : 0;
        },

        implicitRuleCount() {
            return Array.isArray(this.docData?.implicit_rules) ? this.docData.implicit_rules.length : 0;
        },

        totalRuleCount() {
            return this.explicitRuleCount() + this.implicitRuleCount();
        },

        implicitRuleUsesIntegerBounds(rule) {
            const entityRef = String(rule?.entity_ref || '').trim();
            const ruleKind = String(rule?.rule_kind || '').trim();
            if (!entityRef) return false;
            const [entityId, attribute = ''] = entityRef.split('.', 2);
            if (['age', 'year', 'day_of_month'].includes(attribute)) return true;
            if (entityId.startsWith('number_') && ['int', 'str'].includes(attribute)) return true;
            if (ruleKind === 'century_range') return true;
            return false;
        },

        implicitRuleHasYearCap(rule) {
            const entityRef = String(rule?.entity_ref || '').trim();
            const ruleKind = String(rule?.rule_kind || '').trim();
            return entityRef.endsWith('.year') || ruleKind === 'temporal_year_range';
        },

        coerceImplicitRuleBound(rule, field, rawValue) {
            const parsedValue = Number.parseFloat(rawValue);
            if (!Number.isFinite(parsedValue)) return null;
            let nextValue = parsedValue;
            if (this.implicitRuleHasYearCap(rule)) {
                nextValue = Math.min(nextValue, 2026);
            }
            if (this.implicitRuleUsesIntegerBounds(rule)) {
                return field === 'upper_bound' ? Math.floor(nextValue) : Math.ceil(nextValue);
            }
            return Number(nextValue.toFixed(2));
        },

        implicitRuleBoundInputValue(rule, field) {
            const coercedValue = this.coerceImplicitRuleBound(rule, field, rule?.[field]);
            if (!Number.isFinite(coercedValue)) return '';
            if (this.implicitRuleUsesIntegerBounds(rule)) return String(coercedValue);
            return coercedValue.toFixed(2);
        },

        implicitRuleInputStep(rule) {
            return this.implicitRuleUsesIntegerBounds(rule) ? 1 : 0.01;
        },

        implicitRuleInputMax(rule) {
            return this.implicitRuleHasYearCap(rule) ? 2026 : null;
        },

        formatImplicitRuleBound(rule, field, rawValue) {
            const coercedValue = this.coerceImplicitRuleBound(rule, field, rawValue);
            if (!Number.isFinite(coercedValue)) return '';
            if (this.implicitRuleUsesIntegerBounds(rule)) return String(coercedValue);
            return coercedValue.toFixed(2);
        },

        implicitRuleExpression(rule) {
            const entityRef = String(rule?.entity_ref || '').trim();
            const lower = this.coerceImplicitRuleBound(rule, 'lower_bound', rule?.lower_bound);
            const upper = this.coerceImplicitRuleBound(rule, 'upper_bound', rule?.upper_bound);
            if (!entityRef || !Number.isFinite(lower) || !Number.isFinite(upper)) return '';
            const orderedLower = Math.min(lower, upper);
            const orderedUpper = Math.max(lower, upper);
            return (
                `${entityRef} ∈ [` +
                `${this.formatImplicitRuleBound(rule, 'lower_bound', orderedLower)}, ` +
                `${this.formatImplicitRuleBound(rule, 'upper_bound', orderedUpper)}]`
            );
        },

        implicitRuleExplanation(rule) {
            const entityRef = String(rule?.entity_ref || '').trim();
            const factualValue = Number(rule?.factual_value);
            const [entityId, attribute = ''] = entityRef.split('.', 2);
            const usesSmallNumberFixedWindow = (
                String(rule?.rule_kind || '').trim() === 'number_range'
                && entityId.startsWith('number_')
                && ['int', 'str'].includes(attribute)
                && Number.isFinite(factualValue)
                && Math.abs(factualValue) < 10
            );
            if (usesSmallNumberFixedWindow) {
                return 'This rule was generated following the interval of factual value +/- 3, clamped to the valid domain when needed.';
            }
            const percentage = Number(rule?.percentage);
            if (!Number.isFinite(percentage)) return '';
            const percentageLabel = Number.isInteger(percentage) ? `${percentage}%` : `${percentage.toFixed(2)}%`;
            if (this.implicitRuleHasYearCap(rule)) {
                return `This rule was generated following the range of ${percentageLabel} around factual value with an upper bound at 2026.`;
            }
            return `This rule was generated following the range of ${percentageLabel} around factual value.`;
        },

        renderImplicitRuleDisplayHtml(rule) {
            const expressionHtml = this._renderRuleExpressionWithEntityRefs(this.implicitRuleExpression(rule));
            const explanationHtml = this._escapeAgreementHtml(this.implicitRuleExplanation(rule));
            return (
                `<div class="rule-expression-line">${expressionHtml}</div>` +
                `<div class="rule-explanation-line"><em>${explanationHtml}</em></div>`
            );
        },

        onImplicitRuleBoundInput(idx, field, rawValue) {
            if (!this.canEditRules) return;
            if (!Array.isArray(this.docData?.implicit_rules)) this.docData.implicit_rules = [];
            const currentRule = this.docData.implicit_rules[idx];
            if (!currentRule || (field !== 'lower_bound' && field !== 'upper_bound')) return;
            const nextValue = this.coerceImplicitRuleBound(currentRule, field, rawValue);
            if (!Number.isFinite(nextValue)) return;
            currentRule[field] = nextValue;
            this.markDirty();
        },

        deleteImplicitRule(idx) {
            if (!this.canEditRules) return;
            if (!Array.isArray(this.docData?.implicit_rules)) this.docData.implicit_rules = [];
            const currentRule = this.docData.implicit_rules[idx];
            if (!currentRule) return;

            const entityRef = String(currentRule.entity_ref || '').trim();
            if (entityRef) {
                if (!Array.isArray(this.docData.implicit_rule_exclusions)) {
                    this.docData.implicit_rule_exclusions = [];
                }
                if (!this.docData.implicit_rule_exclusions.includes(entityRef)) {
                    this.docData.implicit_rule_exclusions.push(entityRef);
                }
            }

            this.docData.implicit_rules.splice(idx, 1);
            this.markDirty();
        },

        async loadImplicitRules({ resetExclusions = true } = {}) {
            if (!this.canEditRules || !this.docData || this.loadingImplicitRules) return;
            this.loadingImplicitRules = true;
            try {
                const payload = JSON.parse(JSON.stringify(this.docData || {}));
                if (resetExclusions) {
                    delete payload.implicit_rule_exclusions;
                }
                const result = await API.loadImplicitRules(this.theme, this.doc_id, payload, { resetExclusions });
                this.docData.implicit_rules = Array.isArray(result?.implicit_rules) ? result.implicit_rules : [];
                if (Array.isArray(result?.implicit_rule_exclusions) && result.implicit_rule_exclusions.length > 0) {
                    this.docData.implicit_rule_exclusions = result.implicit_rule_exclusions;
                } else {
                    delete this.docData.implicit_rule_exclusions;
                }
                this.markDirty();
                const count = Number(result?.count || this.docData.implicit_rules.length || 0);
                if (count > 0) {
                    showToast(`${count} implicit rule${count === 1 ? '' : 's'} loaded`, 'success');
                } else {
                    showToast('No implicit rules could be generated from the current annotations', 'warning');
                }
            } catch (e) {
                showToast('Failed to load implicit rules: ' + e.message, 'error');
            } finally {
                this.loadingImplicitRules = false;
            }
        },

        ruleExpressionValue(idx) {
            const ruleText = String(this.docData?.rules?.[idx] || '');
            return this._splitRuleString(ruleText).expression;
        },

        ruleExplanationValue(idx) {
            const ruleText = String(this.docData?.rules?.[idx] || '');
            return this._splitRuleString(ruleText).explanation;
        },

        onRuleExpressionInput(idx, nextExpression) {
            if (!this.canEditRules) return;
            if (!Array.isArray(this.docData.rules)) this.docData.rules = [];
            const currentParts = this._splitRuleString(this.docData.rules[idx] || '');
            this.docData.rules[idx] = this._composeRuleString(nextExpression, currentParts.explanation);
            this.markDirty();
        },

        onRuleExplanationInput(idx, nextExplanation) {
            if (!this.canEditRules) return;
            if (!Array.isArray(this.docData.rules)) this.docData.rules = [];
            const currentParts = this._splitRuleString(this.docData.rules[idx] || '');
            this.docData.rules[idx] = this._composeRuleString(currentParts.expression, nextExplanation);
            this.markDirty();
        },

        toggleRuleEdit(idx) {
            if (!this.canEditRules) return;
            this.editingRule = this.editingRule === idx ? null : idx;
            this.$nextTick(() => this.attachRuleReferenceHoverListeners());
        },

        _ruleCommentAliasIsEligible(annotation) {
            const entityType = String(annotation?.entityType || '').trim();
            const attribute = String(annotation?.attribute || '').trim();
            if (!entityType || !attribute) return false;
            if (['number', 'temporal'].includes(entityType)) return false;
            const excluded = new Set([
                'subj_pronoun',
                'obj_pronoun',
                'poss_det_pronoun',
                'poss_pro_pronoun',
                'refl_pronoun',
                'gender',
                'honorific',
                'age',
                'relationship',
            ]);
            return !excluded.has(attribute);
        },

        _getRuleCommentAliasEntries() {
            const raw = String(this.docData?.document_to_annotate || '');
            if (!raw) return [];
            if (this.ruleCommentAliasCacheKey === raw && Array.isArray(this.ruleCommentAliasEntries)) {
                return this.ruleCommentAliasEntries;
            }

            const parsed = parseAnnotations(raw);
            const seen = new Set();
            const out = [];
            for (const ann of parsed) {
                if (!this._ruleCommentAliasIsEligible(ann)) continue;
                const alias = String(ann?.text || '').trim();
                if (alias.length < 2) continue;
                if (/^\d+([.,:/-]\d+)*$/.test(alias)) continue;
                const entityId = String(ann?.entityId || '').trim();
                const entityType = String(ann?.entityType || '').trim();
                const ref = String(ann?.ref || '').trim();
                if (!entityId || !entityType || !ref) continue;
                const key = `${alias.toLowerCase()}::${entityId}`;
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({
                    alias,
                    aliasLower: alias.toLowerCase(),
                    entityId,
                    entityType,
                    ref,
                    attribute: String(ann?.attribute || '').trim(),
                    length: alias.length,
                });
            }
            out.sort((a, b) => b.length - a.length);
            this.ruleCommentAliasCacheKey = raw;
            this.ruleCommentAliasEntries = out;
            return out;
        },

        _isRuleCommentWordChar(ch) {
            return /[A-Za-z0-9_]/.test(String(ch || ''));
        },

        _ruleCommentMatchHasBoundary(text, start, end, alias) {
            const first = String(alias || '').charAt(0);
            const last = String(alias || '').charAt(String(alias || '').length - 1);
            if (this._isRuleCommentWordChar(first) && start > 0 && this._isRuleCommentWordChar(text.charAt(start - 1))) {
                return false;
            }
            if (this._isRuleCommentWordChar(last) && end < text.length && this._isRuleCommentWordChar(text.charAt(end))) {
                return false;
            }
            return true;
        },

        _renderRuleCommentWithEntityMentions(commentText, wrap = true) {
            const text = String(commentText || '');
            if (!text) return '';
            const aliases = this._getRuleCommentAliasEntries();
            if (!Array.isArray(aliases) || aliases.length === 0) {
                const escaped = this._escapeAgreementHtml(text);
                return wrap ? `<span class="rule-comment-text">${escaped}</span>` : escaped;
            }

            const lower = text.toLowerCase();
            const matches = [];
            for (const entry of aliases) {
                let offset = 0;
                while (offset < lower.length) {
                    const found = lower.indexOf(entry.aliasLower, offset);
                    if (found < 0) break;
                    const end = found + entry.aliasLower.length;
                    if (this._ruleCommentMatchHasBoundary(text, found, end, entry.alias)) {
                        matches.push({
                            start: found,
                            end,
                            entry,
                        });
                    }
                    offset = found + 1;
                }
            }

            if (matches.length === 0) {
                const escaped = this._escapeAgreementHtml(text);
                return wrap ? `<span class="rule-comment-text">${escaped}</span>` : escaped;
            }

            matches.sort((a, b) => {
                if (a.start !== b.start) return a.start - b.start;
                return (b.end - b.start) - (a.end - a.start);
            });

            const selected = [];
            let cursorEnd = -1;
            for (const match of matches) {
                if (match.start < cursorEnd) continue;
                selected.push(match);
                cursorEnd = match.end;
            }

            let html = '';
            let cursor = 0;
            for (const match of selected) {
                if (match.start > cursor) {
                    html += this._escapeAgreementHtml(text.slice(cursor, match.start));
                }
                const token = this._sanitizeAgreementClassToken(match.entry.entityType);
                const cls = token ? `ann ann-${token} rule-comment-entity` : 'ann rule-comment-entity';
                const titleParts = [
                    `Entity: ${match.entry.entityId}`,
                    `Type: ${match.entry.entityType}`,
                ];
                if (match.entry.attribute) titleParts.push(`Attribute: ${match.entry.attribute}`);
                const title = titleParts.join(' | ');
                html += `<span class="${cls}" data-ref="${this._escapeAgreementHtml(match.entry.ref)}" data-entity-id="${this._escapeAgreementHtml(match.entry.entityId)}" data-entity-type="${this._escapeAgreementHtml(match.entry.entityType)}" title="${this._escapeAgreementHtml(title)}">${this._escapeAgreementHtml(text.slice(match.start, match.end))}</span>`;
                cursor = match.end;
            }
            if (cursor < text.length) {
                html += this._escapeAgreementHtml(text.slice(cursor));
            }
            return wrap ? `<span class="rule-comment-text">${html}</span>` : html;
        },
        get popupEntityOptions() {
            if (!this.popup.entityType) return [];
            return this.entityGroups[this.popup.entityType] || [];
        },
        get popupAttributeOptions() {
            if (!this.popup.entityId || !this.popup.entityType) return [];
            return this.taxonomy[this.popup.entityType] || [];
        },
        get popupNeedsRelationshipTarget() {
            return this.popup?.entityType === 'person' && String(this.popup?.attribute || '').trim() === 'relationship';
        },
        get popupRelationshipTargetOptions() {
            const currentEntityId = String(this.popup?.entityId || '').trim();
            const people = this.entityGroups?.person || [];
            return people.filter((ent) => String(ent?.id || '').trim() && String(ent.id).trim() !== currentEntityId);
        },
        get canApplyPopupAnnotation() {
            if (!this.popup?.entityId || !this.popup?.attribute) return false;
            if (!this.popupNeedsRelationshipTarget) return true;
            return /^person_\d+$/.test(String(this.popup?.relationshipTarget || '').trim());
        },
        get decisionLogs() {
            const raw = this.docData?.decision_logs ?? this.docData?.decision_log;
            if (!raw) return [];
            if (Array.isArray(raw)) {
                return raw.filter((item) => item && typeof item === 'object');
            }
            if (typeof raw === 'object') {
                return [raw];
            }
            return [];
        },
        get hasPrevDocument() {
            return this.currentDocGlobalIndex > 0;
        },
        get hasNextDocument() {
            return this.currentDocGlobalIndex >= 0 && this.currentDocGlobalIndex < this.allDocuments.length - 1;
        },
        get globalDocProgressLabel() {
            if (this.currentDocGlobalIndex < 0 || this.allDocuments.length === 0) return '—/—';
            return `${this.currentDocGlobalIndex + 1}/${this.allDocuments.length}`;
        },
        get currentNavigatorContext() {
            if (this.currentDocGlobalIndex < 0 || this.allDocuments.length === 0) return '';
            const current = this.allDocuments[this.currentDocGlobalIndex];
            return `${current.theme_label} · ${current.doc_id}`;
        },
        get historyDisplayCount() {
            if ((this.annotationVersions || []).length > 0) return this.annotationVersions.length;
            return (this.historyEntries || []).length;
        },
        agreementVariantLabel(variantKey) {
            const key = String(variantKey || '').toLowerCase();
            if (key === 'source') return 'Opus';
            if (key === 'reviewer_a') return this.agreementReviewerLabel('a');
            if (key === 'reviewer_b') return this.agreementReviewerLabel('b');
            if (key === 'final') return 'Final';
            if (key === 'editor') return 'Current Editor';
            return key || 'Version';
        },
        getCurrentReviewStatus(doc) {
            const data = doc && typeof doc === 'object' ? doc : {};
            const reviewStatuses = data.review_statuses && typeof data.review_statuses === 'object'
                ? data.review_statuses
                : {};
            const reviewBucket = this.reviewTarget ? reviewStatuses[this.reviewTarget] : null;
            const explicitStatus = reviewBucket && typeof reviewBucket === 'object'
                ? String(reviewBucket.status || '').trim()
                : '';
            if (explicitStatus) return explicitStatus;
            const taskStatus = String(data.active_review_task_status || '').trim();
            if (taskStatus === 'in_progress') return 'in_progress';
            if (taskStatus === 'completed') return 'completed';
            if (taskStatus === 'available') return 'draft';
            return 'draft';
        },

        // --- Initialization ---
        async init() {
            this.updateDashboardReturnLinks();
            try {
                const bootstrap = this.referenceMode
                    ? await API.loadReferenceBootstrap(this.theme, this.doc_id, {
                        reviewTarget: this.reviewTarget,
                        referenceReviewer: this.referenceReviewer,
                    })
                    : await API.loadEditorBootstrap(this.theme, this.doc_id, {
                        reviewTarget: this.reviewTarget,
                    });
                const doc = bootstrap.document || {};
                const taxonomy = bootstrap.taxonomy || {};
                const histData = bootstrap.history || { entries: [], annotation_versions: [], current_status: 'draft', last_editor: null };
                const metadata = bootstrap.metadata || { annotations: {}, questions: {}, rules: {}, has_history: false };
                this.referenceMode = Boolean(bootstrap.reference_mode) || this.referenceMode;
                this.docData = this._normalizeEditableDocument(doc);
                this.taxonomy = taxonomy;
                this.entityTypes = this.computeEntityTypes();
                this.applyDynamicEntityTypeStyles();
                if (!this.newEntityDialog.type) {
                    this.newEntityDialog.type = this.entityTypes[0] || 'person';
                }
                this.annotationMetadata = metadata;
                this.historyEntries = histData.entries || [];
                this.annotationVersions = histData.annotation_versions || [];
                const documentReviewStatus = String(
                    doc?.review_statuses?.document_annotation?.status || ''
                ).trim();
                this.currentStatus = this.reviewTarget
                    ? this.getCurrentReviewStatus(doc)
                    : (documentReviewStatus || histData.current_status || 'draft');
                this.refreshEntities();
                this.syncSourceEditorTextFromDoc();
                // Attach question listeners after everything loads
                this.$nextTick(() => {
                    setTimeout(() => {
                        this.attachQuestionAnnotationHoverListeners();
                        this.attachRuleReferenceHoverListeners();
                    }, 100);
                });
                if (this.agreementWorkspace.active) {
                    await this.initAgreementWorkspace();
                }
                this._loadGlobalDocumentIndex();
            } catch (e) {
                showToast('Failed to load document: ' + e.message, 'error');
            } finally {
                this.loading = false;
            }
        },

        normalizeDashboardReturnSection(section) {
            const normalized = String(section || '').trim().toLowerCase();
            return ['documents', 'rules', 'questions'].includes(normalized) ? normalized : 'documents';
        },

        dashboardReturnHref() {
            const section = this.normalizeDashboardReturnSection(this.dashboardReturnSection);
            const params = new URLSearchParams();
            params.set('dashboard_section', section);
            return `/?${params.toString()}`;
        },

        updateDashboardReturnLinks() {
            const href = this.dashboardReturnHref();
            ['nav-home-link', 'nav-dashboard', 'editor-back-link'].forEach((id) => {
                const node = document.getElementById(id);
                if (node) node.setAttribute('href', href);
            });
        },

        _normalizeEditableDocument(rawDoc) {
            const safeRaw = (rawDoc && typeof rawDoc === 'object' && !Array.isArray(rawDoc)) ? rawDoc : {};
            const cloned = JSON.parse(JSON.stringify(safeRaw));
            if (typeof cloned.document_to_annotate !== 'string') {
                cloned.document_to_annotate = String(cloned.document_to_annotate || '');
            }
            if (!Array.isArray(cloned.questions)) cloned.questions = [];
            cloned.questions = cloned.questions.map((q, idx) => {
                const item = (q && typeof q === 'object' && !Array.isArray(q)) ? q : {};
                const answerType = this._normalizeAnswerType(
                    item.answer_type,
                    item.is_answer_invariant
                );
                const reasoningChain = this._normalizeReasoningChain(item.reasoning_chain, item.reasoning_chain_text);
                return {
                    question_id: String(item.question_id || `q_${idx + 1}`),
                    question: String(item.question || ''),
                    question_type: this._normalizeQuestionType(item.question_type),
                    answer: answerType === 'refusal'
                        ? refusalAnswerLiteral
                        : String(item.answer ?? ''),
                    answer_type: answerType,
                    reasoning_chain: reasoningChain,
                    reasoning_chain_text: reasoningChain.join('\n'),
                };
            });
            cloned[qaCoverageExemptionsField] = this._normalizeQaCoverageExemptions(
                cloned[qaCoverageExemptionsField]
            );
            if (!Array.isArray(cloned.rules)) cloned.rules = [];
            if (!Array.isArray(cloned.implicit_rules)) cloned.implicit_rules = [];
            if (
                cloned.implicit_rule_exclusions !== undefined
                && !Array.isArray(cloned.implicit_rule_exclusions)
            ) {
                cloned.implicit_rule_exclusions = [];
            }
            if (!cloned.document_id) cloned.document_id = this.doc_id;
            if (!cloned.document_theme) cloned.document_theme = this.theme;
            this._alignQuestionFieldsWithDocumentSurfaces(cloned);
            cloned.num_questions = cloned.questions.length;
            return cloned;
        },

        _normalizeInlineEntityRef(rawRef) {
            const raw = String(rawRef || '').trim();
            if (!raw) return '';
            return raw
                .split('.')
                .map((part) => String(part || '').trim())
                .filter((part) => part.length > 0)
                .join('.')
                .toLowerCase();
        },

        _buildDocumentAnnotationMaps(documentText) {
            const source = String(documentText || '');
            const surfacesByRef = {};
            const uniqueRefBySurface = {};
            const ambiguousSurfaces = new Set();
            for (const match of source.matchAll(inlineAnnotationRegex)) {
                const surface = String(match[1] || '').trim();
                const ref = this._normalizeInlineEntityRef(match[2]);
                if (!surface || !ref) continue;
                if (!Object.prototype.hasOwnProperty.call(surfacesByRef, ref)) {
                    surfacesByRef[ref] = surface;
                }
                const surfaceKey = surface.toLowerCase();
                if (ambiguousSurfaces.has(surfaceKey)) continue;
                const existing = uniqueRefBySurface[surfaceKey];
                if (existing && existing !== ref) {
                    ambiguousSurfaces.add(surfaceKey);
                    delete uniqueRefBySurface[surfaceKey];
                    continue;
                }
                uniqueRefBySurface[surfaceKey] = ref;
            }
            return { surfacesByRef, uniqueRefBySurface };
        },

        _rewriteAnnotatedTextToDocumentSurfaces(rawText, maps) {
            const source = String(rawText || '');
            if (!source || !maps || !maps.surfacesByRef || Object.keys(maps.surfacesByRef).length === 0) {
                return source;
            }
            return source.replace(inlineAnnotationRegex, (_full, rawSurface, rawRef) => {
                const originalSurface = String(rawSurface || '').trim();
                const rawRefText = String(rawRef || '').trim();
                let normalizedRef = this._normalizeInlineEntityRef(rawRefText);
                if (!normalizedRef) return `[${originalSurface}; ${rawRefText}]`;

                let replacementSurface = maps.surfacesByRef[normalizedRef];
                if (!replacementSurface) {
                    const mappedRef = maps.uniqueRefBySurface[String(originalSurface || '').toLowerCase()];
                    if (mappedRef) {
                        normalizedRef = mappedRef;
                        replacementSurface = maps.surfacesByRef[normalizedRef];
                    }
                }
                if (!replacementSurface) replacementSurface = originalSurface;
                return `[${replacementSurface}; ${normalizedRef}]`;
            });
        },

        _alignQuestionFieldsWithDocumentSurfaces(document) {
            if (!document || typeof document !== 'object' || !Array.isArray(document.questions) || document.questions.length === 0) {
                return;
            }
            const maps = this._buildDocumentAnnotationMaps(document.document_to_annotate || '');
            if (!maps || Object.keys(maps.surfacesByRef).length === 0) return;

            document.questions = document.questions.map((rawQuestion) => {
                const question = (rawQuestion && typeof rawQuestion === 'object' && !Array.isArray(rawQuestion))
                    ? { ...rawQuestion }
                    : {};
                question.question = this._rewriteAnnotatedTextToDocumentSurfaces(question.question, maps);

                if (Array.isArray(question.reasoning_chain)) {
                    question.reasoning_chain = question.reasoning_chain.map((step) => (
                        typeof step === 'string'
                            ? this._rewriteAnnotatedTextToDocumentSurfaces(step, maps)
                            : step
                    ));
                }
                if (typeof question.reasoning_chain_text === 'string') {
                    question.reasoning_chain_text = this._rewriteAnnotatedTextToDocumentSurfaces(
                        question.reasoning_chain_text,
                        maps
                    );
                }
                if (typeof question.answer === 'string') {
                    question.answer = this._rewriteAnnotatedTextToDocumentSurfaces(question.answer, maps);
                } else if (Array.isArray(question.answer)) {
                    question.answer = question.answer.map((item) => (
                        typeof item === 'string'
                            ? this._rewriteAnnotatedTextToDocumentSurfaces(item, maps)
                            : item
                    ));
                }
                const normalizedReasoning = this._normalizeReasoningChain(
                    question.reasoning_chain,
                    question.reasoning_chain_text
                );
                question.reasoning_chain = normalizedReasoning;
                question.reasoning_chain_text = normalizedReasoning.join('\n');
                return question;
            });
            document.num_questions = document.questions.length;
        },

        _normalizeQuestionType(rawType) {
            const normalized = String(rawType || '').trim().toLowerCase();
            return qaRequiredQuestionTypes.includes(normalized) ? normalized : 'extractive';
        },

        _normalizeAnswerType(rawAnswerType, rawInvariant = null) {
            const normalized = String(rawAnswerType || '').trim().toLowerCase();
            if (qaRequiredAnswerTypes.includes(normalized)) return normalized;
            if (rawInvariant === true) return 'invariant';
            if (rawInvariant === false) return 'variant';
            return 'variant';
        },

        _normalizeReasoningChain(rawChain, rawText = null) {
            if (typeof rawText === 'string' && rawText.trim()) {
                return rawText
                    .split(/\r?\n/)
                    .map((step) => String(step || '').trim())
                    .filter((step) => step.length > 0);
            }
            if (!Array.isArray(rawChain)) return [];
            return rawChain
                .map((step) => String(step || '').trim())
                .filter((step) => step.length > 0);
        },

        _qaCoverageKey(questionType, answerType) {
            return `${questionType}::${answerType}`;
        },

        _qaCoveragePairFromQuestion(questionLike) {
            if (!questionLike || typeof questionLike !== 'object') return null;
            return {
                question_type: this._normalizeQuestionType(questionLike.question_type),
                answer_type: this._normalizeAnswerType(
                    questionLike.answer_type,
                    questionLike.is_answer_invariant
                ),
            };
        },

        _qaCoveragePairKeyFromQuestion(questionLike) {
            const pair = this._qaCoveragePairFromQuestion(questionLike);
            if (!pair) return '';
            return this._qaCoverageKey(pair.question_type, pair.answer_type);
        },

        _qaCoverageExemptionMapFromDocument(documentLike) {
            const normalized = this._normalizeQaCoverageExemptions(
                documentLike?.[qaCoverageExemptionsField]
            );
            const map = {};
            for (const item of normalized) {
                const key = this._qaCoverageKey(item.question_type, item.answer_type);
                map[key] = String(item.justification ?? '');
            }
            return map;
        },

        _qaCoveragePresentPairSetFromQuestions(questionsLike) {
            const set = new Set();
            const questions = Array.isArray(questionsLike) ? questionsLike : [];
            for (const question of questions) {
                if (!question || typeof question !== 'object') continue;
                const questionType = this._normalizeQuestionType(question.question_type);
                const answerType = this._normalizeAnswerType(
                    question.answer_type,
                    question.is_answer_invariant
                );
                set.add(this._qaCoverageKey(questionType, answerType));
            }
            return set;
        },

        _qaCoveragePresentPairSetFromDocument(documentLike) {
            return this._qaCoveragePresentPairSetFromQuestions(documentLike?.questions || []);
        },

        _qaCoveragePresentPairMapFromDocument(documentLike) {
            const map = {};
            for (const key of this._qaCoveragePresentPairSetFromDocument(documentLike)) {
                map[key] = true;
            }
            return map;
        },

        _qaCoverageQuestionsByPairFromDocument(documentLike) {
            const out = {};
            const questions = Array.isArray(documentLike?.questions) ? documentLike.questions : [];
            for (let idx = 0; idx < questions.length; idx++) {
                const normalized = this._normalizeQuestionForAgreement(questions[idx], '');
                const pairKey = this._qaCoveragePairKeyFromQuestion(normalized);
                if (!pairKey) continue;
                const key = this._questionConflictKeyFromQuestion(normalized, idx);
                if (!Array.isArray(out[pairKey])) out[pairKey] = [];
                out[pairKey].push({
                    key,
                    question_id: String(normalized.question_id || key || '').trim(),
                    question: String(normalized.question || ''),
                    answer: String(normalized.answer || ''),
                    question_type: this._normalizeQuestionType(normalized.question_type),
                    answer_type: this._normalizeAnswerType(
                        normalized.answer_type,
                        normalized.is_answer_invariant
                    ),
                    reasoning_chain: this._normalizeReasoningChain(
                        normalized.reasoning_chain,
                        normalized.reasoning_chain_text
                    ),
                });
            }
            return out;
        },

        _requiredQaCoveragePairs() {
            const pairs = [];
            for (const questionType of qaRequiredQuestionTypes) {
                for (const answerType of qaRequiredAnswerTypes) {
                    pairs.push({ question_type: questionType, answer_type: answerType });
                }
            }
            return pairs;
        },

        _normalizeQaCoverageExemptions(rawExemptions, { dropCovered = false } = {}) {
            const list = Array.isArray(rawExemptions) ? rawExemptions : [];
            const presentPairs = dropCovered ? this._qaCoveragePresentPairSet() : new Set();
            const normalized = [];
            const seen = new Set();
            for (const item of list) {
                if (!item || typeof item !== 'object') continue;
                const questionType = this._normalizeQuestionType(item.question_type);
                const answerType = this._normalizeAnswerType(item.answer_type);
                const key = this._qaCoverageKey(questionType, answerType);
                if (dropCovered && presentPairs.has(key)) continue;
                if (seen.has(key)) continue;
                const justification = String(item.justification ?? '');
                if (!justification.trim()) continue;
                normalized.push({
                    question_type: questionType,
                    answer_type: answerType,
                    justification,
                });
                seen.add(key);
            }
            return normalized;
        },

        _qaCoverageExemptionsMap() {
            const normalized = this._normalizeQaCoverageExemptions(
                this.docData?.[qaCoverageExemptionsField]
            );
            const result = new Map();
            for (const item of normalized) {
                result.set(
                    this._qaCoverageKey(item.question_type, item.answer_type),
                    String(item.justification ?? '')
                );
            }
            return result;
        },

        _annotationDisplayText(el) {
            if (!el) return '';
            const clone = el.cloneNode(true);
            clone.querySelectorAll('.ann-delete-btn, .resize-handle').forEach((node) => node.remove());
            return String(clone.textContent || '').trim();
        },

        _qaCoveragePresentPairSet() {
            return this._qaCoveragePresentPairSetFromQuestions(this.docData?.questions || []);
        },

        _qaCoverageNormalizedQuestions() {
            const questions = Array.isArray(this.docData?.questions) ? this.docData.questions : [];
            return questions
                .filter((question) => question && typeof question === 'object')
                .map((question, idx) => ({
                    index: idx,
                    question_id: String(question.question_id || '').trim(),
                    question_type: this._normalizeQuestionType(question.question_type),
                    answer_type: this._normalizeAnswerType(
                        question.answer_type,
                        question.is_answer_invariant
                    ),
                    question: String(question.question || '').trim(),
                    answer: String(question.answer || '').trim(),
                }));
        },

        _qaCoverageDuplicatePairs() {
            const counts = new Map();
            for (const question of this._qaCoverageNormalizedQuestions()) {
                const key = this._qaCoverageKey(question.question_type, question.answer_type);
                counts.set(key, (counts.get(key) || 0) + 1);
            }
            return Array.from(counts.entries())
                .filter(([, count]) => count > 1)
                .map(([key, count]) => {
                    const [question_type, answer_type] = key.split('::');
                    return { question_type, answer_type, count };
                });
        },

        _qaCoverageDuplicateQuestionIds() {
            const counts = new Map();
            for (const question of this._qaCoverageNormalizedQuestions()) {
                if (!question.question_id) continue;
                counts.set(question.question_id, (counts.get(question.question_id) || 0) + 1);
            }
            return Array.from(counts.entries())
                .filter(([, count]) => count > 1)
                .map(([question_id, count]) => ({ question_id, count }));
        },

        qaCoverageRows() {
            const present = this._qaCoveragePresentPairSet();
            const exemptions = this._qaCoverageExemptionsMap();
            return this._requiredQaCoveragePairs().map((pair) => {
                const key = this._qaCoverageKey(pair.question_type, pair.answer_type);
                const justification = String(exemptions.get(key) || '');
                const hasQuestion = present.has(key);
                return {
                    ...pair,
                    key,
                    has_question: hasQuestion,
                    justification,
                    status: hasQuestion
                        ? 'present'
                        : (justification ? 'exempted' : 'missing'),
                };
            });
        },

        _qaCoverageMissingRows() {
            return this.qaCoverageRows().filter((row) => row.status === 'missing');
        },

        qaCoverageSummaryLabel() {
            const rows = this.qaCoverageRows();
            const doneCount = rows.filter((row) => row.status !== 'missing').length;
            return `${doneCount}/${rows.length}`;
        },

        qaCoverageIssues() {
            const issues = [];
            const questionCount = this._qaCoverageNormalizedQuestions().length;
            const maxQuestions = this._requiredQaCoveragePairs().length;
            if (questionCount > maxQuestions) {
                issues.push(`Too many questions: ${questionCount}/${maxQuestions}. Keep at most one QA per combination.`);
            }

            const duplicatePairs = this._qaCoverageDuplicatePairs();
            if (duplicatePairs.length) {
                const labels = duplicatePairs
                    .map((pair) => `${this.qaCoverageQuestionTypeLabel(pair.question_type)} + ${this.qaCoverageAnswerTypeLabel(pair.answer_type)}`)
                    .join(', ');
                issues.push(`Duplicate question/answer-type combinations: ${labels}.`);
            }

            const duplicateQuestionIds = this._qaCoverageDuplicateQuestionIds();
            if (duplicateQuestionIds.length) {
                const labels = duplicateQuestionIds.map((item) => item.question_id).join(', ');
                issues.push(`Duplicate question IDs: ${labels}.`);
            }
            return issues;
        },

        qaCoverageQuestionTypeLabel(value) {
            const key = this._normalizeQuestionType(value);
            return key.charAt(0).toUpperCase() + key.slice(1);
        },

        qaCoverageAnswerTypeLabel(value) {
            const key = this._normalizeAnswerType(value);
            return key.charAt(0).toUpperCase() + key.slice(1);
        },

        qaCoverageSetJustification(questionType, answerType, justification) {
            if (!this.docData || typeof this.docData !== 'object') return;
            const qType = this._normalizeQuestionType(questionType);
            const aType = this._normalizeAnswerType(answerType);
            const key = this._qaCoverageKey(qType, aType);
            const existing = this._normalizeQaCoverageExemptions(this.docData[qaCoverageExemptionsField]);
            const nextValue = String(justification || '');
            const trimmed = nextValue.trim();
            const next = [];
            let replaced = false;
            for (const item of existing) {
                const itemKey = this._qaCoverageKey(item.question_type, item.answer_type);
                if (itemKey !== key) {
                    next.push(item);
                    continue;
                }
                replaced = true;
                if (trimmed) {
                    next.push({
                        question_type: qType,
                        answer_type: aType,
                        justification: nextValue,
                    });
                }
            }
            if (!replaced && trimmed) {
                next.push({
                    question_type: qType,
                    answer_type: aType,
                    justification: nextValue,
                });
            }
            this.docData[qaCoverageExemptionsField] = next;
            this.markDirty();
        },

        _prepareQuestionPayloadForPersistence() {
            if (!this.docData || typeof this.docData !== 'object') return;
            const questions = Array.isArray(this.docData.questions) ? this.docData.questions : [];
            this.docData.questions = questions.map((question, idx) => {
                const item = question && typeof question === 'object' ? question : {};
                const answerType = this._normalizeAnswerType(item.answer_type, item.is_answer_invariant);
                const reasoningChain = this._normalizeReasoningChain(
                    item.reasoning_chain,
                    item.reasoning_chain_text
                );
                return {
                    question_id: String(item.question_id || `q_${idx + 1}`),
                    question: String(item.question || '').trim(),
                    question_type: this._normalizeQuestionType(item.question_type),
                    answer: answerType === 'refusal'
                        ? refusalAnswerLiteral
                        : String(item.answer ?? '').trim(),
                    answer_type: answerType,
                    reasoning_chain: reasoningChain,
                    reasoning_chain_text: reasoningChain.join('\n'),
                };
            });
            this.docData[qaCoverageExemptionsField] = this._normalizeQaCoverageExemptions(
                this.docData[qaCoverageExemptionsField],
                { dropCovered: true }
            );
            this.docData.num_questions = this.docData.questions.length;
        },

        qaCoverageValidationForSubmit() {
            const missingRows = this._qaCoverageMissingRows();
            const duplicatePairs = this._qaCoverageDuplicatePairs();
            const duplicateQuestionIds = this._qaCoverageDuplicateQuestionIds();
            const questionCount = this._qaCoverageNormalizedQuestions().length;
            const maxQuestions = this._requiredQaCoveragePairs().length;
            const errors = [];

            if (questionCount > maxQuestions) {
                errors.push(`Too many questions: ${questionCount}/${maxQuestions}. Keep at most one QA per combination.`);
            }

            if (duplicatePairs.length) {
                const labels = duplicatePairs
                    .map((pair) => `${this.qaCoverageQuestionTypeLabel(pair.question_type)} + ${this.qaCoverageAnswerTypeLabel(pair.answer_type)}`)
                    .join(', ');
                errors.push(`Duplicate question/answer-type combinations: ${labels}.`);
            }

            if (duplicateQuestionIds.length) {
                const labels = duplicateQuestionIds.map((item) => item.question_id).join(', ');
                errors.push(`Duplicate question IDs: ${labels}.`);
            }

            return {
                valid: !missingRows.length && !errors.length,
                missingRows,
                duplicatePairs,
                duplicateQuestionIds,
                tooManyQuestions: questionCount > maxQuestions,
                errors,
            };
        },

        _ensureAgreementWorkspaceState() {
            const ws = (this.agreementWorkspace && typeof this.agreementWorkspace === 'object')
                ? this.agreementWorkspace
                : {};
            this.agreementWorkspace = ws;

            ws.versions = (ws.versions && typeof ws.versions === 'object') ? ws.versions : {};
            const defaultVersion = (key, label) => ({
                key,
                label,
                username: '',
                available: false,
                path: '',
                document_to_annotate: '',
                editable_document: null,
            });
            ws.versions.source = (ws.versions.source && typeof ws.versions.source === 'object')
                ? ws.versions.source
                : defaultVersion('source', 'Opus');
            ws.versions.reviewer_a = (ws.versions.reviewer_a && typeof ws.versions.reviewer_a === 'object')
                ? ws.versions.reviewer_a
                : defaultVersion('reviewer_a', 'Reviewer 1');
            ws.versions.reviewer_b = (ws.versions.reviewer_b && typeof ws.versions.reviewer_b === 'object')
                ? ws.versions.reviewer_b
                : defaultVersion('reviewer_b', 'Reviewer 2');
            ws.versions.final = (ws.versions.final && typeof ws.versions.final === 'object')
                ? ws.versions.final
                : defaultVersion('final', 'Final');

            ws.merge = (ws.merge && typeof ws.merge === 'object') ? ws.merge : {};
            if (!Array.isArray(ws.merge.conflicts)) ws.merge.conflicts = [];
            if (!Array.isArray(ws.merge.inlineSegments)) ws.merge.inlineSegments = [];
            ws.merge.decisions = (ws.merge.decisions && typeof ws.merge.decisions === 'object') ? ws.merge.decisions : {};
            ws.merge.manualResolutions = (ws.merge.manualResolutions && typeof ws.merge.manualResolutions === 'object')
                ? ws.merge.manualResolutions
                : {};
            ws.merge.manualTextResolutions = (ws.merge.manualTextResolutions && typeof ws.merge.manualTextResolutions === 'object')
                ? ws.merge.manualTextResolutions
                : {};
            ws.merge.selectionMode = (ws.merge.selectionMode && typeof ws.merge.selectionMode === 'object')
                ? ws.merge.selectionMode
                : {};

            ws.structured = (ws.structured && typeof ws.structured === 'object') ? ws.structured : {};
            if (!Array.isArray(ws.structured.questionConflicts)) ws.structured.questionConflicts = [];
            if (!Array.isArray(ws.structured.ruleConflicts)) ws.structured.ruleConflicts = [];
            ws.structured.decisions = (ws.structured.decisions && typeof ws.structured.decisions === 'object')
                ? ws.structured.decisions
                : {};
            ws.structured.manualResolutions = (ws.structured.manualResolutions && typeof ws.structured.manualResolutions === 'object')
                ? ws.structured.manualResolutions
                : {};
            ws.structured.qaCoverageExemptionsBySide = (
                ws.structured.qaCoverageExemptionsBySide
                && typeof ws.structured.qaCoverageExemptionsBySide === 'object'
            )
                ? ws.structured.qaCoverageExemptionsBySide
                : { source: {}, a: {}, b: {} };
            ws.structured.qaCoveragePairsBySide = (
                ws.structured.qaCoveragePairsBySide
                && typeof ws.structured.qaCoveragePairsBySide === 'object'
            )
                ? ws.structured.qaCoveragePairsBySide
                : { source: {}, a: {}, b: {} };
            ws.structured.qaCoverageQuestionsBySide = (
                ws.structured.qaCoverageQuestionsBySide
                && typeof ws.structured.qaCoverageQuestionsBySide === 'object'
            )
                ? ws.structured.qaCoverageQuestionsBySide
                : { source: {}, a: {}, b: {} };
            ws.comparison = (ws.comparison && typeof ws.comparison === 'object') ? ws.comparison : {};
            const allowedComparisonKeys = new Set(['reviewer_a', 'reviewer_b', 'final', 'source']);
            const sideAKey = String(ws.comparison.sideAKey || 'reviewer_a').trim().toLowerCase();
            const sideBKey = String(ws.comparison.sideBKey || 'reviewer_b').trim().toLowerCase();
            ws.comparison.mode = String(ws.comparison.mode || 'annotator_agreement').trim().toLowerCase();
            ws.comparison.sideAKey = allowedComparisonKeys.has(sideAKey) ? sideAKey : 'reviewer_a';
            ws.comparison.sideBKey = allowedComparisonKeys.has(sideBKey) ? sideBKey : 'reviewer_b';
            ws.comparison.contestVariant = String(ws.comparison.contestVariant || '').trim().toLowerCase();
            ws.disableCurrentInference = !!ws.disableCurrentInference;
            ws.inferResolutionFromCurrent = !!ws.inferResolutionFromCurrent;
        },

        _escapeAgreementHtml(value) {
            return String(value || '')
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        },

        _sanitizeAgreementClassToken(value) {
            return String(value || '').replace(/[^a-zA-Z0-9_-]/g, '-');
        },

        _parseAgreementAnnotatedText(text) {
            const raw = String(text || '');
            const annotations = parseAnnotations(raw);
            const result = {
                plainText: '',
                annotations: [],
            };
            if (!raw) return result;

            let rawCursor = 0;
            let plainCursor = 0;
            for (const ann of annotations) {
                const prefix = raw.substring(rawCursor, ann.start);
                result.plainText += prefix;
                plainCursor += prefix.length;

                const annText = String(ann.text || '');
                const start = plainCursor;
                const end = start + annText.length;
                result.annotations.push({
                    start,
                    end,
                    text: annText,
                    ref: String(ann.ref || '').trim(),
                    entityId: String(ann.entityId || ''),
                    entityType: String(ann.entityType || ''),
                    attribute: ann.attribute ? String(ann.attribute) : '',
                });
                result.plainText += annText;
                plainCursor = end;
                rawCursor = ann.end;
            }

            const suffix = raw.substring(rawCursor);
            result.plainText += suffix;
            return result;
        },

        _tokenizeAgreementText(text) {
            const raw = String(text || '');
            if (!raw) return [];
            const regex = /\s+|[^\s]+/g;
            const tokens = [];
            let match = null;
            while ((match = regex.exec(raw)) !== null) {
                const value = String(match[0] || '');
                const start = Number(match.index || 0);
                const end = start + value.length;
                tokens.push({ value, start, end });
            }
            return tokens;
        },

        _computeAgreementTextEdits(baseText, targetText) {
            const base = String(baseText || '');
            const target = String(targetText || '');
            if (base === target) return [];

            const a = this._tokenizeAgreementText(base);
            const b = this._tokenizeAgreementText(target);
            const n = a.length;
            const m = b.length;

            // Guard very large documents: fallback to one coarse replacement region.
            if ((n + 1) * (m + 1) > 6000000) {
                return [{
                    baseStart: 0,
                    baseEnd: base.length,
                    targetStart: 0,
                    targetEnd: target.length,
                    baseText: base,
                    targetText: target,
                }];
            }

            const stride = m + 1;
            const dirs = new Uint8Array((n + 1) * (m + 1));
            let prev = new Uint32Array(m + 1);
            let curr = new Uint32Array(m + 1);

            for (let j = 0; j <= m; j++) prev[j] = j;
            for (let i = 1; i <= n; i++) {
                curr[0] = i;
                for (let j = 1; j <= m; j++) {
                    const ai = a[i - 1].value;
                    const bj = b[j - 1].value;
                    const idx = i * stride + j;
                    if (ai === bj) {
                        curr[j] = prev[j - 1];
                        dirs[idx] = 0; // diagonal equal
                        continue;
                    }
                    const sub = prev[j - 1] + 1;
                    const del = prev[j] + 1;
                    const ins = curr[j - 1] + 1;
                    if (sub <= del && sub <= ins) {
                        curr[j] = sub;
                        dirs[idx] = 1; // diagonal replace
                    } else if (del <= ins) {
                        curr[j] = del;
                        dirs[idx] = 2; // up delete
                    } else {
                        curr[j] = ins;
                        dirs[idx] = 3; // left insert
                    }
                }
                const tmp = prev;
                prev = curr;
                curr = tmp;
            }

            const steps = [];
            let i = n;
            let j = m;
            while (i > 0 || j > 0) {
                const idx = i * stride + j;
                const dir = dirs[idx];
                if (i > 0 && j > 0 && (dir === 0 || dir === 1)) {
                    steps.push(dir === 0 ? 'eq' : 'sub');
                    i -= 1;
                    j -= 1;
                    continue;
                }
                if (i > 0 && (j === 0 || dir === 2)) {
                    steps.push('del');
                    i -= 1;
                    continue;
                }
                steps.push('ins');
                j -= 1;
            }
            steps.reverse();

            const edits = [];
            let idxA = 0;
            let idxB = 0;
            let open = null;

            const flush = () => {
                if (!open) return;
                const baseEnd = idxA < n ? a[idxA].start : base.length;
                const targetEnd = idxB < m ? b[idxB].start : target.length;
                const chunk = {
                    baseStart: open.baseStart,
                    baseEnd,
                    targetStart: open.targetStart,
                    targetEnd,
                    baseText: base.substring(open.baseStart, baseEnd),
                    targetText: target.substring(open.targetStart, targetEnd),
                };
                if (!(chunk.baseStart === chunk.baseEnd && chunk.targetStart === chunk.targetEnd) &&
                    (chunk.baseText !== chunk.targetText)) {
                    edits.push(chunk);
                }
                open = null;
            };

            for (const step of steps) {
                if (step === 'eq') {
                    flush();
                    idxA += 1;
                    idxB += 1;
                    continue;
                }
                if (!open) {
                    open = {
                        baseStart: idxA < n ? a[idxA].start : base.length,
                        targetStart: idxB < m ? b[idxB].start : target.length,
                    };
                }
                if (step === 'sub') {
                    idxA += 1;
                    idxB += 1;
                } else if (step === 'del') {
                    idxA += 1;
                } else {
                    idxB += 1;
                }
            }
            flush();

            if (edits.length === 0 && base !== target) {
                edits.push({
                    baseStart: 0,
                    baseEnd: base.length,
                    targetStart: 0,
                    targetEnd: target.length,
                    baseText: base,
                    targetText: target,
                });
            }
            return edits;
        },

        _buildAgreementBaseToTargetMapper(edits) {
            const list = (Array.isArray(edits) ? edits : [])
                .slice()
                .sort((x, y) => (Number(x.baseStart || 0) - Number(y.baseStart || 0)) || (Number(x.baseEnd || 0) - Number(y.baseEnd || 0)));
            return (basePos) => {
                const pos = Number(basePos || 0);
                let delta = 0;
                for (const op of list) {
                    const opEnd = Number(op.baseEnd || 0);
                    const opStart = Number(op.baseStart || 0);
                    const targetStart = Number(op.targetStart || 0);
                    const targetEnd = Number(op.targetEnd || targetStart);
                    const baseLen = Math.max(0, opEnd - opStart);
                    const targetLen = Math.max(0, targetEnd - targetStart);
                    if (opEnd < pos || (opStart === opEnd && opEnd <= pos)) {
                        delta += targetLen - baseLen;
                        continue;
                    }
                    if (pos >= opStart && pos <= opEnd && baseLen > 0) {
                        if (targetLen === 0) return targetStart;
                        const offset = pos - opStart;
                        const mappedOffset = Math.round((offset / baseLen) * targetLen);
                        return targetStart + mappedOffset;
                    }
                    break;
                }
                return pos + delta;
            };
        },

        _buildAgreementTargetToBaseMapper(edits) {
            const list = (Array.isArray(edits) ? edits : [])
                .slice()
                .sort((x, y) => (Number(x.targetStart || 0) - Number(y.targetStart || 0)) || (Number(x.targetEnd || 0) - Number(y.targetEnd || 0)));
            return (targetPos) => {
                const pos = Number(targetPos || 0);
                let delta = 0;
                for (const op of list) {
                    const baseStart = Number(op.baseStart || 0);
                    const baseEnd = Number(op.baseEnd || 0);
                    const targetStart = Number(op.targetStart || 0);
                    const targetEnd = Number(op.targetEnd || targetStart);
                    const baseLen = Math.max(0, baseEnd - baseStart);
                    const targetLen = Math.max(0, targetEnd - targetStart);
                    if (targetEnd < pos || (targetStart === targetEnd && targetEnd <= pos)) {
                        delta += baseLen - targetLen;
                        continue;
                    }
                    if (pos >= targetStart && pos <= targetEnd && targetLen > 0) {
                        if (baseLen === 0) return baseStart;
                        const offset = pos - targetStart;
                        const mappedOffset = Math.round((offset / targetLen) * baseLen);
                        return baseStart + mappedOffset;
                    }
                    break;
                }
                return pos + delta;
            };
        },

        _agreementInferEntityTypeFromRef(ref) {
            const raw = String(ref || '').trim();
            if (!raw) return '';
            const entityId = raw.split('.', 1)[0] || '';
            if (!entityId) return '';
            return String(entityId).replace(/_\d+$/, '');
        },

        _agreementLocateProjectedAnnotationInBase(basePlain, annotationText, opts = {}) {
            const base = String(basePlain || '');
            const target = String(annotationText || '');
            if (!base || !target) return null;
            const hintStart = Math.max(0, Number(opts.hintStart || 0));
            const hintEnd = Math.max(hintStart, Number(opts.hintEnd || hintStart));
            const before = String(opts.beforeContext || '');
            const after = String(opts.afterContext || '');
            const beforeTail = before ? before.slice(-18) : '';
            const afterHead = after ? after.slice(0, 18) : '';

            const matches = this._agreementFindOccurrences(base, target, 160);
            if (matches.length === 0) return null;
            let best = null;
            for (const start of matches) {
                const end = start + target.length;
                let score = Math.abs(start - hintStart) + Math.abs(end - hintEnd);
                if (beforeTail) {
                    const left = base.substring(Math.max(0, start - beforeTail.length), start);
                    if (left === beforeTail) score -= 80;
                }
                if (afterHead) {
                    const right = base.substring(end, Math.min(base.length, end + afterHead.length));
                    if (right === afterHead) score -= 80;
                }
                if (!best || score < best.score) {
                    best = { start, end, score };
                }
            }
            return best ? { start: best.start, end: best.end } : null;
        },

        _projectAgreementAnnotationsToBasePlain(parsedReviewer, basePlain, targetToBaseMapper, opts = {}) {
            const parsed = parsedReviewer && typeof parsedReviewer === 'object'
                ? parsedReviewer
                : { plainText: '', annotations: [] };
            const reviewerPlain = String(parsed.plainText || '');
            const base = String(basePlain || '');
            if (!base) return [];
            const mapper = typeof targetToBaseMapper === 'function'
                ? targetToBaseMapper
                : ((value) => Number(value || 0));
            const baseToTargetMapper = typeof opts?.baseToTargetMapper === 'function'
                ? opts.baseToTargetMapper
                : null;
            const changedTargetRanges = Array.isArray(opts?.changedTargetRanges)
                ? opts.changedTargetRanges
                    .map((item) => {
                        const rawStart = Number(item?.targetStart ?? item?.start ?? 0);
                        const rawEnd = Number(item?.targetEnd ?? item?.end ?? rawStart);
                        const start = Number.isFinite(rawStart) ? Math.max(0, rawStart) : 0;
                        const end = Number.isFinite(rawEnd) ? Math.max(start, rawEnd) : start;
                        return { start, end };
                    })
                    .filter((item) => item.end > item.start)
                : [];
            const clamp = (value) => Math.max(0, Math.min(Number(value || 0), base.length));
            const overlapsChangedTargetRange = (start, end) => {
                for (const item of changedTargetRanges) {
                    if (end > item.start && start < item.end) return true;
                }
                return false;
            };
            const out = [];
            const seen = new Set();

            for (const ann of parsed.annotations || []) {
                const ref = String(ann?.ref || '').trim();
                if (!ref) continue;
                const targetStart = Math.max(0, Number(ann?.start || 0));
                const targetEnd = Math.max(targetStart, Number(ann?.end || targetStart));
                if (targetEnd <= targetStart) continue;
                if (overlapsChangedTargetRange(targetStart, targetEnd)) continue;
                const annText = reviewerPlain.substring(targetStart, targetEnd);

                let mappedStart = clamp(mapper(targetStart));
                let mappedEnd = Math.max(mappedStart, clamp(mapper(targetEnd)));
                if (mappedEnd <= mappedStart && annText) {
                    mappedEnd = Math.min(base.length, mappedStart + annText.length);
                }

                let baseSlice = base.substring(mappedStart, mappedEnd);
                if (annText && !this._agreementTextEquivalent(baseSlice, annText)) {
                    const beforeContext = reviewerPlain.substring(Math.max(0, targetStart - 24), targetStart);
                    const afterContext = reviewerPlain.substring(targetEnd, Math.min(reviewerPlain.length, targetEnd + 24));
                    const located = this._agreementLocateProjectedAnnotationInBase(base, annText, {
                        hintStart: mappedStart,
                        hintEnd: mappedEnd,
                        beforeContext,
                        afterContext,
                    });
                    if (located) {
                        mappedStart = clamp(located.start);
                        mappedEnd = Math.max(mappedStart, clamp(located.end));
                        baseSlice = base.substring(mappedStart, mappedEnd);
                    }
                }

                if (annText && !this._agreementTextEquivalent(baseSlice, annText)) {
                    continue;
                }
                if (baseToTargetMapper) {
                    const roundTripStart = Math.max(0, Number(baseToTargetMapper(mappedStart) || 0));
                    const roundTripEnd = Math.max(roundTripStart, Number(baseToTargetMapper(mappedEnd) || roundTripStart));
                    const drift = Math.max(
                        Math.abs(roundTripStart - targetStart),
                        Math.abs(roundTripEnd - targetEnd)
                    );
                    const allowedDrift = Math.max(2, Math.floor((targetEnd - targetStart) * 0.35));
                    if (drift > allowedDrift) {
                        continue;
                    }
                }

                const key = `${mappedStart}:${mappedEnd}:${ref}`;
                if (seen.has(key)) continue;
                seen.add(key);
                out.push({
                    start: mappedStart,
                    end: mappedEnd,
                    text: base.substring(mappedStart, mappedEnd),
                    ref,
                    entityId: String(ann?.entityId || ''),
                    entityType: String(ann?.entityType || this._agreementInferEntityTypeFromRef(ref)),
                    attribute: String(ann?.attribute || ''),
                });
            }

            out.sort((x, y) => (Number(x.start || 0) - Number(y.start || 0)) || (Number(x.end || 0) - Number(y.end || 0)) || String(x.ref || '').localeCompare(String(y.ref || '')));
            return out;
        },

        _agreementAnchorVariants(text, side = 'suffix') {
            const raw = String(text || '');
            if (!raw) return [];

            const out = [];
            const seen = new Set();
            const sizes = [32, 24, 18, 12, 8, 5];
            for (const size of sizes) {
                if (raw.length < size) continue;
                const piece = side === 'prefix'
                    ? raw.substring(0, size)
                    : raw.substring(raw.length - size);
                if (!piece.trim() || seen.has(piece)) continue;
                seen.add(piece);
                out.push(piece);
            }
            if (raw.trim() && !seen.has(raw)) {
                out.push(raw);
            }
            return out;
        },

        _agreementFindOccurrences(haystack, needle, limit = 32) {
            const source = String(haystack || '');
            const target = String(needle || '');
            if (!source || !target) return [];

            const out = [];
            let cursor = 0;
            while (out.length < limit) {
                const idx = source.indexOf(target, cursor);
                if (idx < 0) break;
                out.push(idx);
                cursor = idx + 1;
            }
            return out;
        },

        _agreementLocateConflictRangeInCurrentPlain(conflict, currentPlain, opts = {}) {
            const plain = String(currentPlain || '');
            const basePlain = String(this.agreementWorkspace.merge.plainText || '');
            if (!plain || !basePlain || !conflict) return null;

            const baseStart = this._conflictBaseStart(conflict);
            const baseEnd = this._conflictBaseEnd(conflict);
            const clamp = (value) => Math.max(0, Math.min(Number(value || 0), plain.length));
            const fallback = opts.fallbackRange || { start: baseStart, end: baseEnd };
            const fallbackStart = clamp(fallback.start);
            const fallbackEnd = Math.max(fallbackStart, clamp(fallback.end));

            const storedStart = clamp(
                Object.prototype.hasOwnProperty.call(conflict, 'current_start')
                    ? conflict.current_start
                    : fallbackStart
            );
            const storedEnd = Math.max(
                storedStart,
                clamp(
                    Object.prototype.hasOwnProperty.call(conflict, 'current_end')
                        ? conflict.current_end
                        : fallbackEnd
                )
            );
            let hintStart = fallbackStart;
            let hintEnd = fallbackEnd;
            const storedDrift = Math.abs(storedStart - fallbackStart) + Math.abs(storedEnd - fallbackEnd);
            const hintTolerance = Math.max(24, Math.min(160, ((fallbackEnd - fallbackStart) * 2) + 12));
            if (storedDrift <= hintTolerance) {
                hintStart = storedStart;
                hintEnd = storedEnd;
            }
            const expectedLen = Math.max(0, fallbackEnd - fallbackStart);
            const candidates = [];

            const pushCandidate = (start, end, bonus = 0) => {
                const safeStart = clamp(start);
                const safeEnd = Math.max(safeStart, clamp(end));
                const distance = Math.abs(safeStart - hintStart) + Math.abs(safeEnd - hintEnd);
                const lengthPenalty = expectedLen > 0 ? Math.abs((safeEnd - safeStart) - expectedLen) : 0;
                candidates.push({
                    start: safeStart,
                    end: safeEnd,
                    score: distance + (lengthPenalty * 0.5) - bonus,
                });
            };

            pushCandidate(fallbackStart, fallbackEnd);

            const baseSpan = basePlain.substring(baseStart, baseEnd);
            if (baseSpan) {
                for (const idx of this._agreementFindOccurrences(plain, baseSpan)) {
                    pushCandidate(idx, idx + baseSpan.length, Math.min(48, baseSpan.length));
                }
            }

            const beforeAnchors = this._agreementAnchorVariants(
                basePlain.substring(Math.max(0, baseStart - 40), baseStart),
                'suffix'
            );
            const afterAnchors = this._agreementAnchorVariants(
                basePlain.substring(baseEnd, Math.min(basePlain.length, baseEnd + 40)),
                'prefix'
            );
            const textChoices = Array.from(new Set([
                String(conflict.a_plain_text || ''),
                String(conflict.b_plain_text || ''),
            ])).filter((value) => value.length > 0);
            for (const choiceText of textChoices) {
                for (const idx of this._agreementFindOccurrences(plain, choiceText)) {
                    pushCandidate(idx, idx + choiceText.length, Math.min(64, choiceText.length + 20));
                }
            }
            const reviewerBeforeAnchors = this._agreementAnchorVariants(
                String(conflict.a_context_before || ''),
                'suffix'
            ).concat(this._agreementAnchorVariants(
                String(conflict.b_context_before || ''),
                'suffix'
            ));
            const reviewerAfterAnchors = this._agreementAnchorVariants(
                String(conflict.a_context_after || ''),
                'prefix'
            ).concat(this._agreementAnchorVariants(
                String(conflict.b_context_after || ''),
                'prefix'
            ));
            const mergedBeforeAnchors = Array.from(new Set(beforeAnchors.concat(reviewerBeforeAnchors))).filter(Boolean);
            const mergedAfterAnchors = Array.from(new Set(afterAnchors.concat(reviewerAfterAnchors))).filter(Boolean);

            for (const before of mergedBeforeAnchors) {
                const beforeMatches = this._agreementFindOccurrences(plain, before);
                if (beforeMatches.length === 0) continue;
                for (const beforeIdx of beforeMatches) {
                    const start = beforeIdx + before.length;
                    pushCandidate(start, start, Math.min(32, before.length));
                }
                for (const after of mergedAfterAnchors) {
                    const afterMatches = this._agreementFindOccurrences(plain, after);
                    if (afterMatches.length === 0) continue;
                    for (const beforeIdx of beforeMatches) {
                        const start = beforeIdx + before.length;
                        let bestAfter = null;
                        let bestScore = Number.POSITIVE_INFINITY;
                        for (const afterIdx of afterMatches) {
                            if (afterIdx < start) continue;
                            const score =
                                Math.abs(start - hintStart) +
                                Math.abs(afterIdx - hintEnd) +
                                Math.abs((afterIdx - start) - expectedLen);
                            if (score < bestScore) {
                                bestScore = score;
                                bestAfter = afterIdx;
                            }
                        }
                        if (bestAfter !== null) {
                            pushCandidate(start, bestAfter, before.length + after.length);
                        }
                    }
                }
            }
            for (const after of mergedAfterAnchors) {
                const afterMatches = this._agreementFindOccurrences(plain, after);
                for (const afterIdx of afterMatches) {
                    pushCandidate(afterIdx, afterIdx, Math.min(32, after.length));
                }
            }

            if (candidates.length === 0) {
                return { start: fallbackStart, end: fallbackEnd };
            }
            candidates.sort((a, b) => (a.score - b.score) || (a.start - b.start) || (a.end - b.end));
            return { start: candidates[0].start, end: candidates[0].end };
        },

        _collectAgreementLocalAnnotations(parsedDoc, rangeStart, rangeEnd) {
            const parsed = parsedDoc && typeof parsedDoc === 'object' ? parsedDoc : { plainText: '', annotations: [] };
            const start = Math.max(0, Number(rangeStart || 0));
            const end = Math.max(start, Number(rangeEnd || start));
            const out = [];
            for (const ann of parsed.annotations || []) {
                const annStart = Number(ann.start || 0);
                const annEnd = Number(ann.end || annStart);
                if (annEnd <= annStart) continue;
                if (annStart < start || annEnd > end) continue;
                out.push({
                    start: annStart - start,
                    end: annEnd - start,
                    text: String(parsed.plainText || '').substring(annStart, annEnd),
                    ref: String(ann.ref || '').trim(),
                    entityId: String(ann.entityId || ''),
                    entityType: String(ann.entityType || ''),
                    attribute: ann.attribute ? String(ann.attribute) : '',
                });
            }
            return out;
        },

        _groupAgreementTextEdits(baseText, editsA, editsB) {
            const base = String(baseText || '');
            const all = [];
            for (const item of editsA || []) all.push({ side: 'a', ...item });
            for (const item of editsB || []) all.push({ side: 'b', ...item });
            all.sort((x, y) => (Number(x.baseStart || 0) - Number(y.baseStart || 0)) || (Number(x.baseEnd || 0) - Number(y.baseEnd || 0)));
            if (all.length === 0) return [];

            const groups = [];
            let current = null;
            for (const op of all) {
                const start = Number(op.baseStart || 0);
                const end = Number(op.baseEnd || start);
                if (!current) {
                    current = {
                        baseStart: start,
                        baseEnd: end,
                        aOps: [],
                        bOps: [],
                    };
                } else {
                    const gapStart = Number(current.baseEnd || 0);
                    const gap = Math.max(0, start - gapStart);
                    const between = gap > 0 ? base.substring(gapStart, start) : '';
                    const closeEnough = gap <= 3 && /^[\s,.;:!?()'"-]*$/.test(between);
                    if (start > Number(current.baseEnd || 0) && !closeEnough) {
                        groups.push(current);
                        current = {
                            baseStart: start,
                            baseEnd: end,
                            aOps: [],
                            bOps: [],
                        };
                    } else {
                        current.baseStart = Math.min(Number(current.baseStart || 0), start);
                        current.baseEnd = Math.max(Number(current.baseEnd || 0), end);
                    }
                }
                if (op.side === 'a') current.aOps.push(op);
                else current.bOps.push(op);
            }
            if (current) groups.push(current);
            return groups;
        },

        _conflictBaseStart(conflict) {
            const value = conflict && Object.prototype.hasOwnProperty.call(conflict, 'base_start')
                ? conflict.base_start
                : conflict?.start;
            return Math.max(0, Number(value || 0));
        },

        _conflictBaseEnd(conflict) {
            const fallback = this._conflictBaseStart(conflict);
            const value = conflict && Object.prototype.hasOwnProperty.call(conflict, 'base_end')
                ? conflict.base_end
                : conflict?.end;
            return Math.max(fallback, Number(value || fallback));
        },

        _conflictCurrentStart(conflict) {
            if (conflict && Object.prototype.hasOwnProperty.call(conflict, 'current_start')) {
                return Math.max(0, Number(conflict.current_start || 0));
            }
            return this._conflictBaseStart(conflict);
        },

        _conflictCurrentEnd(conflict) {
            if (conflict && Object.prototype.hasOwnProperty.call(conflict, 'current_end')) {
                const start = this._conflictCurrentStart(conflict);
                return Math.max(start, Number(conflict.current_end || start));
            }
            return this._conflictBaseEnd(conflict);
        },

        _rebuildAnnotatedTextFromPlain(plainText, annotations) {
            const plain = String(plainText || '');
            const anns = Array.isArray(annotations) ? annotations.slice() : [];
            anns.sort((a, b) => (a.start - b.start) || (a.end - b.end) || String(a.ref || '').localeCompare(String(b.ref || '')));

            const deduped = [];
            const seen = new Set();
            for (const ann of anns) {
                const key = `${ann.start}:${ann.end}:${ann.ref}`;
                if (seen.has(key)) continue;
                seen.add(key);
                deduped.push(ann);
            }

            let raw = '';
            let cursor = 0;
            for (const ann of deduped) {
                const start = Number(ann.start);
                const end = Number(ann.end);
                if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
                if (start < cursor || end < start || end > plain.length) continue;
                raw += plain.substring(cursor, start);
                raw += `[${plain.substring(start, end)}; ${String(ann.ref || '').trim()}]`;
                cursor = end;
            }
            raw += plain.substring(cursor);
            return raw;
        },

        _annotationKey(ann) {
            return `${ann.start}:${ann.end}:${ann.ref}:${ann.text}`;
        },

        _spanKey(start, end) {
            return `${start}:${end}`;
        },

        _setConflictSelectedState(conflictId, selected) {
            for (const conflict of this.agreementWorkspace.merge.conflicts || []) {
                if (conflict.id === conflictId) {
                    conflict.selected = selected;
                    break;
                }
            }
        },

        _normalizeAgreementRefs(refs) {
            const seen = new Set();
            const out = [];
            for (const item of refs || []) {
                const value = String(item || '').trim();
                if (!value || seen.has(value)) continue;
                seen.add(value);
                out.push(value);
            }
            out.sort((a, b) => a.localeCompare(b));
            return out;
        },

        _areAgreementRefListsEqual(aRefs, bRefs) {
            const a = Array.isArray(aRefs) ? aRefs : [];
            const b = Array.isArray(bRefs) ? bRefs : [];
            if (a.length !== b.length) return false;
            for (let idx = 0; idx < a.length; idx++) {
                if (a[idx] !== b[idx]) return false;
            }
            return true;
        },

        _formatAgreementRefs(refs, limit = 2) {
            const items = Array.isArray(refs) ? refs : [];
            if (items.length === 0) return 'No annotation';
            const head = items.slice(0, limit).join(', ');
            if (items.length <= limit) return head;
            return `${head} +${items.length - limit}`;
        },

        _agreementRefsSignature(refs) {
            return this._normalizeAgreementRefs(refs || []).join('||');
        },

        _refsBySpanFromDocument(rawDocument) {
            const parsed = this._parseAgreementAnnotatedText(rawDocument || '');
            const spanMap = new Map();
            for (const ann of parsed.annotations || []) {
                const key = this._spanKey(ann.start, ann.end);
                if (!spanMap.has(key)) spanMap.set(key, []);
                spanMap.get(key).push(String(ann.ref || '').trim());
            }
            for (const [key, values] of spanMap.entries()) {
                spanMap.set(key, this._normalizeAgreementRefs(values));
            }
            return spanMap;
        },

        _agreementExpectedTextForChoice(conflict, choice) {
            if (!conflict || conflict.kind !== 'text_edit') return '';
            if (choice === 'b') return String(conflict.b_plain_text || '');
            return String(conflict.a_plain_text || '');
        },

        _agreementTextEquivalent(left, right) {
            const a = String(left ?? '');
            const b = String(right ?? '');
            if (a === b) return true;
            if (a.trim() === b.trim()) return true;
            const normalize = (value) => String(value || '').replace(/\s+/g, ' ').trim();
            return normalize(a) === normalize(b);
        },

        _agreementTextDeltaFromDecision(conflict) {
            if (!conflict || conflict.kind !== 'text_edit') return 0;
            const baseLen = this._conflictBaseEnd(conflict) - this._conflictBaseStart(conflict);
            const id = String(conflict.id || '');
            const decision = String(this.agreementWorkspace.merge.decisions?.[id] || '');
            if (decision === 'a' || decision === 'b') {
                return this._agreementExpectedTextForChoice(conflict, decision).length - baseLen;
            }
            if (decision === 'manual') {
                const manualText = String(this.agreementWorkspace.merge.manualTextResolutions?.[id] || '');
                return manualText.length - baseLen;
            }
            return 0;
        },

        _agreementResolvedTextDeltaBefore(basePos, excludeIds = new Set()) {
            const pos = Math.max(0, Number(basePos || 0));
            const list = (this.agreementWorkspace.merge.conflicts || [])
                .filter((item) => item?.kind === 'text_edit')
                .slice()
                .sort((a, b) => (this._conflictBaseStart(a) - this._conflictBaseStart(b)) || (this._conflictBaseEnd(a) - this._conflictBaseEnd(b)));
            let delta = 0;
            for (const conflict of list) {
                const id = String(conflict.id || '');
                if (excludeIds.has(id)) continue;
                const baseStart = this._conflictBaseStart(conflict);
                const baseEnd = this._conflictBaseEnd(conflict);
                const isInsertion = baseEnd === baseStart;
                if (isInsertion) {
                    if (baseStart >= pos) break;
                } else if (baseEnd > pos) {
                    break;
                }
                delta += this._agreementTextDeltaFromDecision(conflict);
            }
            return delta;
        },

        _agreementCurrentRangeForConflict(conflict, opts = {}) {
            const mapper = typeof opts.mapper === 'function'
                ? opts.mapper
                : (typeof this.agreementWorkspace?.merge?.baseToCurrentMapper === 'function'
                    ? this.agreementWorkspace.merge.baseToCurrentMapper
                    : null);
            if (opts.useDecisionProjection !== true) {
                const baseStart = this._conflictBaseStart(conflict);
                const baseEnd = this._conflictBaseEnd(conflict);
                const fallbackRange = mapper
                    ? {
                        start: Math.max(0, Number(mapper(baseStart) || 0)),
                        end: Math.max(0, Number(mapper(baseEnd) || 0)),
                    }
                    : {
                        start: baseStart,
                        end: baseEnd,
                    };
                const currentPlain = Object.prototype.hasOwnProperty.call(opts, 'currentPlain')
                    ? String(opts.currentPlain || '')
                    : String(this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '').plainText || '');
                const located = this._agreementLocateConflictRangeInCurrentPlain(conflict, currentPlain, { fallbackRange });
                if (located) return located;
            }
            const exclude = new Set(this._normalizeAgreementConflictIds(opts.excludeIds || []));
            const baseStart = this._conflictBaseStart(conflict);
            const baseEnd = this._conflictBaseEnd(conflict);
            const delta = this._agreementResolvedTextDeltaBefore(baseStart, exclude);
            return {
                start: Math.max(0, baseStart + delta),
                end: Math.max(0, baseEnd + delta),
            };
        },

        _agreementProjectBasePosToCurrent(basePos, opts = {}) {
            const mapper = typeof opts.mapper === 'function'
                ? opts.mapper
                : (typeof this.agreementWorkspace?.merge?.baseToCurrentMapper === 'function'
                    ? this.agreementWorkspace.merge.baseToCurrentMapper
                    : null);
            if (mapper && opts.useDecisionProjection !== true) {
                return Math.max(0, Number(mapper(basePos) || 0));
            }
            const exclude = new Set(this._normalizeAgreementConflictIds(opts.excludeIds || []));
            const pos = Math.max(0, Number(basePos || 0));
            return Math.max(0, pos + this._agreementResolvedTextDeltaBefore(pos, exclude));
        },

        _agreementCanInferResolutionFromCurrent() {
            if (!this.agreementWorkspace || !this.agreementWorkspace.active) return false;
            if (this.agreementWorkspace.disableCurrentInference || this.isAgreementContestMode()) return false;
            if (this.agreementWorkspace.inferResolutionFromCurrent) return true;
            const status = String(this.agreementWorkspace.status || '').toLowerCase();
            if (status === 'resolved') return true;
            return String(this.agreementWorkspace.loadedVariant || '').toLowerCase() === 'final';
        },

        _resolveAgreementConflictState(conflict, currentRefs, currentText = '') {
            const id = String(conflict?.id || '');
            const decision = String(this.agreementWorkspace.merge.decisions?.[id] || '');
            const allowCurrentInference = this._agreementCanInferResolutionFromCurrent();
            if (conflict?.kind === 'text_edit') {
                const current = String(currentText || '');
                const expectedA = this._agreementExpectedTextForChoice(conflict, 'a');
                const expectedB = this._agreementExpectedTextForChoice(conflict, 'b');
                const matchesA = this._agreementTextEquivalent(current, expectedA);
                const matchesB = this._agreementTextEquivalent(current, expectedB);
                if (decision === 'a' || decision === 'b') {
                    if (decision === 'a' && matchesA) return 'a';
                    if (decision === 'b' && matchesB) return 'b';
                }
                if (decision === 'manual') {
                    const manualText = String(this.agreementWorkspace.merge.manualTextResolutions?.[id] || '');
                    if (this._agreementTextEquivalent(manualText, current)) return 'custom';
                }
                // Infer from editor content only when loading a saved final/admin draft state.
                if (allowCurrentInference) {
                    if (matchesA && !matchesB) return 'a';
                    if (matchesB && !matchesA) return 'b';
                    if (matchesA && matchesB) return 'custom';
                }
                return 'unresolved';
            }

            const refs = this._normalizeAgreementRefs(currentRefs || []);
            const aRefs = this._normalizeAgreementRefs(conflict?.a_refs || []);
            const bRefs = this._normalizeAgreementRefs(conflict?.b_refs || []);
            const matchesA = this._areAgreementRefListsEqual(refs, aRefs);
            const matchesB = this._areAgreementRefListsEqual(refs, bRefs);
            const manualSig = this.agreementWorkspace.merge.manualResolutions?.[String(conflict?.id || '')];
            const currentSig = this._agreementRefsSignature(refs);

            if (decision === 'a' && matchesA) return 'a';
            if (decision === 'b' && matchesB) return 'b';
            if (decision === 'manual') {
                if (manualSig !== undefined && manualSig === currentSig) return 'custom';
            }
            // Infer from editor content only when loading a saved final/admin draft state.
            if (allowCurrentInference) {
                if (matchesA && !matchesB) return 'a';
                if (matchesB && !matchesA) return 'b';
                if (matchesA && matchesB) return 'custom';
            }
            return 'unresolved';
        },

        _recomputeAgreementConflictStates() {
            if (!this.agreementWorkspace.active) return;
            const merge = this.agreementWorkspace.merge;
            const conflicts = merge.conflicts || [];
            if (conflicts.length === 0) {
                merge.selectedConflictId = '';
                merge.inlineStats = {
                    agreed: merge.agreedCount || 0,
                    total: 0,
                    resolved: 0,
                    remaining: 0,
                };
                this._renderAgreementInlineCompareHtml();
                return;
            }

            const currentParsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
            const currentPlain = String(currentParsed?.plainText || '');
            const basePlain = String(merge.plainText || '');
            if (basePlain === currentPlain) {
                merge.baseToCurrentMapper = (pos) => Math.max(0, Number(pos || 0));
            } else {
                const liveEdits = this._computeAgreementTextEdits(basePlain, currentPlain);
                merge.baseToCurrentMapper = this._buildAgreementBaseToTargetMapper(liveEdits);
            }
            const currentSpanRefs = new Map();
            for (const ann of currentParsed.annotations || []) {
                const key = this._spanKey(ann.start, ann.end);
                if (!currentSpanRefs.has(key)) currentSpanRefs.set(key, []);
                currentSpanRefs.get(key).push(String(ann.ref || '').trim());
            }
            for (const [key, refs] of currentSpanRefs.entries()) {
                currentSpanRefs.set(key, this._normalizeAgreementRefs(refs));
            }

            const retainedConflicts = [];
            const removedConflictIds = [];
            for (const conflict of conflicts) {
                const projected = this._agreementCurrentRangeForConflict(conflict, { currentPlain });
                let start = Math.max(0, Math.min(Number(projected.start || 0), currentPlain.length));
                let end = Math.max(start, Math.min(Number(projected.end || start), currentPlain.length));
                const conflictId = String(conflict?.id || '');
                const decision = String(this.agreementWorkspace.merge.decisions?.[conflictId] || '');

                let refs = [];
                if (conflict?.kind === 'text_edit') {
                    refs = [];
                    if (decision === 'a' || decision === 'b') {
                        const expected = this._agreementExpectedTextForChoice(conflict, decision);
                        end = Math.max(start, Math.min(currentPlain.length, start + expected.length));
                    } else if (decision === 'manual') {
                        const manualText = String(this.agreementWorkspace.merge.manualTextResolutions?.[conflictId] || '');
                        end = Math.max(start, Math.min(currentPlain.length, start + manualText.length));
                    }
                } else {
                    refs = currentSpanRefs.get(this._spanKey(start, end)) || [];
                }
                let currentText = currentPlain.substring(start, end);

                const state = this._resolveAgreementConflictState(conflict, refs, currentText);
                const spanDeleted = end <= start && refs.length === 0 && !String(currentText || '').trim();
                if (spanDeleted) {
                    removedConflictIds.push(conflictId);
                    continue;
                }
                if (conflict?.kind === 'text_edit' && end <= start && currentPlain.length > 0) {
                    if (start >= currentPlain.length) {
                        start = currentPlain.length - 1;
                        end = currentPlain.length;
                    } else {
                        end = Math.min(currentPlain.length, start + 1);
                    }
                }

                conflict.current_start = start;
                conflict.current_end = end;
                conflict.current_refs = refs;
                conflict.current_text = currentText;
                conflict.resolution = state;
                conflict.selected = state === 'a' ? 'a' : (state === 'b' ? 'b' : (state === 'custom' ? 'custom' : ''));
                retainedConflicts.push(conflict);
            }

            if (removedConflictIds.length > 0) {
                for (const conflictId of removedConflictIds) {
                    delete merge.decisions[conflictId];
                    delete merge.selectionMode[conflictId];
                    delete merge.manualResolutions[conflictId];
                    delete merge.manualTextResolutions[conflictId];
                }
                merge.conflicts = retainedConflicts;
            }

            const effectiveConflicts = merge.conflicts || [];
            const unresolved = effectiveConflicts.filter((item) => item.resolution === 'unresolved');
            const resolved = effectiveConflicts.length - unresolved.length;
            merge.inlineStats = {
                agreed: merge.agreedCount || 0,
                total: effectiveConflicts.length,
                resolved,
                remaining: unresolved.length,
            };

            const selectedId = String(merge.selectedConflictId || '');
            const exists = effectiveConflicts.some((item) => item.id === selectedId);
            if (unresolved.length === 0) {
                merge.selectedConflictId = '';
            } else if (!exists || !unresolved.some((item) => item.id === selectedId)) {
                merge.selectedConflictId = unresolved[0].id;
            }

            const modal = this.agreementWorkspace.conflictModal;
            if (modal?.show) {
                const openIds = this._normalizeAgreementConflictIds(
                    Array.isArray(modal.conflictIds) && modal.conflictIds.length > 0
                        ? modal.conflictIds
                        : (modal.conflictId || '')
                );
                const unresolvedOpenIds = openIds.filter((id) => (
                    effectiveConflicts.some((item) => item.id === id && item.resolution === 'unresolved')
                ));
                if (unresolvedOpenIds.length === 0) {
                    this.closeAgreementConflictModal();
                } else {
                    modal.conflictIds = unresolvedOpenIds;
                    modal.conflictId = unresolvedOpenIds[0];
                }
            }

            this._renderAgreementInlineCompareHtml();
            this.$nextTick(() => this.attachAnnotationHoverListeners());
        },

        handleWindowResizeOrScroll() {
            if (this.popup?.show) {
                this.$nextTick(() => this._repositionAnnotationPopup());
            }
            if (!this.agreementWorkspace.active) return;
            if (this.agreementWorkspace.conflictModal.show) {
                this.$nextTick(() => this._positionAgreementConflictPopover(null));
            }
            if (this.agreementWorkspace.questionConflictModal.show) {
                this.$nextTick(() => this._positionAgreementQuestionConflictPopover(null));
            }
        },

        agreementCompareVariantKey(side) {
            const comparison = this.agreementWorkspace?.comparison || {};
            if (side === 'b') {
                const key = String(comparison.sideBKey || '').trim().toLowerCase();
                return key || 'reviewer_b';
            }
            const key = String(comparison.sideAKey || '').trim().toLowerCase();
            return key || 'reviewer_a';
        },

        isAgreementContestMode() {
            return String(this.agreementWorkspace?.comparison?.mode || '').trim().toLowerCase() === 'contest';
        },

        agreementReviewerLabel(side) {
            if (side === 'source') {
                return String(this.agreementWorkspace.versions?.source?.username || 'Opus');
            }
            const key = this.agreementCompareVariantKey(side);
            const variant = this.agreementWorkspace.versions?.[key] || null;
            const fallbackByKey = {
                reviewer_a: 'Reviewer A',
                reviewer_b: 'Reviewer B',
                final: 'Final',
                source: 'Opus',
            };
            return String(variant?.username || variant?.label || fallbackByKey[key] || 'Reviewer');
        },

        agreementConflictStatusLabel(conflict) {
            const status = String(conflict?.resolution || 'unresolved');
            if (status === 'a') return `Accepted ${this.agreementReviewerLabel('a')}`;
            if (status === 'b') return `Accepted ${this.agreementReviewerLabel('b')}`;
            if (status === 'custom') return 'Merged/Edited';
            return 'Unresolved';
        },

        agreementConflictStatusClass(conflict) {
            const status = String(conflict?.resolution || 'unresolved');
            if (status === 'a' || status === 'b' || status === 'custom') return 'badge-validated';
            return 'badge-in_progress';
        },

        agreementConflictPreviewText(conflict, maxLen = 84) {
            const preview = conflict?.kind === 'text_edit'
                ? String(conflict?.base_text || conflict?.text || '')
                : String(conflict?.text || '');
            const text = preview.replace(/\s+/g, ' ').trim() || '(empty span)';
            if (text.length <= maxLen) return text;
            return `${text.substring(0, maxLen)}...`;
        },

        agreementConflictCurrentRefsText(conflict) {
            if (conflict?.kind === 'text_edit') {
                const text = String(conflict?.current_text || '');
                const normalized = text.replace(/\s+/g, ' ').trim() || '(empty)';
                return normalized.length > 72 ? `${normalized.substring(0, 72)}...` : normalized;
            }
            return this._formatAgreementRefs(conflict?.current_refs || [], 5);
        },

        agreementConflictCurrentLabel(conflict) {
            return conflict?.kind === 'text_edit' ? 'Current text:' : 'Current refs:';
        },

        agreementTotalConflictCount() {
            return (this.agreementWorkspace.merge.conflicts || []).length;
        },

        agreementResolvedConflictCount() {
            return (this.agreementWorkspace.merge.conflicts || []).filter((item) => item.resolution !== 'unresolved').length;
        },

        agreementRemainingConflictCount() {
            return (this.agreementWorkspace.merge.conflicts || []).filter((item) => item.resolution === 'unresolved').length;
        },

        agreementIncludesStructuredConflicts() {
            return this.reviewTarget === 'rules' || this.reviewTarget === 'questions';
        },
        agreementIncludesRuleStructuredConflicts() {
            return this.reviewTarget === 'rules';
        },

        agreementQuestionConflictTotalCount() {
            if (!this.agreementIncludesStructuredConflicts()) return 0;
            return (this.agreementWorkspace.structured.questionConflicts || []).length;
        },

        agreementQuestionConflictResolvedCount() {
            if (!this.agreementIncludesStructuredConflicts()) return 0;
            return (this.agreementWorkspace.structured.questionConflicts || []).filter((item) => item.resolution !== 'unresolved').length;
        },

        agreementQuestionConflictRemainingCount() {
            if (!this.agreementIncludesStructuredConflicts()) return 0;
            return (this.agreementWorkspace.structured.questionConflicts || []).filter((item) => item.resolution === 'unresolved').length;
        },

        agreementRuleConflictTotalCount() {
            if (!this.agreementIncludesRuleStructuredConflicts()) return 0;
            return (this.agreementWorkspace.structured.ruleConflicts || []).length;
        },

        agreementRuleConflictResolvedCount() {
            if (!this.agreementIncludesRuleStructuredConflicts()) return 0;
            return (this.agreementWorkspace.structured.ruleConflicts || []).filter((item) => item.resolution !== 'unresolved').length;
        },

        agreementRuleConflictRemainingCount() {
            if (!this.agreementIncludesRuleStructuredConflicts()) return 0;
            return (this.agreementWorkspace.structured.ruleConflicts || []).filter((item) => item.resolution === 'unresolved').length;
        },

        agreementStructuredTotalConflictCount() {
            return this.agreementQuestionConflictTotalCount() + this.agreementRuleConflictTotalCount();
        },

        agreementStructuredResolvedConflictCount() {
            return this.agreementQuestionConflictResolvedCount() + this.agreementRuleConflictResolvedCount();
        },

        agreementStructuredRemainingConflictCount() {
            return this.agreementQuestionConflictRemainingCount() + this.agreementRuleConflictRemainingCount();
        },

        agreementAllRemainingConflictCount() {
            return this.agreementRemainingConflictCount() + this.agreementStructuredRemainingConflictCount();
        },

        agreementHasUnresolvedConflicts() {
            return this.agreementAllRemainingConflictCount() > 0;
        },

        getAgreementUnresolvedConflicts() {
            return (this.agreementWorkspace.merge.conflicts || []).filter((item) => item.resolution === 'unresolved');
        },

        getAgreementUnresolvedQuestionConflicts() {
            if (!this.agreementIncludesStructuredConflicts()) return [];
            return (this.agreementWorkspace.structured.questionConflicts || []).filter((item) => item.resolution === 'unresolved');
        },

        getAgreementUnresolvedRuleConflicts() {
            if (!this.agreementIncludesRuleStructuredConflicts()) return [];
            return (this.agreementWorkspace.structured.ruleConflicts || []).filter((item) => item.resolution === 'unresolved');
        },

        _ruleConflictForSignature(signature, { includeResolved = false } = {}) {
            if (!this.agreementIncludesRuleStructuredConflicts()) return null;
            const normalized = String(signature || '').trim();
            if (!normalized) return null;
            const list = this.agreementWorkspace.structured.ruleConflicts || [];
            for (const conflict of list) {
                if (String(conflict?.rule_sig || '') !== normalized) continue;
                if (!includeResolved && conflict?.resolution !== 'unresolved') continue;
                return conflict;
            }
            return null;
        },

        ruleConflictForRule(rule, idx, { includeResolved = false } = {}) {
            const entry = this._normalizeRuleEntryForAgreement(rule);
            if (!entry) return null;
            return this._ruleConflictForSignature(entry.signature, { includeResolved });
        },

        ruleConflictIdForRule(rule, idx, { includeResolved = false } = {}) {
            const conflict = this.ruleConflictForRule(rule, idx, { includeResolved });
            return conflict ? String(conflict.id || '') : '';
        },

        ruleConflictIdForRuleAny(rule, idx) {
            return this.ruleConflictIdForRule(rule, idx, { includeResolved: true });
        },

        isRuleInUnresolvedConflict(rule, idx) {
            return !!this.ruleConflictForRule(rule, idx, { includeResolved: false });
        },

        isRuleInResolvedConflict(rule, idx) {
            const conflict = this.ruleConflictForRule(rule, idx, { includeResolved: true });
            return !!(conflict && conflict.resolution !== 'unresolved');
        },

        getAgreementUnresolvedRuleConflictsNotInCurrentDoc() {
            const currentMap = this._ruleEntryMapForAgreement(this.docData?.rules || []);
            return this.getAgreementUnresolvedRuleConflicts().filter(
                (conflict) => !currentMap.has(String(conflict?.rule_sig || ''))
            );
        },

        agreementRuleConflictDisplayText(conflict) {
            return String(
                conflict?.source_rule_text
                || conflict?.a_rule_text
                || conflict?.b_rule_text
                || conflict?.rule_text
                || ''
            ).trim() || 'rule';
        },

        _questionConflictForKey(key, { includeResolved = false } = {}) {
            if (!this.agreementIncludesStructuredConflicts()) return null;
            const normalized = String(key || '').trim();
            if (!normalized) return null;
            const list = this.agreementWorkspace.structured.questionConflicts || [];
            for (const conflict of list) {
                if (String(conflict?.key || '') !== normalized) continue;
                if (!includeResolved && conflict?.resolution !== 'unresolved') continue;
                return conflict;
            }
            return null;
        },

        questionConflictForQuestion(question, idx, { includeResolved = false } = {}) {
            if (!this.agreementWorkspace.active) return null;
            const key = this._questionConflictKeyFromQuestion(question, idx);
            return this._questionConflictForKey(key, { includeResolved });
        },

        questionConflictIdForQuestion(question, idx, { includeResolved = false } = {}) {
            const conflict = this.questionConflictForQuestion(question, idx, { includeResolved });
            return conflict ? String(conflict.id || '') : '';
        },

        questionConflictIdForQuestionAny(question, idx) {
            return this.questionConflictIdForQuestion(question, idx, { includeResolved: true });
        },

        isQuestionInUnresolvedConflict(question, idx) {
            return !!this.questionConflictForQuestion(question, idx, { includeResolved: false });
        },

        isQuestionInResolvedConflict(question, idx) {
            const conflict = this.questionConflictForQuestion(question, idx, { includeResolved: true });
            return !!(conflict && conflict.resolution !== 'unresolved');
        },

        getAgreementUnresolvedQuestionConflictsNotInCurrentDoc() {
            const currentMap = this._questionMapForAgreement(this.docData?.questions || []);
            return this.getAgreementUnresolvedQuestionConflicts().filter(
                (conflict) => !currentMap.has(String(conflict?.key || ''))
            );
        },

        agreementSelectedConflictPosition() {
            const selected = this.getAgreementSelectedConflict();
            if (!selected) return '0/0';
            const unresolved = this.getAgreementUnresolvedConflicts();
            const sourceList = unresolved.length > 0
                ? unresolved
                : (this.agreementWorkspace.merge.conflicts || []);
            const idx = sourceList.findIndex((item) => item.id === selected.id);
            if (idx < 0) return `1/${sourceList.length || 1}`;
            return `${idx + 1}/${sourceList.length}`;
        },

        _normalizeQuestionForAgreement(question, fallbackId = '') {
            const q = question && typeof question === 'object' ? question : {};
            const questionId = String(q.question_id || fallbackId || '').trim();
            const answerType = this._normalizeAnswerType(
                q.answer_type,
                q.is_answer_invariant
            );
            const reasoningChain = this._normalizeReasoningChain(q.reasoning_chain, q.reasoning_chain_text);
            return {
                question_id: questionId,
                question: String(q.question || ''),
                question_type: this._normalizeQuestionType(q.question_type),
                answer: answerType === 'refusal'
                    ? refusalAnswerLiteral
                    : String(q.answer ?? ''),
                answer_type: answerType,
                reasoning_chain: reasoningChain,
                reasoning_chain_text: reasoningChain.join('\n'),
            };
        },

        _cloneQuestionForAgreement(question, fallbackId = '') {
            return JSON.parse(JSON.stringify(this._normalizeQuestionForAgreement(question, fallbackId)));
        },

        _questionConflictKeyFromQuestion(question, idx = 0) {
            const qid = String(question?.question_id || '').trim();
            return qid || `__idx_${idx}`;
        },

        _questionAgreementSignature(question) {
            if (!question) return '__none__';
            const q = this._normalizeQuestionForAgreement(question);
            return JSON.stringify([
                q.question_id,
                q.question.trim(),
                q.question_type.trim(),
                q.answer.trim(),
                q.answer_type || 'variant',
                (q.reasoning_chain || []).join('||'),
            ]);
        },

        _questionLabelForConflict(conflict) {
            const key = String(conflict?.key || '').trim();
            if (!key) return 'question';
            if (key.startsWith('__idx_')) {
                const idx = Number(key.replace('__idx_', ''));
                return Number.isFinite(idx) ? `question #${idx + 1}` : 'question';
            }
            return key;
        },

        _questionMapForAgreement(questions) {
            const map = new Map();
            const list = Array.isArray(questions) ? questions : [];
            for (let idx = 0; idx < list.length; idx++) {
                const normalized = this._normalizeQuestionForAgreement(list[idx], '');
                const key = this._questionConflictKeyFromQuestion(normalized, idx);
                map.set(key, normalized);
            }
            return map;
        },

        _normalizeRuleExpressionForAgreement(expressionText) {
            return String(expressionText || '').replace(/\s+/g, '');
        },

        _normalizeRuleExplanationForAgreement(explanationText) {
            return String(explanationText || '').trim().replace(/\s+/g, ' ');
        },

        _normalizeRuleEntryForAgreement(ruleText) {
            const parts = this._splitRuleString(ruleText);
            const expression = this._normalizeRuleExpressionForAgreement(parts.expression);
            if (!expression) return null;
            const explanation = this._normalizeRuleExplanationForAgreement(parts.explanation);
            return {
                signature: JSON.stringify([expression, explanation]),
                rule_text: this._composeRuleString(parts.expression, parts.explanation),
                expression,
                explanation,
            };
        },

        _ruleEntryMapForAgreement(rules) {
            const out = new Map();
            const list = Array.isArray(rules) ? rules : [];
            for (const value of list) {
                const entry = this._normalizeRuleEntryForAgreement(value);
                if (!entry || out.has(entry.signature)) continue;
                out.set(entry.signature, entry);
            }
            return out;
        },

        _orderedRuleSignaturesForAgreement(...ruleLists) {
            const ordered = [];
            const seen = new Set();
            for (const rules of ruleLists) {
                const list = Array.isArray(rules) ? rules : [];
                for (const value of list) {
                    const entry = this._normalizeRuleEntryForAgreement(value);
                    if (!entry || seen.has(entry.signature)) continue;
                    seen.add(entry.signature);
                    ordered.push(entry.signature);
                }
            }
            return ordered;
        },

        _consensusRuleListForAgreement(sourceRules, aRules, bRules) {
            const sourceMap = this._ruleEntryMapForAgreement(sourceRules);
            const aMap = this._ruleEntryMapForAgreement(aRules);
            const bMap = this._ruleEntryMapForAgreement(bRules);
            const ordered = this._orderedRuleSignaturesForAgreement(sourceRules, aRules, bRules);
            const out = [];
            for (const signature of ordered) {
                const aEntry = aMap.get(signature) || null;
                const bEntry = bMap.get(signature) || null;
                if (!aEntry || !bEntry) continue;
                const chosen = sourceMap.get(signature) || aEntry || bEntry;
                out.push(chosen.rule_text);
            }
            return out;
        },

        _currentRuleEntryForConflict(conflict) {
            const explicitSignature = String(conflict?.rule_sig || '').trim();
            const fallbackEntry = this._normalizeRuleEntryForAgreement(conflict?.rule_text || '');
            const targetSignature = explicitSignature || String(fallbackEntry?.signature || '');
            if (!targetSignature) return null;
            const rules = Array.isArray(this.docData?.rules) ? this.docData.rules : [];
            for (const value of rules) {
                const entry = this._normalizeRuleEntryForAgreement(value);
                if (entry && entry.signature === targetSignature) return entry;
            }
            return null;
        },

        _buildRuleAgreementConsensusDocument(sourceDoc, aDoc, bDoc) {
            const baseCandidate = sourceDoc || this.docData || aDoc || bDoc || {};
            const normalized = this._normalizeEditableDocument(baseCandidate);
            const sourceRules = Array.isArray(sourceDoc?.rules) ? sourceDoc.rules : [];
            const aRules = Array.isArray(aDoc?.rules) ? aDoc.rules : [];
            const bRules = Array.isArray(bDoc?.rules) ? bDoc.rules : [];
            normalized.rules = this._consensusRuleListForAgreement(sourceRules, aRules, bRules);
            normalized.num_questions = Array.isArray(normalized.questions) ? normalized.questions.length : 0;
            return normalized;
        },

        _buildRuleAgreementWorkingDocument(sourceDoc, aDoc, bDoc, currentDoc) {
            const baseCandidate = currentDoc || sourceDoc || aDoc || bDoc || {};
            const normalized = this._normalizeEditableDocument(baseCandidate);
            const sourceRules = Array.isArray(sourceDoc?.rules) ? sourceDoc.rules : [];
            const aRules = Array.isArray(aDoc?.rules) ? aDoc.rules : [];
            const bRules = Array.isArray(bDoc?.rules) ? bDoc.rules : [];
            const currentRules = Array.isArray(currentDoc?.rules) ? currentDoc.rules : [];

            const sourceMap = this._ruleEntryMapForAgreement(sourceRules);
            const aMap = this._ruleEntryMapForAgreement(aRules);
            const bMap = this._ruleEntryMapForAgreement(bRules);
            const currentMap = this._ruleEntryMapForAgreement(currentRules);
            const orderedSignatures = this._orderedRuleSignaturesForAgreement(sourceRules, aRules, bRules, currentRules);

            const keptRuleTexts = [];
            const keptSignatures = new Set();
            const pushRule = (signature, preferredEntry = null) => {
                const normalizedSignature = String(signature || '').trim();
                if (!normalizedSignature || keptSignatures.has(normalizedSignature)) return;
                const entry = preferredEntry
                    || currentMap.get(normalizedSignature)
                    || sourceMap.get(normalizedSignature)
                    || aMap.get(normalizedSignature)
                    || bMap.get(normalizedSignature)
                    || null;
                if (!entry) return;
                keptSignatures.add(normalizedSignature);
                keptRuleTexts.push(entry.rule_text);
            };

            // Start from the saved draft when present, but never keep source-only rules
            // that both reviewers removed.
            for (const rawRule of currentRules) {
                const entry = this._normalizeRuleEntryForAgreement(rawRule);
                if (!entry) continue;
                const signature = entry.signature;
                const inSource = sourceMap.has(signature);
                const inA = aMap.has(signature);
                const inB = bMap.has(signature);
                if (inA && inB) {
                    pushRule(signature, sourceMap.get(signature) || aMap.get(signature) || bMap.get(signature) || entry);
                    continue;
                }
                if (inA !== inB) {
                    pushRule(signature, entry);
                    continue;
                }
                if (!inSource) {
                    pushRule(signature, entry);
                }
            }

            // Reviewer consensus should always be present in the working document.
            for (const signature of orderedSignatures) {
                if (!aMap.has(signature) || !bMap.has(signature)) continue;
                pushRule(signature, sourceMap.get(signature) || aMap.get(signature) || bMap.get(signature));
            }

            normalized.rules = keptRuleTexts;
            normalized.num_questions = Array.isArray(normalized.questions) ? normalized.questions.length : 0;
            return normalized;
        },

        _ruleAgreementSignatureForPresence(present) {
            return present ? 'present' : 'absent';
        },

        _structuredConflictById(conflictId) {
            if (!this.agreementIncludesStructuredConflicts()) return null;
            const id = String(conflictId || '').trim();
            if (!id) return null;
            const question = (this.agreementWorkspace.structured.questionConflicts || []).find((item) => item.id === id);
            if (question) return question;
            return (this.agreementWorkspace.structured.ruleConflicts || []).find((item) => item.id === id) || null;
        },

        _currentQuestionForConflict(conflict) {
            const key = String(conflict?.key || '');
            const currentMap = this._questionMapForAgreement(this.docData?.questions || []);
            return currentMap.get(key) || null;
        },

        _currentRulePresenceForConflict(conflict) {
            return !!this._currentRuleEntryForConflict(conflict);
        },

        isRuleConflictInCurrentDoc(conflict) {
            return !!this._currentRuleEntryForConflict(conflict);
        },

        agreementRuleConflictLocationLabel(conflict) {
            return this.isRuleConflictInCurrentDoc(conflict)
                ? 'present in current rules'
                : 'missing from current rules';
        },

        agreementRuleConflictCurrentActionLabel(conflict) {
            return this.isRuleConflictInCurrentDoc(conflict)
                ? 'Keep Current'
                : 'Keep Removed';
        },

        _resolveAgreementStructuredConflictState(conflict) {
            const structured = this.agreementWorkspace.structured || {};
            const decisions = structured.decisions || {};
            const manual = structured.manualResolutions || {};
            const id = String(conflict?.id || '');
            const decision = String(decisions[id] || '');
            const allowCurrentInference = this._agreementCanInferResolutionFromCurrent();

            if (conflict?.kind === 'question') {
                const currentQuestion = this._currentQuestionForConflict(conflict);
                const currentSig = this._questionAgreementSignature(currentQuestion);
                const matchesA = currentSig === String(conflict?.a_sig || '__none__');
                const matchesB = currentSig === String(conflict?.b_sig || '__none__');
                const aMissing = String(conflict?.a_sig || '__none__') === '__none__';
                const bMissing = String(conflict?.b_sig || '__none__') === '__none__';

                if (decision === 'a' && matchesA) return 'a';
                if (decision === 'b' && matchesB) return 'b';
                if (decision === 'manual' && manual[id] === currentSig) return 'custom';
                // If a reviewer variant is "no question", deleting the current question
                // should count as a resolved choice for that side (not bounce back to unresolved).
                if (currentSig === '__none__' && (aMissing || bMissing)) {
                    if (matchesA && !matchesB) return 'a';
                    if (matchesB && !matchesA) return 'b';
                    if (matchesA && matchesB) return 'custom';
                }
                if (allowCurrentInference) {
                    if (matchesA && !matchesB) return 'a';
                    if (matchesB && !matchesA) return 'b';
                    if (matchesA && matchesB) return 'custom';
                }
                return 'unresolved';
            }

            if (conflict?.kind === 'rule') {
                const currentPresent = this._currentRulePresenceForConflict(conflict);
                const currentSig = this._ruleAgreementSignatureForPresence(currentPresent);
                const matchesA = currentSig === String(conflict?.a_sig || 'absent');
                const matchesB = currentSig === String(conflict?.b_sig || 'absent');

                if (decision === 'a' && matchesA) return 'a';
                if (decision === 'b' && matchesB) return 'b';
                if (decision === 'manual' && manual[id] === currentSig) return 'custom';
                if (allowCurrentInference) {
                    if (matchesA && !matchesB) return 'a';
                    if (matchesB && !matchesA) return 'b';
                    if (matchesA && matchesB) return 'custom';
                }
                return 'unresolved';
            }

            return 'unresolved';
        },

        _recomputeAgreementStructuredConflictStates() {
            if (!this.agreementWorkspace.active) return;
            const structured = this.agreementWorkspace.structured;
            if (!structured) return;
            if (!this.agreementIncludesStructuredConflicts()) {
                structured.warning = '';
                structured.decisions = {};
                structured.manualResolutions = {};
                structured.questionConflicts = [];
                structured.ruleConflicts = [];
                structured.stats = {
                    question: { total: 0, resolved: 0, remaining: 0 },
                    rule: { total: 0, resolved: 0, remaining: 0 },
                };
                if (this.agreementWorkspace.questionConflictModal?.show) {
                    this.closeAgreementQuestionConflictModal();
                }
                return;
            }

            const applyState = (conflict) => {
                const state = this._resolveAgreementStructuredConflictState(conflict);
                conflict.resolution = state;
                conflict.selected = state === 'a' ? 'a' : (state === 'b' ? 'b' : (state === 'custom' ? 'custom' : ''));
            };

            for (const conflict of structured.questionConflicts || []) {
                applyState(conflict);
            }
            for (const conflict of structured.ruleConflicts || []) {
                applyState(conflict);
            }

            const questionModal = this.agreementWorkspace.questionConflictModal;
            if (questionModal?.show) {
                const openId = String(questionModal.conflictId || '').trim();
                const openConflict = openId
                    ? (structured.questionConflicts || []).find((item) => String(item.id || '') === openId)
                    : null;
                if (!openConflict || openConflict.resolution !== 'unresolved') {
                    this.closeAgreementQuestionConflictModal();
                } else {
                    this.$nextTick(() => this._positionAgreementQuestionConflictPopover(null));
                }
            }

            const qTotal = (structured.questionConflicts || []).length;
            const qResolved = (structured.questionConflicts || []).filter((item) => item.resolution !== 'unresolved').length;
            const rTotal = (structured.ruleConflicts || []).length;
            const rResolved = (structured.ruleConflicts || []).filter((item) => item.resolution !== 'unresolved').length;

            structured.stats = {
                question: {
                    total: qTotal,
                    resolved: qResolved,
                    remaining: Math.max(0, qTotal - qResolved),
                },
                rule: {
                    total: rTotal,
                    resolved: rResolved,
                    remaining: Math.max(0, rTotal - rResolved),
                },
            };
        },

        _buildAgreementStructuredConflictsFromAnnotators(aDoc, bDoc, sourceDoc) {
            this._ensureAgreementWorkspaceState();
            const structured = this.agreementWorkspace.structured;
            structured.warning = '';
            structured.decisions = {};
            structured.manualResolutions = {};
            structured.questionConflicts = [];
            structured.ruleConflicts = [];
            structured.qaCoverageExemptionsBySide = {
                source: this._qaCoverageExemptionMapFromDocument(sourceDoc),
                a: this._qaCoverageExemptionMapFromDocument(aDoc),
                b: this._qaCoverageExemptionMapFromDocument(bDoc),
            };
            structured.qaCoveragePairsBySide = {
                source: this._qaCoveragePresentPairMapFromDocument(sourceDoc),
                a: this._qaCoveragePresentPairMapFromDocument(aDoc),
                b: this._qaCoveragePresentPairMapFromDocument(bDoc),
            };
            structured.qaCoverageQuestionsBySide = {
                source: this._qaCoverageQuestionsByPairFromDocument(sourceDoc),
                a: this._qaCoverageQuestionsByPairFromDocument(aDoc),
                b: this._qaCoverageQuestionsByPairFromDocument(bDoc),
            };
            structured.stats = {
                question: { total: 0, resolved: 0, remaining: 0 },
                rule: { total: 0, resolved: 0, remaining: 0 },
            };
            if (!this.agreementIncludesStructuredConflicts()) {
                return;
            }

            const aQuestions = Array.isArray(aDoc?.questions) ? aDoc.questions : [];
            const bQuestions = Array.isArray(bDoc?.questions) ? bDoc.questions : [];
            const sourceQuestions = Array.isArray(sourceDoc?.questions) ? sourceDoc.questions : [];

            const mapA = this._questionMapForAgreement(aQuestions);
            const mapB = this._questionMapForAgreement(bQuestions);
            const mapSource = this._questionMapForAgreement(sourceQuestions);

            const questionKeys = [];
            const seenQuestionKeys = new Set();
            const pushQuestionKey = (key) => {
                const normalized = String(key || '').trim();
                if (!normalized || seenQuestionKeys.has(normalized)) return;
                seenQuestionKeys.add(normalized);
                questionKeys.push(normalized);
            };
            for (const key of mapSource.keys()) pushQuestionKey(key);
            for (const key of mapA.keys()) pushQuestionKey(key);
            for (const key of mapB.keys()) pushQuestionKey(key);

            for (const key of questionKeys) {
                const aItem = mapA.get(key) || null;
                const bItem = mapB.get(key) || null;
                const sourceItem = mapSource.get(key) || null;
                const aSig = this._questionAgreementSignature(aItem);
                const bSig = this._questionAgreementSignature(bItem);
                if (aSig === bSig) continue;
                structured.questionConflicts.push({
                    id: `q:${key}`,
                    kind: 'question',
                    key,
                    label: this._questionLabelForConflict({ key }),
                    source_item: sourceItem,
                    a_item: aItem,
                    b_item: bItem,
                    a_sig: aSig,
                    b_sig: bSig,
                    resolution: 'unresolved',
                    selected: '',
                });
            }

            const aRules = Array.isArray(aDoc?.rules) ? aDoc.rules : [];
            const bRules = Array.isArray(bDoc?.rules) ? bDoc.rules : [];
            const sourceRules = Array.isArray(sourceDoc?.rules) ? sourceDoc.rules : [];
            if (this.agreementIncludesRuleStructuredConflicts()) {
                const ruleMapA = this._ruleEntryMapForAgreement(aRules);
                const ruleMapB = this._ruleEntryMapForAgreement(bRules);
                const ruleMapSource = this._ruleEntryMapForAgreement(sourceRules);
                const orderedRuleSignatures = this._orderedRuleSignaturesForAgreement(sourceRules, aRules, bRules);

                for (let idx = 0; idx < orderedRuleSignatures.length; idx++) {
                    const signature = orderedRuleSignatures[idx];
                    const sourceEntry = ruleMapSource.get(signature) || null;
                    const aEntry = ruleMapA.get(signature) || null;
                    const bEntry = ruleMapB.get(signature) || null;
                    const inA = !!aEntry;
                    const inB = !!bEntry;
                    if (inA === inB) continue;
                    structured.ruleConflicts.push({
                        id: `r:${idx}`,
                        kind: 'rule',
                        index: idx,
                        rule_sig: signature,
                        rule_text: (sourceEntry || aEntry || bEntry || {}).rule_text || '',
                        source_rule_text: sourceEntry?.rule_text || '',
                        a_rule_text: aEntry?.rule_text || '',
                        b_rule_text: bEntry?.rule_text || '',
                        source_present: !!sourceEntry,
                        a_present: inA,
                        b_present: inB,
                        a_sig: this._ruleAgreementSignatureForPresence(inA),
                        b_sig: this._ruleAgreementSignatureForPresence(inB),
                        resolution: 'unresolved',
                        selected: '',
                    });
                }
            }

            this._recomputeAgreementStructuredConflictStates();
        },

        agreementQuestionConflictSummary(conflict, side) {
            const item = side === 'b' ? conflict?.b_item : conflict?.a_item;
            if (!item) return 'No question';
            const question = String(item.question || '').replace(/\s+/g, ' ').trim();
            const shortQ = question.length > 80 ? `${question.substring(0, 80)}...` : question;
            const answer = String(item.answer || '').trim() || '—';
            return `${shortQ} (type: ${item.question_type || 'extractive'}, answer: ${answer})`;
        },

        _agreementQuestionConflictPairKey(conflict) {
            return String(
                this._qaCoveragePairKeyFromQuestion(
                    conflict?.source_item || conflict?.a_item || conflict?.b_item || null
                ) || ''
            );
        },

        agreementQuestionConflictPairLabel(conflict) {
            const pairKey = this._agreementQuestionConflictPairKey(conflict);
            if (!pairKey) return 'Unknown pair';
            const [questionType, answerType] = pairKey.split('::');
            if (!questionType || !answerType) return 'Unknown pair';
            return `${this.qaCoverageQuestionTypeLabel(questionType)} + ${this.qaCoverageAnswerTypeLabel(answerType)}`;
        },

        _agreementQuestionConflictCoverageForSide(conflict, side) {
            const sideKey = String(side || '').trim().toLowerCase();
            const pairKey = this._agreementQuestionConflictPairKey(conflict);
            if (!pairKey || !['source', 'a', 'b'].includes(sideKey)) {
                return { state: 'unknown', text: 'Coverage context unavailable', justification: '' };
            }
            const structured = this.agreementWorkspace?.structured || {};
            const presentBySide = structured.qaCoveragePairsBySide || {};
            const exemptionsBySide = structured.qaCoverageExemptionsBySide || {};
            const hasPair = Boolean((presentBySide[sideKey] || {})[pairKey]);
            if (hasPair) {
                return {
                    state: 'covered',
                    text: 'Covered by another question in this submission',
                    justification: '',
                };
            }
            const justification = String((exemptionsBySide[sideKey] || {})[pairKey] || '').trim();
            if (justification) {
                return {
                    state: 'justified',
                    text: 'Ignored with justification',
                    justification,
                };
            }
            return {
                state: 'missing',
                text: 'Missing pair with no ignore justification',
                justification: '',
            };
        },

        agreementQuestionConflictCoverageState(conflict, side) {
            return this._agreementQuestionConflictCoverageForSide(conflict, side).state;
        },

        agreementQuestionConflictCoverageText(conflict, side) {
            return this._agreementQuestionConflictCoverageForSide(conflict, side).text;
        },

        agreementQuestionConflictCoverageJustification(conflict, side) {
            return this._agreementQuestionConflictCoverageForSide(conflict, side).justification;
        },

        agreementQuestionConflictCoverageCandidates(conflict, side) {
            const sideKey = String(side || '').trim().toLowerCase();
            const pairKey = this._agreementQuestionConflictPairKey(conflict);
            if (!pairKey || !['source', 'a', 'b'].includes(sideKey)) return [];
            const bySide = this.agreementWorkspace?.structured?.qaCoverageQuestionsBySide || {};
            const byPair = bySide?.[sideKey] || {};
            const list = Array.isArray(byPair?.[pairKey]) ? byPair[pairKey] : [];
            if (!list.length) return [];

            const conflictKey = String(conflict?.key || '').trim();
            const filtered = conflictKey
                ? list.filter((item) => String(item?.key || '').trim() !== conflictKey)
                : list;
            return (filtered.length ? filtered : list).map((item) => ({
                key: String(item?.key || '').trim(),
                question_id: String(item?.question_id || '').trim(),
                question: String(item?.question || ''),
                answer: String(item?.answer || ''),
                question_type: this._normalizeQuestionType(item?.question_type),
                answer_type: this._normalizeAnswerType(item?.answer_type),
                reasoning_chain: this._normalizeReasoningChain(item?.reasoning_chain, item?.reasoning_chain_text),
            }));
        },

        agreementQuestionConflictSourceSummary(conflict) {
            const item = conflict?.source_item;
            if (!item) return 'No Opus question';
            const question = String(item.question || '').replace(/\s+/g, ' ').trim();
            const shortQ = question.length > 80 ? `${question.substring(0, 80)}...` : question;
            return `${shortQ} (type: ${item.question_type || 'extractive'})`;
        },

        agreementQuestionConflictCurrentSummary(conflict) {
            const current = this._currentQuestionForConflict(conflict);
            if (!current) return 'No question';
            const question = String(current.question || '').replace(/\s+/g, ' ').trim();
            const shortQ = question.length > 80 ? `${question.substring(0, 80)}...` : question;
            const answer = String(current.answer || '').trim() || '—';
            return `${shortQ} (type: ${current.question_type || 'extractive'}, answer: ${answer})`;
        },

        agreementRuleConflictSummary(conflict, side) {
            const ruleText = side === 'b' ? String(conflict?.b_rule_text || '') : String(conflict?.a_rule_text || '');
            return ruleText ? `Has rule: ${ruleText}` : 'No rule';
        },

        agreementRuleConflictSummaryHtml(conflict, side) {
            const ruleText = side === 'b' ? String(conflict?.b_rule_text || '') : String(conflict?.a_rule_text || '');
            if (!ruleText) {
                return '<span class="text-xs text-muted">No rule</span>';
            }
            return `<div class="text-xs text-muted"><strong>Has rule:</strong></div>${this.renderRuleDisplayHtml(ruleText)}`;
        },

        agreementRuleConflictSourceSummary(conflict) {
            return conflict?.source_rule_text ? `Opus has rule: ${conflict.source_rule_text}` : 'Opus has no rule';
        },

        agreementRuleConflictCurrentSummary(conflict) {
            const currentEntry = this._currentRuleEntryForConflict(conflict);
            return currentEntry
                ? `Current has rule: ${currentEntry.rule_text}`
                : 'Current has no rule';
        },

        agreementRuleConflictCurrentSummaryHtml(conflict) {
            const currentEntry = this._currentRuleEntryForConflict(conflict);
            if (!currentEntry) {
                return '<span class="text-xs text-muted">Current has no rule</span>';
            }
            return `<div class="text-xs text-muted"><strong>Current has rule:</strong></div>${this.renderRuleDisplayHtml(currentEntry.rule_text || '')}`;
        },

        _applyQuestionConflictChoiceIntoCurrentDoc(conflict, choice) {
            if (!conflict) return false;
            const preferred = choice === 'b' ? 'b' : 'a';
            const selected = preferred === 'a' ? conflict.a_item : conflict.b_item;
            const key = String(conflict.key || '');
            if (!Array.isArray(this.docData.questions)) this.docData.questions = [];

            const current = this.docData.questions.slice();
            const idx = current.findIndex((q, i) => this._questionConflictKeyFromQuestion(q, i) === key);
            const appliedReplacementKeys = [];

            const sideExemptions = this.agreementWorkspace?.structured?.qaCoverageExemptionsBySide || {};
            const selectedSideExemptions = preferred === 'b'
                ? (sideExemptions.b || {})
                : (sideExemptions.a || {});
            const sideCoverageQuestionsByPair = this.agreementWorkspace?.structured?.qaCoverageQuestionsBySide || {};
            const selectedSideCoverageQuestions = preferred === 'b'
                ? (sideCoverageQuestionsByPair.b || {})
                : (sideCoverageQuestionsByPair.a || {});

            const impactedPairKeys = new Set();
            [conflict.source_item, conflict.a_item, conflict.b_item].forEach((item) => {
                const pairKey = this._qaCoveragePairKeyFromQuestion(item);
                if (pairKey) impactedPairKeys.add(pairKey);
            });

            const upsertCoverageCandidate = (candidate, { pairKeyHint = '' } = {}) => {
                if (!candidate || typeof candidate !== 'object') return false;
                const normalizedCandidate = this._cloneQuestionForAgreement(candidate, '');
                const candidateKey = String(
                    candidate?.key
                    || this._questionConflictKeyFromQuestion(normalizedCandidate, current.length)
                ).trim();
                const normalizedKey = candidateKey || this._questionConflictKeyFromQuestion(
                    normalizedCandidate,
                    current.length
                );
                const fallbackId = normalizedKey && !normalizedKey.startsWith('__idx_')
                    ? normalizedKey
                    : '';
                const toApply = this._cloneQuestionForAgreement(candidate, fallbackId);

                const existingIdx = normalizedKey
                    ? current.findIndex((q, i) => this._questionConflictKeyFromQuestion(q, i) === normalizedKey)
                    : -1;
                if (existingIdx >= 0) {
                    const previousPairKey = this._qaCoveragePairKeyFromQuestion(current[existingIdx]);
                    current[existingIdx] = toApply;
                    const nextPairKey = this._qaCoveragePairKeyFromQuestion(toApply);
                    if (previousPairKey && previousPairKey !== nextPairKey) impactedPairKeys.add(previousPairKey);
                } else {
                    current.push(toApply);
                }

                if (normalizedKey) appliedReplacementKeys.push(normalizedKey);
                if (pairKeyHint) impactedPairKeys.add(pairKeyHint);
                return true;
            };

            const ensurePairCoverageFromSelectedSide = () => {
                let changed = false;
                let iterations = 0;
                while (iterations < 12) {
                    iterations += 1;
                    let passChanged = false;
                    const presentPairs = this._qaCoveragePresentPairSetFromQuestions(current);
                    for (const pairKey of impactedPairKeys) {
                        if (!pairKey || presentPairs.has(pairKey)) continue;
                        const justification = String(selectedSideExemptions[pairKey] || '').trim();
                        if (justification) continue;
                        const candidates = Array.isArray(selectedSideCoverageQuestions[pairKey])
                            ? selectedSideCoverageQuestions[pairKey]
                            : [];
                        if (!candidates.length) continue;
                        let applied = false;
                        for (const candidate of candidates) {
                            const normalizedCandidate = this._cloneQuestionForAgreement(candidate, '');
                            const candidateKey = String(
                                candidate?.key
                                || this._questionConflictKeyFromQuestion(normalizedCandidate, current.length)
                            ).trim();
                            const existingIdx = candidateKey
                                ? current.findIndex((q, i) => this._questionConflictKeyFromQuestion(q, i) === candidateKey)
                                : -1;
                            if (existingIdx >= 0) {
                                const existingSig = this._questionAgreementSignature(current[existingIdx]);
                                const fallbackId = candidateKey && !candidateKey.startsWith('__idx_') ? candidateKey : '';
                                const candidateSig = this._questionAgreementSignature(
                                    this._cloneQuestionForAgreement(candidate, fallbackId)
                                );
                                if (existingSig === candidateSig && presentPairs.has(pairKey)) {
                                    applied = true;
                                    break;
                                }
                            }
                            applied = upsertCoverageCandidate(candidate, { pairKeyHint: pairKey });
                            if (applied) break;
                        }
                        if (applied) {
                            passChanged = true;
                            changed = true;
                        }
                    }
                    if (!passChanged) break;
                }
                return changed;
            };

            if (!selected) {
                if (idx >= 0) current.splice(idx, 1);
                const replacements = this.agreementQuestionConflictCoverageCandidates(conflict, preferred);
                for (const replacement of replacements) {
                    upsertCoverageCandidate(replacement);
                }
                ensurePairCoverageFromSelectedSide();
            } else {
                const normalized = this._cloneQuestionForAgreement(selected, key.startsWith('__idx_') ? '' : key);
                if (idx >= 0) current[idx] = normalized;
                else current.push(normalized);
                ensurePairCoverageFromSelectedSide();
            }
            this.docData.questions = current;
            this.docData.num_questions = current.length;
            if (impactedPairKeys.size > 0) {
                const presentPairs = this._qaCoveragePresentPairSet();
                const nextExemptions = [];
                for (const row of this._normalizeQaCoverageExemptions(this.docData[qaCoverageExemptionsField])) {
                    const rowKey = this._qaCoverageKey(row.question_type, row.answer_type);
                    if (!impactedPairKeys.has(rowKey)) {
                        nextExemptions.push(row);
                    }
                }
                for (const pairKey of impactedPairKeys) {
                    const justification = String(selectedSideExemptions[pairKey] || '').trim();
                    if (!justification || presentPairs.has(pairKey)) continue;
                    const [question_type, answer_type] = pairKey.split('::');
                    if (!question_type || !answer_type) continue;
                    nextExemptions.push({
                        question_type,
                        answer_type,
                        justification,
                    });
                }
                this.docData[qaCoverageExemptionsField] = nextExemptions;
            }

            const id = String(conflict.id || '');
            this.agreementWorkspace.structured.decisions[id] = preferred;
            delete this.agreementWorkspace.structured.manualResolutions[id];
            for (const replacementKey of appliedReplacementKeys) {
                const replacementConflictId = `q:${replacementKey}`;
                const replacementConflict = this._structuredConflictById(replacementConflictId);
                if (!replacementConflict) continue;
                this.agreementWorkspace.structured.decisions[replacementConflictId] = preferred;
                delete this.agreementWorkspace.structured.manualResolutions[replacementConflictId];
            }
            this.markDirty();
            this._recomputeAgreementStructuredConflictStates();
            return true;
        },

        _applyRuleConflictChoiceIntoCurrentDoc(conflict, choice) {
            if (!conflict) return false;
            const preferred = choice === 'b' ? 'b' : 'a';
            const selectedRuleText = preferred === 'a'
                ? String(conflict?.a_rule_text || '').trim()
                : String(conflict?.b_rule_text || '').trim();
            const shouldExist = !!selectedRuleText;
            const targetSignature = String(conflict?.rule_sig || '').trim();
            if (!Array.isArray(this.docData.rules)) this.docData.rules = [];
            const current = [];
            for (const value of this.docData.rules) {
                const entry = this._normalizeRuleEntryForAgreement(value);
                if (targetSignature && entry && entry.signature === targetSignature) continue;
                current.push(value);
            }
            if (shouldExist) current.push(selectedRuleText);
            this.docData.rules = current;

            const id = String(conflict.id || '');
            this.agreementWorkspace.structured.decisions[id] = preferred;
            delete this.agreementWorkspace.structured.manualResolutions[id];
            this.markDirty();
            this._recomputeAgreementStructuredConflictStates();
            return true;
        },

        applyAgreementStructuredConflictChoice(conflictId, choice) {
            const conflict = this._structuredConflictById(conflictId);
            if (!conflict) {
                showToast('Conflict not found', 'warning');
                return false;
            }
            const ok = conflict.kind === 'question'
                ? this._applyQuestionConflictChoiceIntoCurrentDoc(conflict, choice)
                : this._applyRuleConflictChoiceIntoCurrentDoc(conflict, choice);
            if (ok) {
                const reviewer = choice === 'b' ? this.agreementReviewerLabel('b') : this.agreementReviewerLabel('a');
                showToast(`Applied ${reviewer} for ${conflict.kind} conflict`, 'success');
            }
            return ok;
        },

        useCurrentAgreementStructuredConflict(conflictId) {
            const conflict = this._structuredConflictById(conflictId);
            if (!conflict) {
                showToast('Conflict not found', 'warning');
                return false;
            }
            const id = String(conflict.id || '');
            let sig = '';
            if (conflict.kind === 'question') {
                const currentQuestion = this._currentQuestionForConflict(conflict);
                sig = this._questionAgreementSignature(currentQuestion);
            } else {
                sig = this._ruleAgreementSignatureForPresence(this._currentRulePresenceForConflict(conflict));
            }
            this.agreementWorkspace.structured.decisions[id] = 'manual';
            this.agreementWorkspace.structured.manualResolutions[id] = sig;
            this.markDirty();
            this._recomputeAgreementStructuredConflictStates();
            showToast('Marked conflict as resolved using current content', 'success');
            return true;
        },

        openAgreementQuestionConflictForEdit(conflictId) {
            const conflict = this._structuredConflictById(conflictId);
            if (!conflict || conflict.kind !== 'question') return;
            const key = String(conflict.key || '');
            if (!Array.isArray(this.docData.questions)) this.docData.questions = [];
            let idx = this.docData.questions.findIndex((q, i) => this._questionConflictKeyFromQuestion(q, i) === key);
            this.panelState.questions = true;
            if (idx >= 0) {
                this.editingQuestion = idx;
                showToast('Editing current question for this conflict', 'info');
            } else {
                const fallbackItem = conflict.source_item || conflict.a_item || conflict.b_item || {
                    question_id: key.startsWith('__idx_') ? '' : key,
                    question: '',
                    question_type: 'extractive',
                    answer: '',
                    answer_type: 'variant',
                    reasoning_chain: [],
                    reasoning_chain_text: '',
                };
                const draft = this._cloneQuestionForAgreement(fallbackItem, key.startsWith('__idx_') ? '' : key);
                this.docData.questions.push(draft);
                this.docData.num_questions = this.docData.questions.length;
                idx = this.docData.questions.length - 1;
                this.editingQuestion = idx;
                this.markDirty();
                this._recomputeAgreementStructuredConflictStates();
                showToast('Created editable current question for this conflict', 'info');
            }
        },

        _findQuestionConflictAnchorRect(conflictId) {
            const id = String(conflictId || '').trim();
            if (!id) return null;
            const candidates = document.querySelectorAll('[data-question-conflict-id]');
            for (const el of candidates) {
                const candidateId = String(el.getAttribute('data-question-conflict-id') || '').trim();
                if (candidateId === id) {
                    return el.getBoundingClientRect();
                }
            }
            return null;
        },

        _positionAgreementQuestionConflictPopover(anchorRect = null) {
            const modal = this.agreementWorkspace.questionConflictModal;
            const panel = this.$refs?.agreementQuestionConflictPopover;
            if (!panel) return;

            const pad = 12;
            const width = panel.offsetWidth || 900;
            const height = panel.offsetHeight || 420;
            const rect = anchorRect || this._findQuestionConflictAnchorRect(modal.conflictId);

            let x = (window.innerWidth - width) / 2;
            let y = 90;
            if (rect) {
                x = rect.left + (rect.width / 2) - (width / 2);
                y = rect.top - height - 10;
                if (y < pad) {
                    y = rect.bottom + 10;
                }
            }

            x = Math.max(pad, Math.min(x, window.innerWidth - width - pad));
            y = Math.max(pad, Math.min(y, window.innerHeight - height - pad));

            modal.x = Math.round(x);
            modal.y = Math.round(y);
        },

        openAgreementQuestionConflictModal(conflictId, anchorEl = null) {
            const conflict = this._structuredConflictById(conflictId);
            if (!conflict || conflict.kind !== 'question') return;

            this.panelState.questions = true;
            if (this.agreementWorkspace.conflictModal.show) {
                this.closeAgreementConflictModal();
            }
            const modal = this.agreementWorkspace.questionConflictModal;
            modal.show = true;
            modal.conflictId = String(conflict.id || '');

            const anchorRect = anchorEl && typeof anchorEl.getBoundingClientRect === 'function'
                ? anchorEl.getBoundingClientRect()
                : null;
            this.$nextTick(() => {
                this._positionAgreementQuestionConflictPopover(anchorRect);
                this.attachQuestionAnnotationHoverListeners();
            });
        },

        closeAgreementQuestionConflictModal() {
            const modal = this.agreementWorkspace.questionConflictModal;
            modal.show = false;
            modal.conflictId = '';
            modal.x = 24;
            modal.y = 96;
        },

        getActiveAgreementQuestionConflict() {
            const id = String(this.agreementWorkspace.questionConflictModal.conflictId || '').trim();
            if (!id) return null;
            const conflict = this._structuredConflictById(id);
            if (!conflict || conflict.kind !== 'question') return null;
            return conflict;
        },

        getCurrentQuestionForConflict(conflict) {
            if (!conflict || conflict.kind !== 'question') return null;
            return this._currentQuestionForConflict(conflict);
        },

        applyChoiceForActiveAgreementQuestionConflict(choice) {
            const conflict = this.getActiveAgreementQuestionConflict();
            if (!conflict) return;
            const ok = this.applyAgreementStructuredConflictChoice(conflict.id, choice);
            if (!ok) return;
            const updated = this._structuredConflictById(conflict.id);
            if (!updated || updated.resolution !== 'unresolved') {
                this.closeAgreementQuestionConflictModal();
            } else {
                this.$nextTick(() => this._positionAgreementQuestionConflictPopover(null));
            }
        },

        markActiveAgreementQuestionConflictResolved() {
            const conflict = this.getActiveAgreementQuestionConflict();
            if (!conflict) return;
            const ok = this.useCurrentAgreementStructuredConflict(conflict.id);
            if (ok) {
                this.closeAgreementQuestionConflictModal();
            }
        },

        editActiveAgreementQuestionConflict() {
            const conflict = this.getActiveAgreementQuestionConflict();
            if (!conflict) return;
            this.closeAgreementQuestionConflictModal();
            this.openAgreementQuestionConflictForEdit(conflict.id);
        },

        _normalizeAgreementConflictIds(input) {
            const rawItems = Array.isArray(input) ? input : String(input || '').split(',');
            const out = [];
            const seen = new Set();
            for (const item of rawItems) {
                const id = String(item || '').trim();
                if (!id || seen.has(id)) continue;
                seen.add(id);
                out.push(id);
            }
            return out;
        },

        _canBundleAgreementConflicts(left, right) {
            if (!left || !right) return false;
            if (left.kind === 'text_edit' || right.kind === 'text_edit') {
                if (left.kind !== right.kind) return false;
                return Number(right.start) <= Number(left.end);
            }
            if (Number(right.start) <= Number(left.end)) return true;

            const plain = String(this.agreementWorkspace.merge.plainText || '');
            const leftEnd = Number(left.end || 0);
            const rightStart = Number(right.start || 0);
            const gapLen = rightStart - leftEnd;
            if (gapLen < 0) return true;
            if (gapLen > 4) return false;

            const gap = plain.substring(leftEnd, rightStart);
            return /^[\s,.;:!?()'"-]*$/.test(gap);
        },

        _expandAgreementConflictGroupFromSeeds(seedIds) {
            const normalizedSeeds = this._normalizeAgreementConflictIds(seedIds);
            if (normalizedSeeds.length === 0) return [];

            const unresolved = (this.agreementWorkspace.merge.conflicts || [])
                .filter((item) => item.resolution === 'unresolved')
                .sort((a, b) => (a.start - b.start) || (a.end - b.end));
            const fallbackAll = (this.agreementWorkspace.merge.conflicts || [])
                .slice()
                .sort((a, b) => (a.start - b.start) || (a.end - b.end));
            const list = unresolved.length > 0 ? unresolved : fallbackAll;
            if (list.length === 0) return [];

            const seedSet = new Set(normalizedSeeds);
            const seedIndices = [];
            for (let idx = 0; idx < list.length; idx++) {
                if (seedSet.has(list[idx].id)) seedIndices.push(idx);
            }
            if (seedIndices.length === 0) {
                return normalizedSeeds.filter((id) => this.getAgreementConflictById(id));
            }

            let startIdx = Math.min(...seedIndices);
            let endIdx = Math.max(...seedIndices);

            while (startIdx > 0 && this._canBundleAgreementConflicts(list[startIdx - 1], list[startIdx])) {
                startIdx -= 1;
            }
            while (endIdx < list.length - 1 && this._canBundleAgreementConflicts(list[endIdx], list[endIdx + 1])) {
                endIdx += 1;
            }

            return list.slice(startIdx, endIdx + 1).map((item) => item.id);
        },

        _agreementConflictIdsForRange(start, end) {
            const ids = [];
            for (const conflict of this.agreementWorkspace.merge.conflicts || []) {
                if (start < conflict.end && end > conflict.start) {
                    ids.push(conflict.id);
                }
            }
            return ids;
        },

        _agreementSegmentTitle(segment) {
            if (!segment) return '';
            if (segment.kind === 'agreed') return 'Agreed';
            const conflict = this.getAgreementConflictById(segment.primaryConflictId);
            if (!conflict) return 'Conflict';
            if (conflict.kind === 'text_edit') {
                const current = String(conflict.current_text || '');
                const currentLabel = current ? current.replace(/\s+/g, ' ').trim() : '(empty)';
                return `${this.agreementReviewerLabel('a')}: ${this._formatAgreementRefs(conflict.a_refs || [], 4)} | ${this.agreementReviewerLabel('b')}: ${this._formatAgreementRefs(conflict.b_refs || [], 4)} | Current text: ${currentLabel}`;
            }
            const a = this._formatAgreementRefs(conflict.a_refs || [], 4);
            const b = this._formatAgreementRefs(conflict.b_refs || [], 4);
            const current = this._formatAgreementRefs(conflict.current_refs || [], 4);
            return `${this.agreementReviewerLabel('a')}: ${a} | ${this.agreementReviewerLabel('b')}: ${b} | Current: ${current}`;
        },

        _buildAgreementInlineSegments(plainText, aAnnotations, bAnnotations) {
            const plain = String(plainText || '');
            const aList = Array.isArray(aAnnotations) ? aAnnotations : [];
            const bList = Array.isArray(bAnnotations) ? bAnnotations : [];

            const boundaries = new Set([0, plain.length]);
            for (const ann of aList) {
                boundaries.add(Number(ann.start));
                boundaries.add(Number(ann.end));
            }
            for (const ann of bList) {
                boundaries.add(Number(ann.start));
                boundaries.add(Number(ann.end));
            }
            const points = Array.from(boundaries)
                .filter((value) => Number.isFinite(value))
                .sort((x, y) => x - y);

            const segments = [];
            const stats = { agreed: 0, total: 0, resolved: 0, remaining: 0 };

            for (let idx = 0; idx < points.length - 1; idx++) {
                const start = points[idx];
                const end = points[idx + 1];
                if (end <= start) continue;

                const aActive = aList.filter((ann) => ann.start <= start && ann.end >= end);
                const bActive = bList.filter((ann) => ann.start <= start && ann.end >= end);
                const aRefs = this._normalizeAgreementRefs(aActive.map((ann) => ann.ref));
                const bRefs = this._normalizeAgreementRefs(bActive.map((ann) => ann.ref));

                let kind = 'plain';
                if (aRefs.length === 0 && bRefs.length === 0) {
                    kind = 'plain';
                } else if (this._areAgreementRefListsEqual(aRefs, bRefs)) {
                    kind = 'agreed';
                    stats.agreed += 1;
                } else if (aRefs.length > 0 && bRefs.length === 0) {
                    kind = 'only_a';
                } else if (bRefs.length > 0 && aRefs.length === 0) {
                    kind = 'only_b';
                } else {
                    kind = 'conflict';
                }

                const conflictIds = this._agreementConflictIdsForRange(start, end);
                segments.push({
                    id: `${start}:${end}`,
                    start,
                    end,
                    text: plain.substring(start, end),
                    kind,
                    aRefs,
                    bRefs,
                    conflictIds,
                    primaryConflictId: conflictIds[0] || '',
                });
            }

            return { segments, stats };
        },

        _renderAgreementInlineCompareHtml() {
            const merge = this.agreementWorkspace.merge;
            const segments = Array.isArray(merge.inlineSegments) ? merge.inlineSegments : [];
            const selectedConflictId = String(merge.selectedConflictId || '');
            const conflictState = new Map((merge.conflicts || []).map((item) => [item.id, String(item.resolution || 'unresolved')]));

            let html = '';
            for (const segment of segments) {
                if (segment.kind === 'plain') {
                    html += this._escapeAgreementHtml(segment.text);
                    continue;
                }

                const classes = ['agreement-inline-seg'];
                const primaryConflictId = segment.primaryConflictId || '';
                const state = String(conflictState.get(primaryConflictId) || 'unresolved');

                if (segment.kind === 'agreed') {
                    classes.push('kind-agreed');
                } else if (state === 'unresolved') {
                    classes.push('kind-conflict-open');
                } else {
                    // Resolved conflicts should disappear from inline conflict view.
                    html += this._escapeAgreementHtml(segment.text);
                    continue;
                }

                if (selectedConflictId && (segment.conflictIds || []).includes(selectedConflictId)) {
                    classes.push('selected');
                }

                const attrs = [`class="${classes.join(' ')}"`];
                if (segment.primaryConflictId) {
                    attrs.push(`data-agreement-conflict-id="${this._escapeAgreementHtml(segment.primaryConflictId)}"`);
                }
                if ((segment.conflictIds || []).length > 0) {
                    attrs.push(`data-agreement-conflict-ids="${this._escapeAgreementHtml(segment.conflictIds.join(','))}"`);
                }
                if ((segment.conflictIds || []).length > 1) {
                    attrs.push(`data-agreement-conflict-count="${segment.conflictIds.length}"`);
                }
                attrs.push(`title="${this._escapeAgreementHtml(this._agreementSegmentTitle(segment))}"`);

                html += `<span ${attrs.join(' ')}>`;
                html += this._escapeAgreementHtml(segment.text);
                html += '</span>';
            }

            merge.inlineHtml = html;
        },

        getAgreementConflictById(conflictId) {
            const id = String(conflictId || '').trim();
            if (!id) return null;
            return (this.agreementWorkspace.merge.conflicts || []).find((item) => item.id === id)
                || this._structuredConflictById(id)
                || null;
        },

        getAgreementSelectedConflict() {
            const conflictId = String(this.agreementWorkspace.merge.selectedConflictId || '');
            if (!conflictId) return null;
            return this.getAgreementConflictById(conflictId);
        },

        agreementSelectedConflictSummary() {
            const conflict = this.getAgreementSelectedConflict();
            if (!conflict) return '';
            const baseText = conflict.kind === 'text_edit'
                ? String(conflict.base_text || conflict.text || '')
                : String(conflict.text || '');
            const text = baseText.replace(/\s+/g, ' ').trim() || '(empty span)';
            const shortText = text.length > 52 ? `${text.substring(0, 52)}...` : text;
            if (conflict.kind === 'text_edit') {
                const aText = String(conflict.a_plain_text || '').replace(/\s+/g, ' ').trim() || '(empty)';
                const bText = String(conflict.b_plain_text || '').replace(/\s+/g, ' ').trim() || '(empty)';
                const currentText = String(conflict.current_text || '').replace(/\s+/g, ' ').trim() || '(empty)';
                const aShort = aText.length > 36 ? `${aText.substring(0, 36)}...` : aText;
                const bShort = bText.length > 36 ? `${bText.substring(0, 36)}...` : bText;
                const cShort = currentText.length > 36 ? `${currentText.substring(0, 36)}...` : currentText;
                return `"${shortText}" • ${this.agreementReviewerLabel('a')}: ${aShort} • ${this.agreementReviewerLabel('b')}: ${bShort} • Current: ${cShort}`;
            }
            const aRefs = this._formatAgreementRefs(conflict.a_refs || [], 3);
            const bRefs = this._formatAgreementRefs(conflict.b_refs || [], 3);
            const current = this._formatAgreementRefs(conflict.current_refs || [], 3);
            return `"${shortText}" • ${this.agreementReviewerLabel('a')}: ${aRefs} • ${this.agreementReviewerLabel('b')}: ${bRefs} • Current: ${current}`;
        },

        selectAgreementConflict(conflictId, opts = {}) {
            const id = String(conflictId || '').trim();
            if (!id) {
                this.agreementWorkspace.merge.selectedConflictId = '';
                this._renderAgreementInlineCompareHtml();
                this.$nextTick(() => this.attachAnnotationHoverListeners());
                return;
            }
            const exists = !!this.getAgreementConflictById(id);
            if (!exists) {
                if (!opts.silent) {
                    showToast('Difference not found', 'warning');
                }
                return;
            }
            this.agreementWorkspace.merge.selectedConflictId = id;
            this._renderAgreementInlineCompareHtml();
            this.$nextTick(() => this.attachAnnotationHoverListeners());
        },

        selectAdjacentAgreementConflict(step = 1, unresolvedOnly = true) {
            if (!this.agreementWorkspace.active) return;
            const conflicts = this.agreementWorkspace.merge.conflicts || [];
            if (conflicts.length === 0) return;

            const unresolved = conflicts.filter((item) => item.resolution === 'unresolved');
            if (unresolvedOnly && unresolved.length === 0) {
                this.selectAgreementConflict('', { silent: true });
                return;
            }
            const list = unresolvedOnly ? unresolved : conflicts;
            if (list.length === 0) return;

            const currentId = String(this.agreementWorkspace.merge.selectedConflictId || '');
            let idx = list.findIndex((item) => item.id === currentId);
            if (idx < 0) {
                idx = step >= 0 ? -1 : 0;
            }
            const next = (idx + step + list.length) % list.length;
            this.selectAgreementConflict(list[next].id, { silent: true });
        },

        useCurrentForSelectedConflict() {
            const conflict = this.getAgreementSelectedConflict();
            if (!conflict) return;
            if (conflict.kind === 'text_edit') {
                const parsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
                const range = this._agreementCurrentRangeForConflict(conflict);
                const start = Math.max(0, Math.min(Number(range.start || 0), parsed.plainText.length));
                const end = Math.max(start, Math.min(Number(range.end || start), parsed.plainText.length));
                const currentText = String(parsed.plainText || '').substring(start, end);
                this.agreementWorkspace.merge.decisions[conflict.id] = 'manual';
                this.agreementWorkspace.merge.manualTextResolutions[conflict.id] = currentText;
                delete this.agreementWorkspace.merge.manualResolutions[conflict.id];
                this._recomputeAgreementConflictStates();
                this.selectAdjacentAgreementConflict(1, true);
                showToast('Kept current text for this conflict', 'success');
                return;
            }
            const refs = this._normalizeAgreementRefs(conflict.current_refs || []);
            this.agreementWorkspace.merge.decisions[conflict.id] = 'manual';
            this.agreementWorkspace.merge.manualResolutions[conflict.id] = this._agreementRefsSignature(refs);
            delete this.agreementWorkspace.merge.manualTextResolutions[conflict.id];
            this._recomputeAgreementConflictStates();
            this.selectAdjacentAgreementConflict(1, true);
            showToast('Kept current annotation for this conflict', 'success');
        },

        openSelectedConflictInEditor() {
            const conflict = this.getAgreementSelectedConflict();
            if (!conflict) return;
            this.agreementWorkspace.resolveMode = false;
            this.sourceView = false;
            showToast('Current editor view enabled. Edit this span, then return to Resolve mode.', 'info');
        },

        _agreementConflictListFromInput(input) {
            if (Array.isArray(input)) {
                return input.filter((item) => item && typeof item === 'object' && item.id);
            }
            if (input && typeof input === 'object' && input.id) {
                return [input];
            }
            const id = String(input || '').trim();
            if (id) {
                const item = this.getAgreementConflictById(id);
                if (item) return [item];
            }
            const modalItems = this.agreementModalConflicts();
            if (modalItems.length > 0) return modalItems;
            const selected = this.getAgreementSelectedConflict();
            return selected ? [selected] : [];
        },

        _agreementConflictBounds(input) {
            const items = this._agreementConflictListFromInput(input);
            if (items.length === 0) return null;
            let start = this._conflictBaseStart(items[0]);
            let end = this._conflictBaseEnd(items[0]);
            for (const item of items.slice(1)) {
                start = Math.min(start, this._conflictBaseStart(item));
                end = Math.max(end, this._conflictBaseEnd(item));
            }
            return { start, end, conflicts: items };
        },

        agreementConflictContext(conflict, contextChars = 44) {
            const bounds = this._agreementConflictBounds(conflict);
            if (!bounds) return { before: '', span: '', after: '', start: 0, end: 0, left: 0, right: 0 };
            const plain = String(this.agreementWorkspace.merge.plainText || '');
            const start = Number(bounds.start || 0);
            const end = Number(bounds.end || start);
            const left = Math.max(0, start - contextChars);
            const right = Math.min(plain.length, end + contextChars);
            const before = plain.substring(left, start);
            const span = plain.substring(start, end);
            const after = plain.substring(end, right);
            return {
                before: `${left > 0 ? '…' : ''}${before}`,
                span,
                after: `${after}${right < plain.length ? '…' : ''}`,
                start,
                end,
                left,
                right,
            };
        },

        agreementConflictRefsForSide(conflict, side) {
            const items = this._agreementConflictListFromInput(conflict);
            if (items.length === 0) return [];
            const refs = [];
            for (const item of items) {
                let sideRefs = [];
                if (side === 'b') {
                    sideRefs = item.b_refs || [];
                } else if (side === 'source') {
                    sideRefs = item.source_refs || [];
                } else {
                    sideRefs = item.a_refs || [];
                }
                refs.push(...sideRefs);
            }
            return this._normalizeAgreementRefs(refs);
        },

        agreementConflictHasSideAnnotation(conflict, side) {
            return this.agreementConflictRefsForSide(conflict, side).length > 0;
        },

        _agreementConflictAnnotationsForSide(conflict, side) {
            const items = this._agreementConflictListFromInput(conflict);
            const out = [];
            const seen = new Set();
            for (const item of items) {
                let list = [];
                if (side === 'b') {
                    list = item.b_annotations || [];
                } else if (side === 'source') {
                    list = item.source_annotations || [];
                } else {
                    list = item.a_annotations || [];
                }
                for (const ann of list) {
                    const key = `${ann.start}:${ann.end}:${ann.ref}:${ann.text}`;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    out.push(ann);
                }
            }
            out.sort((a, b) => (a.start - b.start) || (a.end - b.end));
            return out;
        },

        _agreementRenderTextDeltaHtml(baseText, targetText) {
            const base = String(baseText || '');
            const target = String(targetText || '');
            if (base === target) return this._escapeAgreementHtml(target);

            const edits = this._computeAgreementTextEdits(base, target);
            if (!Array.isArray(edits) || edits.length === 0) {
                return this._escapeAgreementHtml(target);
            }

            let html = '';
            let cursor = 0;
            const sorted = edits
                .slice()
                .sort((x, y) => (Number(x.baseStart || 0) - Number(y.baseStart || 0)) || (Number(x.baseEnd || 0) - Number(y.baseEnd || 0)));
            for (const edit of sorted) {
                const baseStart = Math.max(0, Number(edit.baseStart || 0));
                const baseEnd = Math.max(baseStart, Number(edit.baseEnd || baseStart));
                const targetStart = Math.max(0, Number(edit.targetStart || 0));
                const targetEnd = Math.max(targetStart, Number(edit.targetEnd || targetStart));
                html += this._escapeAgreementHtml(base.substring(cursor, baseStart));
                const removed = base.substring(baseStart, baseEnd);
                const inserted = target.substring(targetStart, targetEnd);
                if (removed) {
                    html += `<span class="agreement-text-del">${this._escapeAgreementHtml(removed)}</span>`;
                }
                if (inserted) {
                    html += `<span class="agreement-text-ins">${this._escapeAgreementHtml(inserted)}</span>`;
                }
                cursor = baseEnd;
            }
            html += this._escapeAgreementHtml(base.substring(cursor));
            return html;
        },

        _agreementTextConflictSideHtml(conflict, side) {
            const base = String(conflict?.base_text || '');
            const sideText = String(side === 'b' ? (conflict?.b_plain_text || '') : (conflict?.a_plain_text || ''));
            const beforeKey = side === 'b' ? 'b_context_before' : 'a_context_before';
            const afterKey = side === 'b' ? 'b_context_after' : 'a_context_after';
            const leftEllipsisKey = side === 'b' ? 'b_context_left_ellipsis' : 'a_context_left_ellipsis';
            const rightEllipsisKey = side === 'b' ? 'b_context_right_ellipsis' : 'a_context_right_ellipsis';
            const before = String(conflict?.[beforeKey] || '');
            const after = String(conflict?.[afterKey] || '');
            const leftEllipsis = !!conflict?.[leftEllipsisKey];
            const rightEllipsis = !!conflict?.[rightEllipsisKey];
            const body = this._agreementRenderTextDeltaHtml(base, sideText) || '<span class="agreement-context-span muted">(empty)</span>';
            let html = '';
            if (leftEllipsis) html += '…';
            html += this._escapeAgreementHtml(before);
            html += body;
            html += this._escapeAgreementHtml(after);
            if (rightEllipsis) html += '…';
            return html;
        },

        agreementConflictSideContextHtml(conflict, side) {
            const items = this._agreementConflictListFromInput(conflict);
            if (items.length > 0 && items.every((item) => item?.kind === 'text_edit')) {
                return items
                    .map((item) => `<div class="agreement-text-context-block">${this._agreementTextConflictSideHtml(item, side)}</div>`)
                    .join('');
            }
            const ctx = this.agreementConflictContext(conflict);
            if (!ctx.span && !ctx.before && !ctx.after) return '';
            const plain = String(this.agreementWorkspace.merge.plainText || '');
            const anns = this._agreementConflictAnnotationsForSide(conflict, side)
                .filter((ann) => Number(ann.end) > ctx.left && Number(ann.start) < ctx.right);
            if (anns.length === 0) {
                return `${this._escapeAgreementHtml(ctx.before)}<span class="agreement-context-span muted">${this._escapeAgreementHtml(ctx.span)}</span>${this._escapeAgreementHtml(ctx.after)}`;
            }

            let html = '';
            let cursor = ctx.left;
            for (const ann of anns) {
                const start = Math.max(ctx.left, Number(ann.start || 0));
                const end = Math.min(ctx.right, Number(ann.end || 0));
                if (end <= start || start < cursor) continue;
                html += this._escapeAgreementHtml(plain.substring(cursor, start));
                const typeToken = this._sanitizeAgreementClassToken(ann?.entityType || '');
                const cls = typeToken ? `ann ann-${typeToken}` : 'ann';
                const refLabel = String(ann?.ref || '').trim();
                const entId = String(ann?.entityId || '').trim();
                const attr = String(ann?.attribute || '').trim();
                const type = String(ann?.entityType || '').trim();
                const titleParts = [];
                if (entId) titleParts.push(`Entity: ${entId}`);
                if (type) titleParts.push(`Type: ${type}`);
                if (attr) titleParts.push(`Attribute: ${attr}`);
                const hoverTitle = this._escapeAgreementHtml(titleParts.join(' | ') || refLabel);
                html += `<span class="${cls}" data-ref="${this._escapeAgreementHtml(refLabel)}" title="${hoverTitle}">${this._escapeAgreementHtml(plain.substring(start, end))}</span>`;
                cursor = end;
            }
            html += this._escapeAgreementHtml(plain.substring(cursor, ctx.right));
            if (ctx.left > 0) html = `…${html}`;
            if (ctx.right < plain.length) html = `${html}…`;
            return html;
        },

        agreementModalConflicts() {
            const modal = this.agreementWorkspace.conflictModal;
            const ids = this._normalizeAgreementConflictIds(
                Array.isArray(modal?.conflictIds) && modal.conflictIds.length > 0
                    ? modal.conflictIds
                    : (modal?.conflictId || '')
            );
            const out = [];
            for (const id of ids) {
                const item = this.getAgreementConflictById(id);
                if (item) out.push(item);
            }
            return out;
        },

        agreementModalConflict() {
            return this.agreementModalConflicts()[0] || null;
        },

        agreementModalConflictCount() {
            return this.agreementModalConflicts().length;
        },

        agreementModalConflictHeadline() {
            const items = this.agreementModalConflicts();
            if (items.length === 0) return '';
            const ctx = this.agreementConflictContext(items, 0);
            const text = String(ctx.span || '').replace(/\s+/g, ' ').trim() || '(empty span)';
            const shortText = text.length > 88 ? `${text.substring(0, 88)}...` : text;
            if (items.length === 1) return shortText;
            return `${items.length} linked differences: "${shortText}"`;
        },

        agreementModalApplyChoiceLabel(side) {
            const who = this.agreementReviewerLabel(side);
            const count = this.agreementModalConflictCount();
            if (count > 1) return `Use ${who} (${count})`;
            return `Use ${who}`;
        },

        _findConflictAnchorRect(conflictId) {
            const id = String(conflictId || '');
            if (!id) return null;
            const candidates = document.querySelectorAll('[data-agreement-conflict-id]');
            for (const el of candidates) {
                const primaryId = String(el.getAttribute('data-agreement-conflict-id') || '');
                const allIds = this._normalizeAgreementConflictIds(el.getAttribute('data-agreement-conflict-ids') || '');
                if (primaryId === id || allIds.includes(id)) {
                    return el.getBoundingClientRect();
                }
            }
            return null;
        },

        _positionAgreementConflictPopover(anchorRect = null) {
            const modal = this.agreementWorkspace.conflictModal;
            const panel = this.$refs?.agreementConflictPopover;
            if (!panel) return;

            const pad = 12;
            const width = panel.offsetWidth || 760;
            const height = panel.offsetHeight || 340;
            const anchorId = this._normalizeAgreementConflictIds(
                Array.isArray(modal.conflictIds) && modal.conflictIds.length > 0
                    ? modal.conflictIds
                    : (modal.conflictId || '')
            )[0] || '';
            const rect = anchorRect || this._findConflictAnchorRect(anchorId);

            let x = (window.innerWidth - width) / 2;
            let y = 84;
            if (rect) {
                x = rect.left + (rect.width / 2) - (width / 2);
                y = rect.top - height - 10;
                if (y < pad) {
                    y = rect.bottom + 10;
                }
            }

            x = Math.max(pad, Math.min(x, window.innerWidth - width - pad));
            y = Math.max(pad, Math.min(y, window.innerHeight - height - pad));

            modal.x = Math.round(x);
            modal.y = Math.round(y);
        },

        openAgreementConflictModal(conflictIds, anchorEl = null) {
            const seedIds = this._normalizeAgreementConflictIds(conflictIds);
            const fallbackId = this.getAgreementSelectedConflict()?.id || '';
            const expanded = this._expandAgreementConflictGroupFromSeeds(seedIds.length > 0 ? seedIds : fallbackId);
            const ids = expanded.length > 0 ? expanded : seedIds;
            const first = this.getAgreementConflictById(ids[0]) || this.getAgreementSelectedConflict();
            if (!first) return;

            if (this.agreementWorkspace.questionConflictModal.show) {
                this.closeAgreementQuestionConflictModal();
            }

            this.selectAgreementConflict(first.id, { silent: true });

            const modal = this.agreementWorkspace.conflictModal;
            modal.show = true;
            modal.conflictIds = ids.length > 0 ? ids : [first.id];
            modal.conflictId = modal.conflictIds[0] || first.id;
            modal.choice = first.resolution === 'a'
                ? 'a'
                : (first.resolution === 'b' ? 'b' : '');
            modal.manualRefs = '';

            const anchorRect = anchorEl && typeof anchorEl.getBoundingClientRect === 'function'
                ? anchorEl.getBoundingClientRect()
                : null;
            this.$nextTick(() => {
                this._positionAgreementConflictPopover(anchorRect);
            });
        },

        closeAgreementConflictModal() {
            const modal = this.agreementWorkspace.conflictModal;
            modal.show = false;
            modal.conflictIds = [];
            modal.conflictId = '';
            modal.choice = '';
            modal.manualRefs = '';
            modal.x = 24;
            modal.y = 96;
        },

        setAgreementConflictModalChoice(choice) {
            const modal = this.agreementWorkspace.conflictModal;
            modal.choice = choice;
            if (choice === 'a' || choice === 'b') {
                modal.manualRefs = '';
            }
        },

        agreementConflictModalCanMerge() {
            const modal = this.agreementWorkspace.conflictModal;
            if (!modal.show) return false;
            return this.agreementModalConflicts().length > 0;
        },

        _parseAgreementManualRefs(rawValue) {
            const values = String(rawValue || '')
                .split(',')
                .map((item) => item.trim())
                .filter(Boolean);
            const parsed = [];
            for (const item of values) {
                if (!/^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?$/.test(item)) {
                    return { ok: false, refs: [], error: `Invalid reference: "${item}"` };
                }
                parsed.push(item);
            }
            return { ok: true, refs: this._normalizeAgreementRefs(parsed), error: '' };
        },

        _applyManualConflictRefsIntoCurrentDoc(conflict, refs) {
            if (!conflict) return false;
            const currentParsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
            if (conflict?.kind === 'text_edit') {
                const range = this._agreementCurrentRangeForConflict(conflict);
                const start = Math.max(0, Math.min(Number(range.start || 0), currentParsed.plainText.length));
                const end = Math.max(start, Math.min(Number(range.end || start), currentParsed.plainText.length));
                const currentText = String(currentParsed.plainText || '').substring(start, end);
                this.agreementWorkspace.merge.decisions[conflict.id] = 'manual';
                this.agreementWorkspace.merge.manualTextResolutions[conflict.id] = currentText;
                this.agreementWorkspace.loadedVariant = 'editor';
                this._setConflictSelectedState(conflict.id, 'custom');
                this._recomputeAgreementConflictStates();
                return true;
            }

            const range = this._agreementCurrentRangeForConflict(conflict);
            const currentStart = Math.max(0, Number(range.start || 0));
            const currentEnd = Math.max(currentStart, Number(range.end || currentStart));
            const kept = (currentParsed.annotations || []).filter(
                (ann) => Number(ann.end) <= currentStart || Number(ann.start) >= currentEnd
            );

            const next = kept.concat((refs || []).map((ref) => ({
                start: currentStart,
                end: currentEnd,
                text: String(currentParsed.plainText || '').substring(currentStart, currentEnd),
                ref,
            })));

            const rebuilt = this._rebuildAnnotatedTextFromPlain(String(currentParsed.plainText || ''), next);
            this.docData.document_to_annotate = rebuilt;
            this.agreementWorkspace.merge.decisions[conflict.id] = 'manual';
            this.agreementWorkspace.merge.manualResolutions[conflict.id] = this._agreementRefsSignature(refs || []);
            delete this.agreementWorkspace.merge.manualTextResolutions[conflict.id];
            this.refreshEntities();
            this.markDirty();
            this.agreementWorkspace.loadedVariant = 'editor';
            return true;
        },

        applyAgreementConflictModalMerge() {
            const conflicts = this.agreementModalConflicts();
            if (conflicts.length === 0) {
                showToast('Conflict not found', 'warning');
                return;
            }

            const modal = this.agreementWorkspace.conflictModal;
            const choice = String(modal.choice || '').trim();
            let mergedCount = 0;

            if (choice === 'a' || choice === 'b') {
                const applied = this._applyConflictChoiceGroupIntoCurrentDoc(conflicts, choice);
                if (applied) {
                    mergedCount = conflicts.length;
                }
            } else {
                this._recomputeAgreementConflictStates();
                for (const conflict of conflicts) {
                    const conflictId = String(conflict.id || '');
                    if (!conflictId) continue;
                    const currentConflict = this.getAgreementConflictById(conflictId);
                    if (!currentConflict) continue;
                    if (currentConflict.kind === 'text_edit') {
                        const range = this._agreementCurrentRangeForConflict(currentConflict);
                        const parsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
                        const start = Math.max(0, Math.min(Number(range.start || 0), parsed.plainText.length));
                        const end = Math.max(start, Math.min(Number(range.end || start), parsed.plainText.length));
                        const currentText = String(parsed.plainText || '').substring(start, end);
                        this.agreementWorkspace.merge.decisions[conflictId] = 'manual';
                        this.agreementWorkspace.merge.manualTextResolutions[conflictId] = currentText;
                        delete this.agreementWorkspace.merge.manualResolutions[conflictId];
                    } else {
                        const refs = this._normalizeAgreementRefs(currentConflict.current_refs || []);
                        this.agreementWorkspace.merge.decisions[conflictId] = 'manual';
                        this.agreementWorkspace.merge.manualResolutions[conflictId] = this._agreementRefsSignature(refs);
                        delete this.agreementWorkspace.merge.manualTextResolutions[conflictId];
                    }
                    mergedCount += 1;
                }
                this._recomputeAgreementConflictStates();
            }

            if (mergedCount === 0) return;
            this.closeAgreementConflictModal();
            this.selectAdjacentAgreementConflict(1, true);
            if (mergedCount === 1) {
                showToast('Conflict merged into current document', 'success');
            } else {
                showToast(`${mergedCount} linked conflicts merged into current document`, 'success');
            }
        },

        handleAgreementInlineClick(event) {
            if (!this.agreementWorkspace.active) return;
            if (!this.agreementWorkspace.resolveMode) return;
            if (this.showEntityReferences) return;
            const target = event.target;
            if (!target) return;
            if (target.classList?.contains('resize-handle') || target.classList?.contains('ann-delete-btn')) {
                return;
            }

            const segment = target.closest('[data-agreement-conflict-id]');
            if (!segment) return;
            if (event.detail && event.detail > 1) {
                if (this.agreementClickTimer) {
                    clearTimeout(this.agreementClickTimer);
                    this.agreementClickTimer = null;
                }
                return;
            }

            const primaryId = String(segment.getAttribute('data-agreement-conflict-id') || '').trim();
            const inlineIds = this._normalizeAgreementConflictIds(
                segment.getAttribute('data-agreement-conflict-ids') || primaryId
            );
            if (inlineIds.length === 0) return;

            const groupedIds = this._expandAgreementConflictGroupFromSeeds(inlineIds);
            const openIds = groupedIds.length > 0 ? groupedIds : inlineIds;
            if (this.agreementClickTimer) {
                clearTimeout(this.agreementClickTimer);
                this.agreementClickTimer = null;
            }
            this.agreementClickTimer = setTimeout(() => {
                this.selectAgreementConflict(openIds[0], { silent: true });
                const anchor = segment && segment.isConnected ? segment : null;
                this.openAgreementConflictModal(openIds, anchor);
                this.agreementClickTimer = null;
            }, 320);
        },

        refreshAgreementInlineCompare() {
            if (!this.agreementWorkspace.active) return;
            this._recomputeAgreementConflictStates();
            this._recomputeAgreementStructuredConflictStates();
        },

        _buildAgreementMergeFromAnnotators() {
            this._ensureAgreementWorkspaceState();
            const merge = this.agreementWorkspace.merge;
            merge.plainText = '';
            merge.html = '';
            merge.inlineHtml = '';
            merge.inlineSegments = [];
            merge.decisions = {};
            merge.manualResolutions = {};
            merge.manualTextResolutions = {};
            merge.baseToCurrentMapper = null;
            merge.inlineStats = { agreed: 0, total: 0, resolved: 0, remaining: 0 };
            merge.selectedConflictId = '';
            merge.warning = '';
            merge.agreedCount = 0;
            merge.conflictCount = 0;
            merge.conflicts = [];

            const sideAKey = this.agreementCompareVariantKey('a');
            const sideBKey = this.agreementCompareVariantKey('b');
            const sideAVariant = this.agreementWorkspace.versions?.[sideAKey] || this.agreementWorkspace.versions.reviewer_a;
            const sideBVariant = this.agreementWorkspace.versions?.[sideBKey] || this.agreementWorkspace.versions.reviewer_b;
            const aDoc = sideAVariant?.editable_document || null;
            const bDoc = sideBVariant?.editable_document || null;
            const sourceDoc = this.agreementWorkspace.versions.source.editable_document;
            this._buildAgreementStructuredConflictsFromAnnotators(aDoc, bDoc, sourceDoc);

            if (this.reviewTarget === 'rules' || this.reviewTarget === 'questions') {
                merge.plainText = String(
                    this.docData?.document_to_annotate
                    || sourceDoc?.document_to_annotate
                    || aDoc?.document_to_annotate
                    || bDoc?.document_to_annotate
                    || ''
                );
                merge.html = this._escapeAgreementHtml(merge.plainText);
                merge.inlineHtml = merge.html;
                merge.inlineSegments = [];
                merge.decisions = {};
                merge.manualResolutions = {};
                merge.manualTextResolutions = {};
                merge.baseToCurrentMapper = null;
                merge.inlineStats = { agreed: 0, total: 0, resolved: 0, remaining: 0 };
                merge.selectedConflictId = '';
                merge.warning = '';
                merge.agreedCount = 0;
                merge.conflictCount = 0;
                merge.conflicts = [];
                this._recomputeAgreementConflictStates();
                this._recomputeAgreementStructuredConflictStates();
                return;
            }

            const aText = String(aDoc?.document_to_annotate || '');
            const bText = String(bDoc?.document_to_annotate || '');
            if (!aText || !bText) {
                const leftLabel = String(sideAVariant?.username || sideAVariant?.label || 'Reviewer A');
                const rightLabel = String(sideBVariant?.username || sideBVariant?.label || 'Reviewer B');
                merge.warning = `${leftLabel} and ${rightLabel} annotations are required to build merged agreement view.`;
                return;
            }

            const parsedA = this._parseAgreementAnnotatedText(aText);
            const parsedB = this._parseAgreementAnnotatedText(bText);
            const sourceText = String(sourceDoc?.document_to_annotate || '');
            const sourceCandidate = sourceText ? this._parseAgreementAnnotatedText(sourceText) : null;
            const sourcePlain = String(sourceCandidate?.plainText || '');
            const reviewerPlainEqual = parsedA.plainText === parsedB.plainText;
            if (this.isAgreementContestMode()) {
                merge.plainText = parsedB.plainText;
            } else if (reviewerPlainEqual && parsedA.plainText) {
                // When annotators agree on text, compare directly on their text (not Opus/source).
                merge.plainText = parsedA.plainText;
            } else {
                merge.plainText = sourcePlain || parsedA.plainText || parsedB.plainText;
            }

            let parsedSource = { plainText: merge.plainText, annotations: [] };
            if (sourceText) {
                if (sourceCandidate && sourceCandidate.plainText === merge.plainText) {
                    parsedSource = sourceCandidate;
                }
            }

            const renderRows = [];
            const editsA = this._computeAgreementTextEdits(merge.plainText, parsedA.plainText);
            const editsB = this._computeAgreementTextEdits(merge.plainText, parsedB.plainText);
            const mapperA = this._buildAgreementBaseToTargetMapper(editsA);
            const mapperB = this._buildAgreementBaseToTargetMapper(editsB);
            const mapperAToBase = this._buildAgreementTargetToBaseMapper(editsA);
            const mapperBToBase = this._buildAgreementTargetToBaseMapper(editsB);

            const annotationsAForCompare = parsedA.plainText === merge.plainText
                ? (parsedA.annotations || [])
                : this._projectAgreementAnnotationsToBasePlain(parsedA, merge.plainText, mapperAToBase, {
                    changedTargetRanges: editsA,
                    baseToTargetMapper: mapperA,
                });
            const annotationsBForCompare = parsedB.plainText === merge.plainText
                ? (parsedB.annotations || [])
                : this._projectAgreementAnnotationsToBasePlain(parsedB, merge.plainText, mapperBToBase, {
                    changedTargetRanges: editsB,
                    baseToTargetMapper: mapperB,
                });
            const canCompareAnnotationSpans =
                annotationsAForCompare.length > 0 || annotationsBForCompare.length > 0;

            if (canCompareAnnotationSpans) {
                const spanMap = new Map();
                const pushSpan = (side, ann) => {
                    const spanKey = this._spanKey(ann.start, ann.end);
                    if (!spanMap.has(spanKey)) {
                        spanMap.set(spanKey, {
                            id: spanKey,
                            start: ann.start,
                            end: ann.end,
                            text: merge.plainText.substring(ann.start, ann.end),
                            a: [],
                            b: [],
                        });
                    }
                    spanMap.get(spanKey)[side].push(ann);
                };
                for (const ann of annotationsAForCompare) pushSpan('a', ann);
                for (const ann of annotationsBForCompare) pushSpan('b', ann);

                const spanRows = Array.from(spanMap.values()).sort((x, y) => (x.start - y.start) || (x.end - y.end));
                for (const row of spanRows) {
                    const aKeySet = new Set(row.a.map((ann) => this._annotationKey(ann)));
                    const bKeySet = new Set(row.b.map((ann) => this._annotationKey(ann)));
                    const sameSize = aKeySet.size === bKeySet.size;
                    let identical = sameSize;
                    if (identical) {
                        for (const key of aKeySet) {
                            if (!bKeySet.has(key)) {
                                identical = false;
                                break;
                            }
                        }
                    }

                    if (identical) {
                        merge.agreedCount += row.a.length;
                        const first = row.a[0];
                        renderRows.push({
                            kind: 'agreed',
                            start: row.start,
                            end: row.end,
                            text: row.text,
                            entityType: first?.entityType || '',
                            ref: first?.ref || '',
                        });
                        continue;
                    }

                    const aRefs = Array.from(new Set(row.a.map((ann) => String(ann.ref || '').trim()).filter(Boolean)));
                    const bRefs = Array.from(new Set(row.b.map((ann) => String(ann.ref || '').trim()).filter(Boolean)));
                    const sourceAnnotations = (parsedSource.annotations || []).filter(
                        (ann) => Number(ann.start) < Number(row.end) && Number(ann.end) > Number(row.start)
                    );
                    const sourceRefs = Array.from(new Set(sourceAnnotations.map((ann) => String(ann.ref || '').trim()).filter(Boolean)));
                    merge.conflicts.push({
                        id: row.id,
                        kind: 'annotation',
                        start: row.start,
                        end: row.end,
                        base_start: row.start,
                        base_end: row.end,
                        text: row.text,
                        a_refs: aRefs,
                        b_refs: bRefs,
                        source_refs: sourceRefs,
                        a_annotations: row.a,
                        b_annotations: row.b,
                        source_annotations: sourceAnnotations,
                        selected: '',
                    });
                    renderRows.push({
                        kind: 'conflict',
                        start: row.start,
                        end: row.end,
                        text: row.text,
                        aRefs,
                        bRefs,
                    });
                }
            } else if (parsedA.plainText !== parsedB.plainText) {
                merge.warning = 'Reviewer text differs from source. Showing text-edit conflicts; annotation spans could not be projected reliably.';
            }

            const textGroups = this._groupAgreementTextEdits(merge.plainText, editsA, editsB);
            const contextChars = 28;

            const contextWindow = (plain, start, end) => {
                const safePlain = String(plain || '');
                const s = Math.max(0, Math.min(Number(start || 0), safePlain.length));
                const e = Math.max(s, Math.min(Number(end || s), safePlain.length));
                const left = Math.max(0, s - contextChars);
                const right = Math.min(safePlain.length, e + contextChars);
                return {
                    before: safePlain.substring(left, s),
                    after: safePlain.substring(e, right),
                    leftEllipsis: left > 0,
                    rightEllipsis: right < safePlain.length,
                };
            };

            textGroups.forEach((group, idx) => {
                const baseStart = Math.max(0, Number(group.baseStart || 0));
                const baseEnd = Math.max(baseStart, Number(group.baseEnd || baseStart));

                const aTargetStart = group.aOps.length > 0
                    ? Math.min(...group.aOps.map((op) => Number(op.targetStart || 0)))
                    : mapperA(baseStart);
                const aTargetEnd = group.aOps.length > 0
                    ? Math.max(...group.aOps.map((op) => Number(op.targetEnd || 0)))
                    : mapperA(baseEnd);
                const bTargetStart = group.bOps.length > 0
                    ? Math.min(...group.bOps.map((op) => Number(op.targetStart || 0)))
                    : mapperB(baseStart);
                const bTargetEnd = group.bOps.length > 0
                    ? Math.max(...group.bOps.map((op) => Number(op.targetEnd || 0)))
                    : mapperB(baseEnd);

                const aStart = Math.max(0, Math.min(aTargetStart, parsedA.plainText.length));
                const aEnd = Math.max(aStart, Math.min(aTargetEnd, parsedA.plainText.length));
                const bStart = Math.max(0, Math.min(bTargetStart, parsedB.plainText.length));
                const bEnd = Math.max(bStart, Math.min(bTargetEnd, parsedB.plainText.length));

                const baseSlice = merge.plainText.substring(baseStart, baseEnd);
                const aSlice = parsedA.plainText.substring(aStart, aEnd);
                const bSlice = parsedB.plainText.substring(bStart, bEnd);
                const aLocalAnnotations = this._collectAgreementLocalAnnotations(parsedA, aStart, aEnd);
                const bLocalAnnotations = this._collectAgreementLocalAnnotations(parsedB, bStart, bEnd);
                const sourceAnnotations = (parsedSource.annotations || []).filter(
                    (ann) => Number(ann.end) > baseStart && Number(ann.start) < baseEnd
                );
                const aRefs = this._normalizeAgreementRefs(aLocalAnnotations.map((ann) => String(ann.ref || '').trim()).filter(Boolean));
                const bRefs = this._normalizeAgreementRefs(bLocalAnnotations.map((ann) => String(ann.ref || '').trim()).filter(Boolean));
                const sourceRefs = this._normalizeAgreementRefs(sourceAnnotations.map((ann) => String(ann.ref || '').trim()).filter(Boolean));

                if (aSlice === bSlice && this._areAgreementRefListsEqual(aRefs, bRefs)) {
                    return;
                }

                let markStart = baseStart;
                let markEnd = baseEnd;
                if (markEnd <= markStart) {
                    if (merge.plainText.length > 0) {
                        if (markStart >= merge.plainText.length) {
                            markStart = Math.max(0, merge.plainText.length - 1);
                            markEnd = merge.plainText.length;
                        } else {
                            markEnd = Math.min(merge.plainText.length, markStart + 1);
                        }
                    }
                }

                const aCtx = contextWindow(parsedA.plainText, aStart, aEnd);
                const bCtx = contextWindow(parsedB.plainText, bStart, bEnd);
                const conflictId = `text:${baseStart}:${baseEnd}:${idx}`;
                merge.conflicts.push({
                    id: conflictId,
                    kind: 'text_edit',
                    start: markStart,
                    end: markEnd,
                    base_start: baseStart,
                    base_end: baseEnd,
                    text: merge.plainText.substring(markStart, markEnd),
                    base_text: baseSlice,
                    a_plain_text: aSlice,
                    b_plain_text: bSlice,
                    a_target_start: aStart,
                    a_target_end: aEnd,
                    b_target_start: bStart,
                    b_target_end: bEnd,
                    a_context_before: aCtx.before,
                    a_context_after: aCtx.after,
                    a_context_left_ellipsis: aCtx.leftEllipsis,
                    a_context_right_ellipsis: aCtx.rightEllipsis,
                    b_context_before: bCtx.before,
                    b_context_after: bCtx.after,
                    b_context_left_ellipsis: bCtx.leftEllipsis,
                    b_context_right_ellipsis: bCtx.rightEllipsis,
                    a_refs: aRefs,
                    b_refs: bRefs,
                    source_refs: sourceRefs,
                    a_annotations: aLocalAnnotations,
                    b_annotations: bLocalAnnotations,
                    a_local_annotations: aLocalAnnotations,
                    b_local_annotations: bLocalAnnotations,
                    source_annotations: sourceAnnotations,
                    selected: '',
                });
                renderRows.push({
                    kind: 'conflict',
                    start: markStart,
                    end: markEnd,
                    text: merge.plainText.substring(markStart, markEnd),
                    aRefs,
                    bRefs,
                });
            });

            merge.conflicts.sort((x, y) => (Number(x.start || 0) - Number(y.start || 0)) || (Number(x.end || 0) - Number(y.end || 0)));
            merge.conflictCount = merge.conflicts.length;

            let html = '';
            let cursor = 0;
            const sortedRows = renderRows.slice().sort((x, y) => (Number(x.start || 0) - Number(y.start || 0)) || (Number(x.end || 0) - Number(y.end || 0)));
            for (const row of sortedRows) {
                html += this._escapeAgreementHtml(merge.plainText.substring(cursor, row.start));
                if (row.kind === 'agreed') {
                    const typeToken = this._sanitizeAgreementClassToken(row.entityType);
                    const cls = typeToken ? `ann ann-${typeToken} agreement-agreed` : 'agreement-agreed';
                    html += `<span class="${cls}">${this._escapeAgreementHtml(row.text)}</span>`;
                } else {
                    const aRefs = (row.aRefs || []).join(', ') || '—';
                    const bRefs = (row.bRefs || []).join(', ') || '—';
                    html += `<span class="agreement-diff-span">${this._escapeAgreementHtml(row.text)}</span>`;
                    html += `<span class="agreement-diff-meta">${this._escapeAgreementHtml(this.agreementReviewerLabel('a'))}: ${this._escapeAgreementHtml(aRefs)} | ${this._escapeAgreementHtml(this.agreementReviewerLabel('b'))}: ${this._escapeAgreementHtml(bRefs)}</span>`;
                }
                cursor = row.end;
            }
            html += this._escapeAgreementHtml(merge.plainText.substring(cursor));
            merge.html = html;

            const inline = this._buildAgreementInlineSegments(
                merge.plainText,
                canCompareAnnotationSpans ? annotationsAForCompare : [],
                canCompareAnnotationSpans ? annotationsBForCompare : []
            );
            merge.inlineSegments = inline.segments;
            merge.selectedConflictId = (merge.conflicts[0] && merge.conflicts[0].id) || '';
            this._recomputeAgreementConflictStates();
        },

        _setAgreementVariantPayload(variantKey, payload, username = '') {
            this._ensureAgreementWorkspaceState();
            const variant = this.agreementWorkspace.versions[variantKey];
            if (!variant) return;
            const safePayload = (payload && typeof payload === 'object') ? payload : {};
            const structured = (safePayload.structured && typeof safePayload.structured === 'object')
                ? safePayload.structured
                : null;
            const editableFromPayload = (safePayload.editable_document && typeof safePayload.editable_document === 'object')
                ? safePayload.editable_document
                : null;
            const editable = editableFromPayload || structured || null;
            const hasDoc = !!(structured && String(structured.document_to_annotate || '').trim());
            variant.username = username || variant.username || '';
            variant.path = String(safePayload.path || '');
            variant.available = !!editable || hasDoc;
            variant.document_to_annotate = String(structured?.document_to_annotate || '');
            variant.editable_document = editable ? this._normalizeEditableDocument(editable) : null;
        },

        getAgreementVariant(variantKey) {
            return this.agreementWorkspace?.versions?.[variantKey] || null;
        },

        async initAgreementWorkspace() {
            if (!this.agreementWorkspace.active) return;
            this._ensureAgreementWorkspaceState();
            this.agreementWorkspace.loading = true;
            this.agreementWorkspace.error = '';
            try {
                const useReviewAgreement = !!(this.reviewTarget && isPowerUser);
                const data = useReviewAgreement
                    ? await API.getAdminReviewSubmissionSummary(this.reviewTarget, this.theme, this.doc_id)
                    : await API.getAdminSubmissionSummary(this.theme, this.doc_id);
                const run = useReviewAgreement ? (data?.campaign || null) : (data?.run || null);
                const item = data?.submission || null;
                if (!run || !item) {
                    this.agreementWorkspace.error = 'Agreement context is not available for this document.';
                    return;
                }

                this.agreementWorkspace.runId = Number(run.id || 0) || null;
                this.agreementWorkspace.runName = String(run.name || '');
                this.agreementWorkspace.status = String(item.agreement_status || 'pending');
                this.agreementWorkspace.awaitingReviewerAcceptance = Boolean(item.awaiting_reviewer_acceptance);

                const reviewerA = String(item?.reviewer_a?.username || '');
                const reviewerB = String(item?.reviewer_b?.username || '');
                this.agreementWorkspace.versions.reviewer_a.username = reviewerA;
                this.agreementWorkspace.versions.reviewer_b.username = reviewerB;
                this.agreementWorkspace.versions.source.username = String(item.source_agent || run.source_agent || 'Opus');
                this.agreementWorkspace.versions.final.username = String(item.resolved_by || item.final_source_label || 'admin');

                const [sourcePayload, reviewerAPayload, reviewerBPayload, finalPayload] = await Promise.all([
                    useReviewAgreement
                        ? API.getAdminReviewSubmissionContent(this.reviewTarget, this.theme, this.doc_id, 'source').catch(() => null)
                        : API.getAdminSubmissionContent(this.theme, this.doc_id, 'source').catch(() => null),
                    useReviewAgreement
                        ? API.getAdminReviewSubmissionContent(this.reviewTarget, this.theme, this.doc_id, 'reviewer_a').catch(() => null)
                        : API.getAdminSubmissionContent(this.theme, this.doc_id, 'reviewer_a').catch(() => null),
                    useReviewAgreement
                        ? API.getAdminReviewSubmissionContent(this.reviewTarget, this.theme, this.doc_id, 'reviewer_b').catch(() => null)
                        : API.getAdminSubmissionContent(this.theme, this.doc_id, 'reviewer_b').catch(() => null),
                    useReviewAgreement
                        ? API.getAdminReviewSubmissionContent(this.reviewTarget, this.theme, this.doc_id, 'final').catch(() => null)
                        : API.getAdminSubmissionContent(this.theme, this.doc_id, 'final').catch(() => null),
                ]);

                this._setAgreementVariantPayload('source', sourcePayload, this.agreementWorkspace.versions.source.username);
                this._setAgreementVariantPayload('reviewer_a', reviewerAPayload, reviewerA);
                this._setAgreementVariantPayload('reviewer_b', reviewerBPayload, reviewerB);
                this._setAgreementVariantPayload('final', finalPayload, this.agreementWorkspace.versions.final.username);

                const sourceDoc = this.agreementWorkspace.versions.source.editable_document || null;
                const reviewerADoc = this.agreementWorkspace.versions.reviewer_a.editable_document || null;
                const reviewerBDoc = this.agreementWorkspace.versions.reviewer_b.editable_document || null;
                const finalDoc = this.agreementWorkspace.versions.final.editable_document || null;

                // Default to Final when agreement is resolved or when an admin draft exists.
                const isResolved = String(this.agreementWorkspace.status || '').toLowerCase() === 'resolved';
                const finalSourceLabel = String(item.final_source_label || '').toLowerCase();
                const hasAdminDraft = useReviewAgreement
                    ? Boolean(item.has_final_snapshot)
                    : finalSourceLabel === 'admin_draft';
                const contestVariant = String(this.agreementWorkspace.contestVariant || '').toLowerCase();
                const contestVariantAvailable = (
                    (contestVariant === 'reviewer_a' || contestVariant === 'reviewer_b')
                    && !!this.agreementWorkspace.versions?.[contestVariant]?.available
                );
                const contestComparisonAvailable = contestVariantAvailable && !!this.agreementWorkspace.versions?.final?.available;
                const reviewerAAvailable = !!this.agreementWorkspace.versions?.reviewer_a?.available;
                const reviewerBAvailable = !!this.agreementWorkspace.versions?.reviewer_b?.available;
                const sourceAvailable = !!this.agreementWorkspace.versions?.source?.available;
                if (contestComparisonAvailable) {
                    this.agreementWorkspace.comparison.mode = 'contest';
                    this.agreementWorkspace.comparison.sideAKey = contestVariant;
                    this.agreementWorkspace.comparison.sideBKey = 'final';
                    this.agreementWorkspace.comparison.contestVariant = contestVariant;
                    this.agreementWorkspace.disableCurrentInference = true;
                } else if (
                    this.reviewTarget === 'questions'
                ) {
                    // Sequential QA review is always: source AI draft (Opus) vs single annotator output.
                    this.agreementWorkspace.comparison.mode = 'annotator_agreement';
                    this.agreementWorkspace.comparison.sideAKey = 'source';
                    this.agreementWorkspace.comparison.sideBKey = 'reviewer_a';
                    this.agreementWorkspace.comparison.contestVariant = '';
                    this.agreementWorkspace.disableCurrentInference = false;
                    if (!sourceAvailable || !reviewerAAvailable) {
                        const missing = [];
                        if (!sourceAvailable) missing.push('Opus source snapshot');
                        if (!reviewerAAvailable) missing.push('annotator submission');
                        this.agreementWorkspace.error = `QA agreement requires ${missing.join(' and ')}.`;
                    }
                } else {
                    this.agreementWorkspace.comparison.mode = 'annotator_agreement';
                    this.agreementWorkspace.comparison.sideAKey = 'reviewer_a';
                    this.agreementWorkspace.comparison.sideBKey = 'reviewer_b';
                    this.agreementWorkspace.comparison.contestVariant = '';
                    this.agreementWorkspace.disableCurrentInference = false;
                }
                this.agreementWorkspace.inferResolutionFromCurrent = isResolved || hasAdminDraft;
                if (contestComparisonAvailable) {
                    this.loadAgreementVariantIntoEditor(contestVariant, {
                        silent: true,
                        skipDirtyCheck: true,
                        allowReviewerLoad: true,
                    });
                    this.agreementWorkspace.inferResolutionFromCurrent = false;
                } else if (this.reviewTarget === 'rules' && !isResolved) {
                    this.docData = this._buildRuleAgreementWorkingDocument(
                        sourceDoc,
                        reviewerADoc,
                        reviewerBDoc,
                        hasAdminDraft ? finalDoc : null,
                    );
                    this.refreshEntities();
                    this.syncSourceEditorTextFromDoc();
                    this.dirty = false;
                    this.agreementWorkspace.loadedVariant = 'editor';
                    this.agreementWorkspace.resolveMode = true;
                    this.agreementWorkspace.inferResolutionFromCurrent = false;
                    if (String(this.currentStatus || '').toLowerCase() !== 'completed') {
                        this.currentStatus = 'in_progress';
                    }
                } else if ((isResolved || hasAdminDraft) && this.agreementWorkspace.versions.final.available) {
                    this.loadAgreementVariantIntoEditor('final', { silent: true, skipDirtyCheck: true });
                } else if (this.agreementWorkspace.versions.source.available) {
                    this.loadAgreementVariantIntoEditor('source', { silent: true, skipDirtyCheck: true });
                } else if (this.agreementWorkspace.versions.final.available) {
                    this.loadAgreementVariantIntoEditor('final', { silent: true, skipDirtyCheck: true });
                } else if (this.agreementWorkspace.versions.reviewer_a.available) {
                    this.loadAgreementVariantIntoEditor('reviewer_a', { silent: true, skipDirtyCheck: true, allowReviewerLoad: true });
                } else if (this.agreementWorkspace.versions.reviewer_b.available) {
                    this.loadAgreementVariantIntoEditor('reviewer_b', { silent: true, skipDirtyCheck: true, allowReviewerLoad: true });
                }
                this._buildAgreementMergeFromAnnotators();
            } catch (e) {
                this.agreementWorkspace.error = 'Failed to load agreement context.';
                showToast('Failed to load agreement workspace: ' + e.message, 'error');
                console.error('Agreement workspace init failed', e);
            } finally {
                this.agreementWorkspace.loading = false;
            }
        },

        loadAgreementVariantIntoEditor(variantKey, opts = {}) {
            if (!this.agreementWorkspace.active) return;
            if ((variantKey === 'reviewer_a' || variantKey === 'reviewer_b') && this.agreementWorkspace.resolveMode && !opts.allowReviewerLoad) {
                if (!opts.silent) {
                    showToast('Main document stays on Opus in Resolve mode. Reviewer details are available in conflict popups.', 'info');
                }
                return;
            }
            const variant = this.getAgreementVariant(variantKey);
            if (!variant || !variant.available || !variant.editable_document) {
                showToast('Requested version is not available', 'warning');
                return;
            }
            if (!opts.skipDirtyCheck && this.dirty) {
                if (!confirm('Discard unsaved changes and load this version?')) return;
            }

            this.docData = this._normalizeEditableDocument(variant.editable_document);
            this.refreshEntities();
            this.syncSourceEditorTextFromDoc();
            this._recomputeAgreementStructuredConflictStates();
            this.dirty = false;
            this.agreementWorkspace.loadedVariant = variantKey;
            this.agreementWorkspace.resolveMode = true;
            if (variantKey === 'final') {
                this.agreementWorkspace.inferResolutionFromCurrent = true;
            } else if (String(this.agreementWorkspace.status || '').toLowerCase() !== 'resolved') {
                this.agreementWorkspace.inferResolutionFromCurrent = false;
            }
            if (variantKey === 'reviewer_a' || variantKey === 'reviewer_b') {
                const selected = variantKey === 'reviewer_a' ? 'a' : 'b';
                for (const conflict of this.agreementWorkspace.merge.conflicts || []) {
                    conflict.selected = selected;
                }
            }
            const current = String(this.currentStatus || '').toLowerCase();
            const resolved = String(this.agreementWorkspace.status || '').toLowerCase() === 'resolved';
            const finalized = !this.agreementWorkspace.awaitingReviewerAcceptance;
            if (variantKey === 'final' || (resolved && finalized)) {
                this.currentStatus = 'completed';
            } else if (current !== 'completed' && current !== 'validated') {
                this.currentStatus = 'in_progress';
            }
            if (!opts.silent) {
                showToast(this.agreementVariantLabel(variantKey) + ' loaded into editor', 'success');
            }
        },

        _applyTextConflictChoiceGroupIntoCurrentDoc(conflicts, choice) {
            const items = (Array.isArray(conflicts) ? conflicts : [])
                .filter((item) => item && item.kind === 'text_edit')
                .slice()
                .sort((a, b) => this._conflictBaseStart(b) - this._conflictBaseStart(a));
            if (items.length === 0) return false;

            const preferred = choice === 'b' ? 'b' : 'a';
            let currentParsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
            let currentPlain = String(currentParsed.plainText || '');
            let currentAnnotations = (currentParsed.annotations || []).map((ann) => ({
                start: Number(ann.start || 0),
                end: Number(ann.end || ann.start || 0),
                text: String(ann.text || ''),
                ref: String(ann.ref || '').trim(),
            }));

            for (const conflict of items) {
                const id = String(conflict.id || '');
                if (!id) continue;

                const range = this._agreementCurrentRangeForConflict(conflict, { excludeIds: [id] });
                const start = Math.max(0, Math.min(Number(range.start || 0), currentPlain.length));
                const end = Math.max(start, Math.min(Number(range.end || start), currentPlain.length));
                const chosenText = this._agreementExpectedTextForChoice(conflict, preferred);
                const delta = chosenText.length - (end - start);

                const shifted = [];
                for (const ann of currentAnnotations) {
                    const annStart = Number(ann.start || 0);
                    const annEnd = Math.max(annStart, Number(ann.end || annStart));
                    if (annEnd <= start) {
                        shifted.push({ ...ann, start: annStart, end: annEnd });
                        continue;
                    }
                    if (annStart >= end) {
                        shifted.push({ ...ann, start: annStart + delta, end: annEnd + delta });
                        continue;
                    }
                    // Drop annotations that overlap the replaced region.
                }

                const local = preferred === 'b'
                    ? (conflict.b_local_annotations || conflict.b_annotations || [])
                    : (conflict.a_local_annotations || conflict.a_annotations || []);
                for (const ann of local) {
                    const localStart = Number(ann.start || 0);
                    const localEnd = Math.max(localStart, Number(ann.end || localStart));
                    const ref = String(ann.ref || '').trim();
                    if (!ref || localEnd <= localStart) continue;
                    const mappedStart = start + localStart;
                    const mappedEnd = start + localEnd;
                    shifted.push({
                        start: mappedStart,
                        end: mappedEnd,
                        text: String(chosenText || '').substring(localStart, localEnd),
                        ref,
                    });
                }

                currentPlain = currentPlain.substring(0, start) + chosenText + currentPlain.substring(end);
                currentAnnotations = shifted;

                this.agreementWorkspace.merge.decisions[id] = preferred;
                delete this.agreementWorkspace.merge.manualResolutions[id];
                delete this.agreementWorkspace.merge.manualTextResolutions[id];
                this._setConflictSelectedState(id, preferred);
            }

            this.docData.document_to_annotate = this._rebuildAnnotatedTextFromPlain(currentPlain, currentAnnotations);
            this.refreshEntities();
            this.markDirty();
            this.agreementWorkspace.loadedVariant = 'editor';
            return true;
        },

        _applyConflictChoiceIntoCurrentDoc(conflict, choice) {
            if (!conflict) return false;
            if (conflict.kind === 'text_edit') {
                return this._applyTextConflictChoiceGroupIntoCurrentDoc([conflict], choice);
            }

            const preferred = choice === 'b' ? 'b' : 'a';
            const selectedAnnotations = preferred === 'a'
                ? (conflict.a_annotations || [])
                : (conflict.b_annotations || []);
            const currentParsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
            const range = this._agreementCurrentRangeForConflict(conflict);
            const currentStart = Math.max(0, Number(range.start || 0));
            const currentEnd = Math.max(currentStart, Number(range.end || currentStart));
            const currentText = String(currentParsed.plainText || '').substring(currentStart, currentEnd);
            const selectedRefs = this._normalizeAgreementRefs(selectedAnnotations.map((ann) => String(ann.ref || '').trim()));

            const kept = (currentParsed.annotations || []).filter(
                (ann) => Number(ann.end) <= currentStart || Number(ann.start) >= currentEnd
            );
            const next = kept.concat(selectedRefs.map((ref) => ({
                start: currentStart,
                end: currentEnd,
                text: currentText,
                ref,
            })).filter((ann) => ann.ref && ann.end > ann.start));

            const rebuilt = this._rebuildAnnotatedTextFromPlain(String(currentParsed.plainText || ''), next);
            this.docData.document_to_annotate = rebuilt;
            this.agreementWorkspace.merge.decisions[conflict.id] = preferred;
            delete this.agreementWorkspace.merge.manualResolutions[conflict.id];
            delete this.agreementWorkspace.merge.manualTextResolutions[conflict.id];
            this.refreshEntities();
            this.markDirty();
            this.agreementWorkspace.loadedVariant = 'editor';
            this._setConflictSelectedState(conflict.id, preferred);
            return true;
        },

        _applyConflictChoiceGroupIntoCurrentDoc(conflicts, choice) {
            const items = Array.isArray(conflicts) ? conflicts.filter(Boolean) : [];
            if (items.length === 0) return false;

            const preferred = choice === 'b' ? 'b' : 'a';
            const textItems = items.filter((item) => item.kind === 'text_edit');
            const annItems = items.filter((item) => item.kind !== 'text_edit');
            let ok = false;

            if (textItems.length > 0) {
                ok = this._applyTextConflictChoiceGroupIntoCurrentDoc(textItems, preferred) || ok;
            }
            if (annItems.length === 0) {
                return ok;
            }

            const currentParsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
            const conflictRanges = annItems.map((item) => ({
                conflict: item,
                range: this._agreementCurrentRangeForConflict(item),
            }));
            const kept = (currentParsed.annotations || []).filter(
                (ann) => !conflictRanges.some(({ range }) => (
                    Number(ann.end) > Number(range.start || 0) &&
                    Number(ann.start) < Number(range.end || range.start || 0)
                ))
            );

            const selected = [];
            const seen = new Set();
            for (const { conflict, range } of conflictRanges) {
                const selectedAnnotations = preferred === 'a'
                    ? (conflict.a_annotations || [])
                    : (conflict.b_annotations || []);
                const start = Math.max(0, Number(range.start || 0));
                const end = Math.max(start, Number(range.end || start));
                const refs = this._normalizeAgreementRefs(
                    selectedAnnotations.map((ann) => String(ann.ref || '').trim())
                );
                const currentText = String(currentParsed.plainText || '').substring(start, end);

                for (const ref of refs) {
                    if (!ref || end <= start) continue;
                    const key = `${start}:${end}:${ref}`;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    selected.push({
                        start,
                        end,
                        text: currentText,
                        ref,
                    });
                }
            }

            const rebuilt = this._rebuildAnnotatedTextFromPlain(
                String(currentParsed.plainText || ''),
                kept.concat(selected)
            );
            this.docData.document_to_annotate = rebuilt;

            for (const conflict of annItems) {
                const id = String(conflict.id || '');
                if (!id) continue;
                this.agreementWorkspace.merge.decisions[id] = preferred;
                delete this.agreementWorkspace.merge.manualResolutions[id];
                delete this.agreementWorkspace.merge.manualTextResolutions[id];
                this._setConflictSelectedState(id, preferred);
            }

            this.refreshEntities();
            this.markDirty();
            this.agreementWorkspace.loadedVariant = 'editor';
            return true;
        },

        applyAgreementConflictChoice(conflictId, choice, opts = {}) {
            if (!this.agreementWorkspace.active) return;
            const advance = opts.advance !== false;
            const silentToast = opts.silentToast === true;
            const conflict = this.getAgreementConflictById(conflictId);
            if (!conflict) {
                if (!silentToast) showToast('Conflict not found', 'warning');
                return false;
            }

            if (conflict.kind === 'rule' || conflict.kind === 'question') {
                const ok = this.applyAgreementStructuredConflictChoice(conflictId, choice);
                if (ok && advance) {
                    const unresolvedRules = this.getAgreementUnresolvedRuleConflicts();
                    const unresolvedQuestions = this.getAgreementUnresolvedQuestionConflicts();
                    const nextConflict = unresolvedRules[0] || unresolvedQuestions[0] || null;
                    if (nextConflict) {
                        this.selectAgreementConflict(nextConflict.id, { silent: true });
                    } else {
                        this.selectAgreementConflict('', { silent: true });
                    }
                }
                return ok;
            }

            // Apply choice atomically across linked/overlapping conflicts to avoid
            // one conflict update reverting another.
            const conflicts = this.agreementWorkspace.merge.conflicts || [];
            const linkedIds = this._expandAgreementConflictGroupFromSeeds([conflict.id]);
            const linkedConflicts = linkedIds
                .map((id) => this.getAgreementConflictById(id))
                .filter((item) => item && item.resolution === 'unresolved');
            let ok = false;
            if (linkedConflicts.length > 1) {
                ok = this._applyConflictChoiceGroupIntoCurrentDoc(linkedConflicts, choice);
            } else {
                ok = this._applyConflictChoiceIntoCurrentDoc(conflict, choice);
            }
            if (ok) {
                if (advance) {
                    this.selectAdjacentAgreementConflict(1, true);
                } else {
                    this.selectAgreementConflict(conflict.id, { silent: true });
                }
                if (!silentToast) {
                    showToast(`Applied ${choice === 'b' ? this.agreementReviewerLabel('b') : this.agreementReviewerLabel('a')} for selected difference`, 'success');
                }
                return true;
            }
            return false;
        },

        applyAllAgreementConflicts(choice) {
            if (!this.agreementWorkspace.active) return;
            const conflicts = this.agreementWorkspace.merge.conflicts || [];
            if (conflicts.length === 0) return;
            const preferred = choice === 'b' ? 'b' : 'a';
            if (!confirm(`Apply ${preferred === 'a' ? this.agreementReviewerLabel('a') : this.agreementReviewerLabel('b')} to all differences?`)) return;

            const unresolved = conflicts.filter((item) => item.resolution === 'unresolved');
            let applied = 0;
            const visited = new Set();
            for (const conflict of unresolved) {
                if (visited.has(conflict.id)) continue;
                const groupIds = this._expandAgreementConflictGroupFromSeeds([conflict.id]);
                const group = groupIds
                    .map((id) => this.getAgreementConflictById(id))
                    .filter((item) => item && item.resolution === 'unresolved');
                for (const item of group) visited.add(item.id);
                if (group.length === 0) continue;

                let ok = false;
                if (group.length > 1) {
                    ok = this._applyConflictChoiceGroupIntoCurrentDoc(group, preferred);
                    if (ok) {
                        applied += group.length;
                    }
                } else {
                    ok = this._applyConflictChoiceIntoCurrentDoc(group[0], preferred);
                    if (ok) {
                        applied += 1;
                    }
                }
            }
            if (applied > 0) {
                this.selectAdjacentAgreementConflict(1, true);
                showToast(`Applied ${preferred === 'a' ? this.agreementReviewerLabel('a') : this.agreementReviewerLabel('b')} to ${applied} difference(s)`, 'success');
            }
        },

        preferredEntityTypeOrder() {
            return window.DEFAULT_ENTITY_TYPES || [
                'person',
                'place',
                'event',
                'military_org',
                'entreprise_org',
                'ngo',
                'government_org',
                'educational_org',
                'media_org',
                'temporal',
                'number',
                'award',
                'legal',
                'product',
            ];
        },

        mergeEntityTypes(primary = [], secondary = []) {
            const set = new Set(
                [...(primary || []), ...(secondary || [])]
                    .filter(Boolean)
                    .filter((entityType) => !['organization', 'organisation'].includes(String(entityType).trim()))
            );
            const ordered = [];
            for (const t of this.preferredEntityTypeOrder()) {
                if (set.has(t)) {
                    ordered.push(t);
                    set.delete(t);
                }
            }
            return ordered.concat(Array.from(set).sort());
        },

        computeEntityTypes() {
            const taxonomyTypes = Object.keys(this.taxonomy || {});
            const discoveredTypes = Object.keys(extractEntitiesFromText(this.docData?.document_to_annotate || '', taxonomyTypes));
            const merged = this.mergeEntityTypes(taxonomyTypes, discoveredTypes);
            return merged.length ? merged : this.preferredEntityTypeOrder();
        },

        // --- Entity management ---
        refreshEntities() {
            if (!this.docData) return;
            this.entityGroups = extractEntitiesFromText(this.docData.document_to_annotate || '', this.entityTypes);
            const mergedTypes = this.mergeEntityTypes(this.entityTypes, Object.keys(this.entityGroups || {}));
            if (JSON.stringify(mergedTypes) !== JSON.stringify(this.entityTypes)) {
                this.entityTypes = mergedTypes;
                this.applyDynamicEntityTypeStyles();
            }
            if (this.agreementWorkspace.active) {
                this._recomputeAgreementConflictStates();
                this._recomputeAgreementStructuredConflictStates();
            }
            if (this.sourceView) {
                this.syncSourceEditorTextFromDoc();
            }
            // Attach hover listeners to annotation spans after rendering
            this.$nextTick(() => {
                this.attachAnnotationHoverListeners();
                this.attachQuestionAnnotationHoverListeners();
                this.attachRuleReferenceHoverListeners();
                this.enhanceTooltipsWithMetadata();
            });
        },

        applyDynamicEntityTypeStyles() {
            const fixed = {
                person: { bg: '#dbeafe', fg: '#1e40af' },
                place: { bg: '#dcfce7', fg: '#166534' },
                event: { bg: '#f3e8ff', fg: '#6b21a8' },
                military_org: { bg: '#e2e8f0', fg: '#334155' },
                entreprise_org: { bg: '#ffedd5', fg: '#9a3412' },
                ngo: { bg: '#ccfbf1', fg: '#0f766e' },
                government_org: { bg: '#fef3c7', fg: '#854d0e' },
                educational_org: { bg: '#e0e7ff', fg: '#4338ca' },
                media_org: { bg: '#ffe4e6', fg: '#be123c' },
                temporal: { bg: '#fef9c3', fg: '#854d0e' },
                number: { bg: '#fce7f3', fg: '#9d174d' },
                award: { bg: '#fde68a', fg: '#92400e' },
                legal: { bg: '#e2e8f0', fg: '#1f2937' },
                product: { bg: '#cffafe', fg: '#0e7490' },
            };
            const palette = [
                ['#ccfbf1', '#0f766e'],
                ['#fee2e2', '#b91c1c'],
                ['#fef3c7', '#92400e'],
                ['#dbeafe', '#1d4ed8'],
                ['#ede9fe', '#5b21b6'],
                ['#dcfce7', '#166534'],
                ['#fce7f3', '#9d174d'],
                ['#cffafe', '#0e7490'],
                ['#ffedd5', '#9a3412'],
                ['#e2e8f0', '#334155'],
                ['#f1f5f9', '#1f2937'],
                ['#ecfeff', '#155e75'],
                ['#ecfccb', '#4d7c0f'],
                ['#fae8ff', '#7e22ce'],
                ['#e0e7ff', '#4338ca'],
                ['#fef9c3', '#a16207'],
                ['#ffe4e6', '#be123c'],
                ['#e0f2fe', '#0369a1'],
                ['#f0fdf4', '#14532d'],
                ['#fee2e2', '#7f1d1d'],
            ];

            const allTypes = (this.entityTypes || []).filter(Boolean);
            const colorMap = {};
            const used = new Set();

            for (const type of allTypes) {
                if (fixed[type]) {
                    colorMap[type] = fixed[type];
                    used.add(fixed[type].fg.toLowerCase());
                }
            }

            const unknown = allTypes.filter((type) => !fixed[type]).sort();
            for (const type of unknown) {
                const hash = [...type].reduce((acc, ch) => ((acc << 5) - acc + ch.charCodeAt(0)) | 0, 0);
                const start = Math.abs(hash) % palette.length;
                let selected = null;
                for (let i = 0; i < palette.length; i++) {
                    const [bg, fg] = palette[(start + i) % palette.length];
                    if (!used.has(fg.toLowerCase())) {
                        selected = { bg, fg };
                        break;
                    }
                }
                if (!selected) {
                    selected = { bg: '#e2e8f0', fg: '#334155' };
                }
                used.add(selected.fg.toLowerCase());
                colorMap[type] = selected;
            }

            const styleId = 'dynamic-entity-type-styles';
            let styleEl = document.getElementById(styleId);
            if (!styleEl) {
                styleEl = document.createElement('style');
                styleEl.id = styleId;
                document.head.appendChild(styleEl);
            }

            const rules = [];
            for (const type of allTypes) {
                const colors = colorMap[type];
                if (!colors) continue;
                const cls = String(type).replace(/[^a-zA-Z0-9_-]/g, '-');
                rules.push(`.ann-${cls}{background:${colors.bg};color:${colors.fg};}`);
                rules.push(`.entity-type-btn-${cls}{background:${colors.bg};color:${colors.fg};}`);
                rules.push(`.rule-chip-${cls}{background:${colors.bg};color:${colors.fg};}`);
            }
            styleEl.textContent = rules.join('\n');
        },

        enhanceTooltipsWithMetadata() {
            if (!isPowerUser) {
                return;
            }
            
            // Add creator/editor info to annotation tooltips
            const annotations = document.querySelectorAll('.ann');
            
            annotations.forEach(ann => {
                const ref = ann.dataset.ref;  // e.g., "person_1.name"
                
                // Build tooltip with entity reference
                let tooltip = ref || '';
                
                // Look up metadata by entity reference only (not position)
                const metadata = this.annotationMetadata?.annotations?.[ref];
                
                if (metadata) {
                    // Has history - show who created/edited it
                    const creator = metadata.username || 'Unknown';
                    const timestamp = metadata.timestamp ? new Date(metadata.timestamp).toLocaleString() : '';
                    const lastEditor = metadata.last_editor || creator;
                    const lastModified = metadata.last_modified ? new Date(metadata.last_modified).toLocaleString() : timestamp;
                    
                    if (lastEditor === creator) {
                        tooltip += `\n\nCreated by: ${creator}`;
                        if (timestamp) tooltip += `\n${timestamp}`;
                    } else {
                        tooltip += `\n\nCreated by: ${creator}`;
                        tooltip += `\nLast edited by: ${lastEditor}`;
                        if (lastModified) tooltip += `\n${lastModified}`;
                    }
                } else if (!this.annotationMetadata?.has_history) {
                    // No history at all - mark as original
                    tooltip += '\n\nOriginal annotation';
                }
                
                ann.title = tooltip;
            });
            
            // Enhance question tooltips
            const questions = document.querySelectorAll('.question-card');
            questions.forEach(card => {
                const qidElement = card.querySelector('.question-id');
                if (qidElement) {
                    const qid = qidElement.textContent.trim();
                    const metadata = this.annotationMetadata?.questions?.[qid];
                    
                    if (metadata) {
                        const creator = metadata.username || 'Unknown';
                        const timestamp = metadata.timestamp ? new Date(metadata.timestamp).toLocaleString() : '';
                        qidElement.title = `Created by: ${creator}${timestamp ? '\n' + timestamp : ''}`;
                        qidElement.style.cursor = 'help';
                    } else if (!this.annotationMetadata?.has_history) {
                        qidElement.title = 'Original question';
                        qidElement.style.cursor = 'help';
                    }
                }
            });
            
            // Enhance rule tooltips
            const rulesList = this.$refs.rulesList;
            if (rulesList) {
                const ruleItems = rulesList.querySelectorAll('.rule-card[data-rule-kind="explicit"]');
                ruleItems.forEach((item, idx) => {
                    const ruleText = this.docData?.rules?.[idx];
                    
                    if (ruleText) {
                        const metadata = this.annotationMetadata?.rules?.[ruleText];
                        
                        if (metadata) {
                            const creator = metadata.username || 'Unknown';
                            const timestamp = metadata.timestamp ? new Date(metadata.timestamp).toLocaleString() : '';
                            item.title = `Created by: ${creator}${timestamp ? '\n' + timestamp : ''}`;
                            item.style.cursor = 'help';
                        } else if (!this.annotationMetadata?.has_history) {
                            item.title = 'Original rule';
                            item.style.cursor = 'help';
                        }
                    }
                });
            }
        },

        attachQuestionAnnotationHoverListeners() {
            // Attach hover listeners to all read-only question annotations
            // (question text, answer, and reasoning chain).
            const questionAnns = document.querySelectorAll('.question-readonly .ann');
            console.log('Attaching listeners to', questionAnns.length, 'question annotations');
            
            questionAnns.forEach(el => {
                // Remove existing listeners to avoid duplicates
                const newEl = el.cloneNode(true);
                el.parentNode.replaceChild(newEl, el);
                
                newEl.addEventListener('mouseenter', (e) => {
                    if (this.lockedHighlight) return;
                    const entityId = e.currentTarget.dataset.entityId;
                    console.log('Hover on entity:', entityId);
                    if (entityId) {
                        highlightEntity(entityId, e.currentTarget);
                    }
                });
                
                newEl.addEventListener('mouseleave', () => {
                    if (this.lockedHighlight) return;
                    clearEntityHighlight();
                });
                
                // Single click to lock/unlock highlight
                newEl.addEventListener('click', (e) => {
                    if (e.target.classList.contains('resize-handle') || 
                        e.target.classList.contains('ann-delete-btn')) return;
                    
                    const entityId = newEl.dataset.entityId;
                    
                    if (this.lockedHighlight === entityId) {
                        this.lockedHighlight = null;
                        clearEntityHighlight();
                    } else {
                        this.lockedHighlight = entityId;
                        highlightEntity(entityId, newEl);
                    }
                });

                newEl.addEventListener('dblclick', (e) => {
                    if (e.target.classList.contains('resize-handle') ||
                        e.target.classList.contains('ann-delete-btn')) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const questionCard = newEl.closest('.question-card');
                    if (!questionCard) return;
                    this.editQuestionAnnotation(newEl);
                });

                newEl.querySelectorAll('.resize-handle').forEach((handle) => {
                    handle.addEventListener('mousedown', (evt) => this.startQuestionDragResize(evt));
                });

                const btn = newEl.querySelector('.ann-delete-btn');
                if (btn) {
                    btn.addEventListener('click', (evt) => {
                        evt.preventDefault();
                        evt.stopPropagation();
                        this.deleteAnnotationFromQuestionSpan(newEl);
                    });
                }
            });
        },

        attachRuleReferenceHoverListeners() {
            const ruleRefs = document.querySelectorAll('.rule-readonly .ann');
            for (const el of Array.from(ruleRefs)) {
                if (el.dataset.ruleHoverBound === '1') continue;
                el.dataset.ruleHoverBound = '1';

                el.addEventListener('mouseenter', (e) => {
                    if (this.lockedHighlight) return;
                    const entityId = e.currentTarget?.dataset?.entityId || '';
                    const hoveredRef = String(e.currentTarget?.dataset?.ref || '').trim();
                    if (entityId) {
                        highlightEntity(entityId, e.currentTarget);
                    }
                    if (hoveredRef) {
                        document.querySelectorAll('.ann').forEach((candidate) => {
                            if (String(candidate?.dataset?.ref || '').trim() === hoveredRef) {
                                candidate.classList.add('glow-ref');
                                candidate.classList.remove('glow-dim');
                            }
                        });
                        // Make the exact hovered attribute stand out against same-entity siblings.
                        document.querySelectorAll('.ann.glow, .ann.glow-strong').forEach((candidate) => {
                            const candidateRef = String(candidate?.dataset?.ref || '').trim();
                            if (candidateRef && candidateRef === hoveredRef) {
                                candidate.classList.remove('glow-dim');
                            } else {
                                candidate.classList.add('glow-dim');
                            }
                        });
                    }
                });

                el.addEventListener('mouseleave', () => {
                    if (this.lockedHighlight) return;
                    clearEntityHighlight();
                });

                el.addEventListener('click', () => {
                    const entityId = el.dataset.entityId;
                    if (!entityId) return;
                    if (this.lockedHighlight === entityId) {
                        this.lockedHighlight = null;
                        clearEntityHighlight();
                    } else {
                        this.lockedHighlight = entityId;
                        highlightEntity(entityId, el);
                    }
                });
            }
        },

        attachAnnotationHoverListeners() {
            const container = this.$refs.docText;
            if (!container) return;

            // Replacing each annotation node clears stale listeners after x-html rerenders.
            const nodes = Array.from(container.querySelectorAll('.ann'));
            for (const original of nodes) {
                const el = original.cloneNode(true);
                if (original.parentNode) {
                    original.parentNode.replaceChild(el, original);
                }

                el.addEventListener('mouseenter', (e) => {
                    if (this.lockedHighlight) return;
                    const entityId = e.currentTarget?.dataset?.entityId || '';
                    if (entityId) {
                        highlightEntity(entityId, e.currentTarget);
                    }
                });
                el.addEventListener('mouseleave', () => {
                    if (this.lockedHighlight) return;
                    clearEntityHighlight();
                });

                el.addEventListener('click', (e) => {
                    if (e.target.classList.contains('resize-handle') ||
                        e.target.classList.contains('ann-delete-btn')) return;

                    const entityRef = String(el.dataset.ref || '').trim();
                    const displayText = this._annotationDisplayText(el);
                    if (
                        this.canEditQuestions
                        && this.editingQuestion !== null
                        && Number.isInteger(this.qaInsertionTarget?.questionIdx)
                        && ['question', 'answer'].includes(String(this.qaInsertionTarget?.field || ''))
                        && entityRef
                        && displayText
                    ) {
                        this.insertEntityAnnotationIntoActiveQuestion(entityRef, displayText);
                    }

                    const entityId = el.dataset.entityId;
                    if (this.lockedHighlight === entityId) {
                        this.lockedHighlight = null;
                        clearEntityHighlight();
                    } else {
                        this.lockedHighlight = entityId;
                        highlightEntity(entityId, el);
                    }
                });

                el.addEventListener('dblclick', (e) => {
                    if (e.target.classList.contains('resize-handle') ||
                        e.target.classList.contains('ann-delete-btn')) return;
                    e.preventDefault();
                    e.stopPropagation();
                    if (this.agreementClickTimer) {
                        clearTimeout(this.agreementClickTimer);
                        this.agreementClickTimer = null;
                    }
                    if (this.agreementWorkspace.active && this.agreementWorkspace.conflictModal.show) {
                        this.closeAgreementConflictModal();
                    }
                    this.editAnnotation(el);
                });

                el.querySelectorAll('.resize-handle').forEach((handle) => {
                    handle.addEventListener('mousedown', (evt) => this.startDragResize(evt));
                });

                const btn = el.querySelector('.ann-delete-btn');
                if (btn) {
                    btn.addEventListener('click', (evt) => {
                        evt.preventDefault();
                        evt.stopPropagation();
                        this.deleteAnnotationInstance(el);
                    });
                }
            }
        },

        deleteAnnotationFromQuestion(annSpan, questionCard) {
            if (!this.canEditQuestions) return;
            const start = parseInt(annSpan.dataset.start);
            const end = parseInt(annSpan.dataset.end);
            const ref = annSpan.dataset.ref;
            
            if (!confirm(`Delete annotation "${ref}" from question?`)) return;
            
            // Find which question this belongs to
            const questionIndex = Array.from(document.querySelectorAll('.question-card')).indexOf(questionCard);
            if (questionIndex === -1) return;
            
            const question = this.docData.questions[questionIndex];
            if (!question) return;
            
            this.pushUndo();
            
            // Remove annotation from question text
            if (question.question) {
                question.question = removeAnnotation(question.question, start, end);
            }
            
            this.refreshEntities();
            this.markDirty();
            showToast('Annotation deleted from question', 'success');
        },

        _resolveQuestionAnnotationContext(annSpan) {
            if (!annSpan) return null;
            const questionCard = annSpan.closest('.question-card');
            if (!questionCard) return null;

            const allCards = Array.from(document.querySelectorAll('.question-card'));
            const questionIdx = allCards.indexOf(questionCard);
            if (questionIdx < 0) return null;

            const isAnswer = !!annSpan.closest('.q-answer-text');
            const field = isAnswer ? 'answer' : 'question';
            const fieldContainer = questionCard.querySelector(isAnswer ? '.q-answer-text' : '.q-text');
            if (!fieldContainer) return null;

            return { questionIdx, field, fieldContainer };
        },

        editQuestionAnnotation(annSpan) {
            if (!this.canEditQuestions) return;
            const start = parseInt(annSpan.dataset.start);
            const end = parseInt(annSpan.dataset.end);
            const entityId = annSpan.dataset.entityId;
            const entityType = annSpan.dataset.entityType;
            const ref = annSpan.dataset.ref;
            const ctx = this._resolveQuestionAnnotationContext(annSpan);
            if (!ctx) return;

            const parsedAttr = this.parsePopupAttributeRef(ref);

            let textContent = '';
            for (const child of annSpan.childNodes) {
                if (child.nodeType === Node.TEXT_NODE) {
                    textContent += child.textContent;
                }
            }

            const rect = annSpan.getBoundingClientRect();
            const popupWidth = 350;
            const popupHeight = 400;
            const sidebarWidth = 220;

            let x = rect.left + (rect.width / 2);
            let y = rect.top - 10;

            if (y + popupHeight > window.innerHeight) {
                y = rect.top - popupHeight - 10;
            }
            if (y < 10) y = 10;

            const minX = sidebarWidth + 30;
            if (x < minX) x = minX;

            const maxX = window.innerWidth - popupWidth - 20;
            if (x > maxX) x = maxX;

            this.popup.show = true;
            this.popup.x = x;
            this.popup.y = y;
            this.popup.selectedText = textContent.trim();
            this.popup.rawStart = start;
            this.popup.rawEnd = end;
            this.popup.entityType = entityType;
            this.popup.entityId = entityId;
            this.popup.attribute = parsedAttr.attribute;
            this.popup.relationshipTarget = parsedAttr.relationshipTarget;
            this.popup.editing = true;
            this.popup.editRef = ref;
            this.popup.questionIdx = ctx.questionIdx;
            this.popup.questionField = ctx.field;
            this.$nextTick(() => this._repositionAnnotationPopup());
        },

        deleteAnnotationFromQuestionSpan(annSpan) {
            if (!this.canEditQuestions) return;
            const ctx = this._resolveQuestionAnnotationContext(annSpan);
            if (!ctx) return;
            const start = parseInt(annSpan.dataset.start);
            const end = parseInt(annSpan.dataset.end);
            const ref = annSpan.dataset.ref;

            if (!confirm(`Delete annotation "${ref}" from ${ctx.field}?`)) return;

            const question = this.docData.questions?.[ctx.questionIdx];
            if (!question) return;

            this.pushUndo();
            const currentText = ctx.field === 'question' ? String(question.question || '') : String(question.answer || '');
            const nextText = removeAnnotation(currentText, start, end);
            if (ctx.field === 'question') {
                question.question = nextText;
            } else {
                question.answer = nextText;
            }

            this.refreshEntities();
            this.markDirty();
            showToast(`Annotation deleted from ${ctx.field}`, 'success');
            this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
        },

        deleteAnnotationInstance(annSpan) {
            if (!this.canEditDocumentAnnotations) return;
            const start = parseInt(annSpan.dataset.start);
            const end = parseInt(annSpan.dataset.end);
            const ref = annSpan.dataset.ref;
            
            if (!confirm(`Delete annotation "${ref}"?`)) return;
            
            this.pushUndo();
            const raw = this.docData.document_to_annotate;
            this.docData.document_to_annotate = removeAnnotation(raw, start, end);
            this.refreshEntities();
            this.markDirty();
            showToast('Annotation deleted', 'success');
        },

        deleteEntity(entityId) {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            const entityGroups = (this.entityGroups && typeof this.entityGroups === 'object') ? this.entityGroups : {};
            const entity = Object.values(entityGroups).flat().find(e => e.id === entityId);
            if (!entity) return;
            
            const confirmMsg = `Delete entity "${entityId}" and all its ${entity.count} instance(s)?`;
            if (!confirm(confirmMsg)) return;
            
            this.pushUndo();
            
            // Parse annotations to find all instances of this entity
            let raw = this.docData.document_to_annotate;
            const annotations = parseAnnotations(raw);
            
            // Sort by position (descending) to remove from end to start
            const toRemove = annotations
                .filter(ann => ann.entityId === entityId)
                .sort((a, b) => b.start - a.start);
            
            // Remove each annotation
            for (const ann of toRemove) {
                raw = removeAnnotation(raw, ann.start, ann.end);
            }
            
            this.docData.document_to_annotate = raw;
            this.refreshEntities();
            this.markDirty();
            showToast(`Entity "${entityId}" deleted (${toRemove.length} instances removed)`, 'success');
        },

        editAnnotation(annSpan) {
            if (!this.canEditDocumentAnnotations) return;
            const start = parseInt(annSpan.dataset.start);
            const end = parseInt(annSpan.dataset.end);
            const entityId = annSpan.dataset.entityId;
            const entityType = annSpan.dataset.entityType;
            const ref = annSpan.dataset.ref;
            
            // Extract current attribute
            const parsedAttr = this.parsePopupAttributeRef(ref);
            
            // Extract just the text content (skip handles and delete button)
            let textContent = '';
            for (const child of annSpan.childNodes) {
                if (child.nodeType === Node.TEXT_NODE) {
                    textContent += child.textContent;
                }
            }
            
            // Get the rect for positioning
            const rect = annSpan.getBoundingClientRect();
            
            const popupWidth = 350; // max-width of popup
            const popupHeight = 400; // estimated height of popup
            const sidebarWidth = 220; // width of left sidebar from CSS
            
            let x = rect.left + (rect.width / 2);
            let y = rect.top - 10;
            
            // Check if popup would go off bottom of screen
            if (y + popupHeight > window.innerHeight) {
                // Position above the annotation instead
                y = rect.top - popupHeight - 10;
            }
            
            // Ensure popup doesn't go off top of screen
            if (y < 10) {
                y = 10;
            }
            
            // Ensure popup doesn't overlap with left sidebar
            const minX = sidebarWidth + 30;
            if (x < minX) {
                x = minX;
            }
            
            // Ensure popup doesn't go off right edge of screen
            const maxX = window.innerWidth - popupWidth - 20;
            if (x > maxX) {
                x = maxX;
            }
            
            // Show edit popup
            this.popup.show = true;
            this.popup.x = x;
            this.popup.y = y;
            this.popup.selectedText = textContent.trim();
            this.popup.rawStart = start;
            this.popup.rawEnd = end;
            this.popup.entityType = entityType;
            this.popup.entityId = entityId;
            this.popup.attribute = parsedAttr.attribute;
            this.popup.relationshipTarget = parsedAttr.relationshipTarget;
            this.popup.editing = true;
            this.popup.editRef = ref;
            this.popup.questionIdx = undefined;
            this.popup.questionField = null;
            this.$nextTick(() => this._repositionAnnotationPopup());
        },

        startDragResize(e) {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            e.preventDefault();
            e.stopPropagation();
            
            const handle = e.target;
            const annSpan = handle.closest('.ann');
            if (!annSpan) return;
            
            const side = handle.dataset.side;
            const entityRef = annSpan.dataset.ref;
            const oldStart = parseInt(annSpan.dataset.start);
            const oldEnd = parseInt(annSpan.dataset.end);
            
            const raw = this.docData.document_to_annotate;
            const originalRaw = raw;
            const container = this.$refs.docText;
            
            // Visual feedback
            annSpan.classList.add('resizing');
            document.body.style.cursor = 'ew-resize';
            document.body.classList.add('annotation-interacting');
            
            // Track the boundary we're moving
            let newBoundary = side === 'left' ? oldStart : oldEnd;
            let lastRenderedBoundary = newBoundary;
            let isMoving = false;
            let hasInvalidTarget = false;
            
            // Throttle for performance
            let lastUpdate = 0;
            const throttleMs = 16; // ~60fps
            
            const onMouseMove = (moveEvent) => {
                const now = Date.now();
                if (now - lastUpdate < throttleMs) return;
                lastUpdate = now;
                
                isMoving = true;
                
                // Get character position under cursor
                const range = this.caretRangeFromPoint(moveEvent.clientX, moveEvent.clientY);
                if (!range || !container) return;

                // Calculate position in rendered text, excluding delete buttons and resize handles
                const preRange = document.createRange();
                preRange.setStart(container, 0);
                preRange.setEnd(range.startContainer, range.startOffset);
                const renderedOffset = this.getVisibleTextOffset(preRange);

                // Convert to raw position
                const rawPos = this.renderedToRawOffset(renderedOffset);
                
                // Update boundary position
                if (side === 'left') {
                    // Moving left boundary - must stay before right boundary
                    newBoundary = Math.max(0, Math.min(rawPos, oldEnd - 1));
                } else {
                    // Moving right boundary - must stay after left boundary
                    newBoundary = Math.max(oldStart + 1, Math.min(rawPos, raw.length));
                }
                
                // Only re-render if boundary changed
                if (newBoundary === lastRenderedBoundary) return;
                lastRenderedBoundary = newBoundary;
                
                // Calculate preview positions
                const previewStart = side === 'left' ? newBoundary : oldStart;
                const previewEnd = side === 'right' ? newBoundary : oldEnd;

                // Prevent resizing into another annotation span.
                const isValid = isValidAnnotationPosition(
                    originalRaw,
                    previewStart,
                    previewEnd,
                    oldStart,
                    oldEnd
                );
                hasInvalidTarget = !isValid;
                if (!isValid) {
                    return;
                }
                
                // Apply resize and render
                try {
                    const previewRaw = resizeAnnotation(originalRaw, oldStart, oldEnd, previewStart, previewEnd);
                    container.innerHTML = renderAnnotatedHtml(previewRaw);
                } catch (err) {
                    console.error('Resize preview error:', err);
                }
            };
            
            const onMouseUp = (upEvent) => {
                upEvent.preventDefault();
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                
                document.body.style.cursor = '';
                document.body.classList.remove('annotation-interacting');
                
                // If didn't move, just restore
                if (!isMoving) {
                    container.innerHTML = renderAnnotatedHtml(originalRaw);
                    this.attachAnnotationHoverListeners();
                    return;
                }
                
                // Calculate final positions
                const finalStart = side === 'left' ? newBoundary : oldStart;
                const finalEnd = side === 'right' ? newBoundary : oldEnd;

                // Final collision check: never commit overlap with another annotation.
                const isValid = isValidAnnotationPosition(
                    originalRaw,
                    finalStart,
                    finalEnd,
                    oldStart,
                    oldEnd
                );
                if (!isValid) {
                    container.innerHTML = renderAnnotatedHtml(originalRaw);
                    this.attachAnnotationHoverListeners();
                    if (hasInvalidTarget) {
                        showToast('Cannot resize into another annotation span', 'error');
                    }
                    return;
                }
                
                // If no change, restore
                if (finalStart === oldStart && finalEnd === oldEnd) {
                    container.innerHTML = renderAnnotatedHtml(originalRaw);
                    this.attachAnnotationHoverListeners();
                    return;
                }
                
                // Apply permanently
                this.applyAnnotationResize(oldStart, oldEnd, finalStart, finalEnd, entityRef);
            };
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        },

        startQuestionDragResize(e) {
            if (!this.canEditQuestions) return;
            e.preventDefault();
            e.stopPropagation();

            const handle = e.target;
            const annSpan = handle.closest('.ann');
            if (!annSpan) return;
            const ctx = this._resolveQuestionAnnotationContext(annSpan);
            if (!ctx) return;

            const question = this.docData.questions?.[ctx.questionIdx];
            if (!question) return;

            const side = handle.dataset.side;
            const oldStart = parseInt(annSpan.dataset.start);
            const oldEnd = parseInt(annSpan.dataset.end);
            const originalRaw = ctx.field === 'question' ? String(question.question || '') : String(question.answer || '');
            const container = ctx.fieldContainer;

            annSpan.classList.add('resizing');
            document.body.style.cursor = 'ew-resize';
            document.body.classList.add('annotation-interacting');

            let newBoundary = side === 'left' ? oldStart : oldEnd;
            let lastRenderedBoundary = newBoundary;
            let isMoving = false;
            let hasInvalidTarget = false;

            let lastUpdate = 0;
            const throttleMs = 16;

            const onMouseMove = (moveEvent) => {
                const now = Date.now();
                if (now - lastUpdate < throttleMs) return;
                lastUpdate = now;
                isMoving = true;

                const range = this.caretRangeFromPoint(moveEvent.clientX, moveEvent.clientY);
                if (!range || !container) return;

                const preRange = document.createRange();
                preRange.setStart(container, 0);
                preRange.setEnd(range.startContainer, range.startOffset);
                const renderedOffset = this.getVisibleTextOffset(preRange, container);
                const rawPos = this.renderedToRawOffset(renderedOffset, originalRaw);

                if (side === 'left') {
                    newBoundary = Math.max(0, Math.min(rawPos, oldEnd - 1));
                } else {
                    newBoundary = Math.max(oldStart + 1, Math.min(rawPos, originalRaw.length));
                }
                if (newBoundary === lastRenderedBoundary) return;
                lastRenderedBoundary = newBoundary;

                const previewStart = side === 'left' ? newBoundary : oldStart;
                const previewEnd = side === 'right' ? newBoundary : oldEnd;
                const isValid = isValidAnnotationPosition(
                    originalRaw,
                    previewStart,
                    previewEnd,
                    oldStart,
                    oldEnd
                );
                hasInvalidTarget = !isValid;
                if (!isValid) return;

                try {
                    const previewRaw = resizeAnnotation(originalRaw, oldStart, oldEnd, previewStart, previewEnd);
                    container.innerHTML = renderAnnotatedHtml(previewRaw);
                } catch (err) {
                    console.error('Question resize preview error:', err);
                }
            };

            const onMouseUp = (upEvent) => {
                upEvent.preventDefault();
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);

                document.body.style.cursor = '';
                document.body.classList.remove('annotation-interacting');

                if (!isMoving) {
                    container.innerHTML = renderAnnotatedHtml(originalRaw);
                    this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
                    return;
                }

                const finalStart = side === 'left' ? newBoundary : oldStart;
                const finalEnd = side === 'right' ? newBoundary : oldEnd;
                const isValid = isValidAnnotationPosition(
                    originalRaw,
                    finalStart,
                    finalEnd,
                    oldStart,
                    oldEnd
                );
                if (!isValid) {
                    container.innerHTML = renderAnnotatedHtml(originalRaw);
                    this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
                    if (hasInvalidTarget) {
                        showToast('Cannot resize into another annotation span', 'error');
                    }
                    return;
                }

                if (finalStart === oldStart && finalEnd === oldEnd) {
                    container.innerHTML = renderAnnotatedHtml(originalRaw);
                    this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
                    return;
                }

                this.applyQuestionAnnotationResize(ctx, oldStart, oldEnd, finalStart, finalEnd);
            };

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        },

        applyQuestionAnnotationResize(ctx, oldStart, oldEnd, newStart, newEnd) {
            if (!this.canEditQuestions) return;
            if (newStart === oldStart && newEnd === oldEnd) {
                showToast('No change in span', 'info');
                return;
            }

            const question = this.docData.questions?.[ctx.questionIdx];
            if (!question) return;

            this.pushUndo();
            const currentText = ctx.field === 'question' ? String(question.question || '') : String(question.answer || '');
            const newText = resizeAnnotation(currentText, oldStart, oldEnd, newStart, newEnd);

            if (newText === currentText) {
                showToast('Failed to resize annotation', 'error');
                return;
            }

            if (ctx.field === 'question') {
                question.question = newText;
            } else {
                question.answer = newText;
            }

            this.refreshEntities();
            this.markDirty();
            showToast(`Annotation span resized in ${ctx.field}`, 'success');
            this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
        },

        applyAnnotationResize(oldStart, oldEnd, newStart, newEnd, entityRef) {
            if (!this.canEditDocumentAnnotations) return;
            if (newStart === oldStart && newEnd === oldEnd) {
                showToast('No change in span', 'info');
                return;
            }
            
            this.pushUndo();
            
            const raw = this.docData.document_to_annotate;
            
            // Use the resizeAnnotation utility function
            const newText = resizeAnnotation(raw, oldStart, oldEnd, newStart, newEnd);
            
            if (newText === raw) {
                showToast('Failed to resize annotation', 'error');
                return;
            }
            
            this.docData.document_to_annotate = newText;
            this.refreshEntities();
            this.markDirty();
            showToast('Annotation span resized', 'success');
        },

        togglePaintMode(entityId) {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            if (this.paintEntity === entityId) {
                this.paintEntity = null;
                this.expandedEntity = null;
            } else {
                this.paintEntity = entityId;
                this.expandedEntity = entityId;
            }
        },

        getNextEntityIndex(type) {
            const group = this.entityGroups[type] || [];
            if (group.length === 0) return 1;
            const maxNum = Math.max(...group.map(e => {
                const parts = e.id.split('_');
                return parseInt(parts[parts.length - 1]) || 0;
            }));
            return maxNum + 1;
        },

        addNewEntity() {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            this.newEntityDialog = { show: true, type: this.entityTypes[0] || 'person' };
        },

        confirmNewEntity() {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            const type = this.newEntityDialog.type;
            const idx = this.getNextEntityIndex(type);
            const newId = `${type}_${idx}`;
            this.newEntityDialog.show = false;
            this.paintEntity = newId;
            showToast(`Created ${newId} — select text to annotate`, 'info');
        },

        // --- Entity hover highlighting ---
        onEntityHover(entityId) {
            // Don't highlight on hover if there's a locked highlight
            if (this.lockedHighlight) return;
            highlightEntity(entityId);
        },
        onEntityLeave() {
            // Don't clear if there's a locked highlight
            if (this.lockedHighlight) return;
            clearEntityHighlight();
        },

        // --- Entity type filtering ---
        toggleTypeHighlight(type) {
            if (this.highlightedType === type) {
                this.highlightedType = null;
                clearTypeHighlight();
            } else {
                this.highlightedType = type;
                highlightEntityType(type);
            }
        },

        // --- Text selection & annotation ---
        handleDocumentMouseDown(event) {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            if (!event || !event.target) return;

            const resizeHandle = event.target.closest('.resize-handle');
            if (resizeHandle) {
                this.startDragResize(event);
                return;
            }

            const deleteBtn = event.target.closest('.ann-delete-btn');
            if (deleteBtn) {
                event.preventDefault();
                event.stopPropagation();
                const annSpan = deleteBtn.closest('.ann');
                if (annSpan) {
                    this.deleteAnnotationInstance(annSpan);
                }
            }
        },

        handleDocumentDoubleClick(event) {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            if (!event || !event.target) return;
            const annSpan = event.target.closest('.ann');
            if (!annSpan) return;
            if (event.target.closest('.resize-handle') || event.target.closest('.ann-delete-btn')) return;

            event.preventDefault();
            event.stopPropagation();

            if (this.agreementClickTimer) {
                clearTimeout(this.agreementClickTimer);
                this.agreementClickTimer = null;
            }
            if (this.agreementWorkspace.active && this.agreementWorkspace.conflictModal.show) {
                this.closeAgreementConflictModal();
            }
            this.editAnnotation(annSpan);
        },

        handleTextSelection(event) {
            if (!this.canEditDocumentAnnotations || this.showEntityReferences) return;
            // Don't show popup if clicking on annotation elements or while resizing.
            if (event.target.closest('.ann') || document.body.style.cursor === 'ew-resize') {
                return;
            }

            const selection = window.getSelection();
            if (!selection || selection.isCollapsed) return;

            const selectedText = selection.toString().trim();
            if (!selectedText) return;

            const container = this.$refs.docText;
            if (!container) return;

            const range = selection.getRangeAt(0);
            const preRange = document.createRange();
            preRange.setStart(container, 0);
            preRange.setEnd(range.startContainer, range.startOffset);

            // Calculate offset excluding delete buttons and resize handles
            const renderedOffset = this.getVisibleTextOffset(preRange);

            const rawStart = this.renderedToRawOffset(renderedOffset);
            const rawEnd = this.renderedToRawOffset(renderedOffset + selectedText.length);

            if (this.paintEntity) {
                const entityId = this.paintEntity;
                const type = entityId.substring(0, entityId.lastIndexOf('_'));
                this.showPopup(event, selectedText, rawStart, rawEnd, type, entityId);
                // Pre-select the entity in the popup
                this.$nextTick(() => {
                    if (this.popup.entityId === entityId && !this.popup.attribute) {
                        // Auto-focus attribute selector
                        setTimeout(() => {
                            const attrSelect = document.querySelector('.annotation-popup select[x-model="popup.attribute"]');
                            if (attrSelect) attrSelect.focus();
                        }, 100);
                    }
                });
            } else {
                this.showPopup(event, selectedText, rawStart, rawEnd);
            }

            selection.removeAllRanges();
        },

        getRenderedPlainText() {
            const raw = this.docData.document_to_annotate || '';
            const annotations = parseAnnotations(raw);
            let plain = '';
            let lastEnd = 0;
            for (const ann of annotations) {
                plain += raw.substring(lastEnd, ann.start);
                plain += ann.text;
                lastEnd = ann.end;
            }
            plain += raw.substring(lastEnd);
            return plain;
        },

        caretRangeFromPoint(clientX, clientY) {
            if (typeof document.caretRangeFromPoint === 'function') {
                return document.caretRangeFromPoint(clientX, clientY);
            }
            if (typeof document.caretPositionFromPoint === 'function') {
                const pos = document.caretPositionFromPoint(clientX, clientY);
                if (!pos) return null;
                const range = document.createRange();
                range.setStart(pos.offsetNode, pos.offset);
                range.collapse(true);
                return range;
            }
            return null;
        },

        /**
         * Calculate visible text offset from a range, excluding delete buttons and resize handles.
         * This fixes the offset bug where Range.toString() includes "×" from delete buttons.
         */
        getVisibleTextOffset(range, rootContainer = null) {
            let container = rootContainer;
            if (!container) {
                // Find the root container
                container = range.commonAncestorContainer;
                if (container.nodeType === Node.TEXT_NODE) {
                    container = container.parentElement;
                }

                // Find the actual container element (should be docText)
                while (container && !container.hasAttribute('x-ref')) {
                    container = container.parentElement;
                }
            }

            if (!container) {
                // Fallback to old method
                return range.toString().length;
            }

            let offset = 0;
            let reachedEnd = false;
            let childIndexTarget = -1;

            // If endContainer is an element node, we need to count up to a specific child
            if (range.endContainer.nodeType === Node.ELEMENT_NODE) {
                childIndexTarget = range.endOffset;
            }

            // Recursively walk the DOM tree
            const walk = (node, childIndex) => {
                if (reachedEnd) return;

                if (node.nodeType === Node.TEXT_NODE) {
                    // Check if this text node should be excluded
                    let parent = node.parentElement;
                    let shouldExclude = false;

                    while (parent && parent !== container) {
                        if (parent.classList && (
                            parent.classList.contains('ann-delete-btn') ||
                            parent.classList.contains('resize-handle')
                        )) {
                            shouldExclude = true;
                            break;
                        }
                        parent = parent.parentElement;
                    }

                    if (shouldExclude) {
                        return; // Skip this text node
                    }

                    // Check if we've reached the end of our range (text node case)
                    if (node === range.endContainer && range.endContainer.nodeType === Node.TEXT_NODE) {
                        offset += range.endOffset;
                        reachedEnd = true;
                        return;
                    }

                    // Add this text node's length
                    offset += node.length;

                } else if (node.nodeType === Node.ELEMENT_NODE) {
                    // Check if this element is the endContainer (element node case)
                    if (node === range.endContainer && range.endContainer.nodeType === Node.ELEMENT_NODE) {
                        // Process only up to childIndexTarget
                        for (let i = 0; i < Math.min(childIndexTarget, node.childNodes.length); i++) {
                            walk(node.childNodes[i], i);
                            if (reachedEnd) return;
                        }
                        reachedEnd = true;
                        return;
                    }

                    // Recursively process all child nodes
                    for (let i = 0; i < node.childNodes.length; i++) {
                        walk(node.childNodes[i], i);
                        if (reachedEnd) return;
                    }
                }
            };

            walk(container, 0);
            return offset;
        },

        renderedToRawOffset(renderedOffset, rawText = null) {
            const raw = rawText === null ? (this.docData.document_to_annotate || '') : String(rawText || '');
            const annotations = parseAnnotations(raw);
            let rawPos = 0;
            let renderedPos = 0;

            for (const ann of annotations) {
                const plainLen = ann.start - rawPos;
                if (renderedOffset <= renderedPos + plainLen) {
                    const out = rawPos + (renderedOffset - renderedPos);
                    return Math.max(0, Math.min(out, raw.length));
                }
                renderedPos += plainLen;
                rawPos = ann.start;

                const annTextLen = ann.text.length;
                if (renderedOffset <= renderedPos + annTextLen) {
                    // Map inside rendered annotation text to the corresponding
                    // position inside "[text; entity.attr]" (i.e., after '[').
                    const innerOffset = renderedOffset - renderedPos;
                    const out = ann.start + 1 + innerOffset;
                    return Math.max(0, Math.min(out, raw.length));
                }
                renderedPos += annTextLen;
                rawPos = ann.end;
            }

            const out = rawPos + (renderedOffset - renderedPos);
            return Math.max(0, Math.min(out, raw.length));
        },

        _repositionAnnotationPopup() {
            if (!this.popup?.show) return;
            const popupEl = this.$refs?.popup;
            if (!popupEl) return;

            const pad = 10;
            const viewportW = window.innerWidth || 0;
            const viewportH = window.innerHeight || 0;
            if (viewportW <= 0 || viewportH <= 0) return;

            const maxHeight = Math.max(220, viewportH - (pad * 2));
            popupEl.style.maxHeight = `${maxHeight}px`;

            const width = popupEl.offsetWidth || 350;
            const height = popupEl.offsetHeight || 400;

            let x = Number(this.popup.x || pad);
            let y = Number(this.popup.y || pad);
            x = Math.max(pad, Math.min(x, viewportW - width - pad));
            y = Math.max(pad, Math.min(y, viewportH - height - pad));

            this.popup.x = Math.round(x);
            this.popup.y = Math.round(y);
        },

        showPopup(event, text, rawStart, rawEnd, entityType, entityId) {
            const sel = window.getSelection();
            const rect = sel && sel.rangeCount ? sel.getRangeAt(0).getBoundingClientRect() : null;
            let x = rect ? rect.left : event.clientX;
            let y = rect ? rect.bottom + 8 : event.clientY;
            
            const popupWidth = 350; // max-width of popup
            const popupHeight = 400; // estimated height
            const sidebarWidth = 220; // width of left sidebar from CSS
            
            // Check if popup would go off bottom of screen
            if (y + popupHeight > window.innerHeight) {
                // Position above the selection instead
                y = (rect ? rect.top : event.clientY) - popupHeight - 10;
            }
            
            // Ensure popup doesn't go off top of screen
            if (y < 10) {
                y = 10;
            }
            
            // Ensure popup doesn't overlap with left sidebar
            // Add a buffer of 30px from sidebar edge
            const minX = sidebarWidth + 30;
            if (x < minX) {
                x = minX;
            }
            
            // Ensure popup doesn't go off right edge of screen
            const maxX = window.innerWidth - popupWidth - 20;
            if (x > maxX) {
                x = maxX;
            }

            this.popup = {
                show: true,
                x,
                y,
                selectedText: text,
                rawStart,
                rawEnd,
                entityType: entityType || '',
                entityId: entityId || '',
                attribute: '',
                relationshipTarget: '',
                editing: false,
                editRef: null,
            };
            this.$nextTick(() => this._repositionAnnotationPopup());
        },

        _resolvePopupEntityId() {
            const selectedId = String(this.popup?.entityId || '').trim();
            if (!selectedId) return '';
            if (selectedId.startsWith('__new_')) {
                const type = selectedId.replace('__new_', '').trim() || String(this.popup?.entityType || 'person').trim() || 'person';
                return `${type}_${this.getNextEntityIndex(type)}`;
            }
            return selectedId;
        },

        selectEntityType(type) {
            this.popup.entityType = type;
            this.popup.entityId = '';
            this.popup.attribute = '';
            this.popup.relationshipTarget = '';
            this.$nextTick(() => this._repositionAnnotationPopup());
        },

        onEntitySelected() {
            const id = this.popup.entityId;
            if (id.startsWith('__new_')) {
                const type = id.replace('__new_', '').trim() || this.popup.entityType || 'person';
                this.popup.entityType = type;
            } else if (id) {
                const underscoreIdx = id.lastIndexOf('_');
                this.popup.entityType = id.substring(0, underscoreIdx);
            }
            if (this.popup.entityType !== 'person' && this.popup.attribute === 'relationship') {
                this.popup.attribute = '';
                this.popup.relationshipTarget = '';
            }
            if (this.popup.attribute === 'relationship') {
                const hasSelected = this.popupRelationshipTargetOptions.some(
                    (ent) => ent.id === this.popup.relationshipTarget
                );
                if (!hasSelected) {
                    this.popup.relationshipTarget = this.popupRelationshipTargetOptions[0]?.id || '';
                }
            }
            this.$nextTick(() => this._repositionAnnotationPopup());
        },

        onAttributeSelected() {
            if (this.popup.attribute !== 'relationship') {
                this.popup.relationshipTarget = '';
                this.$nextTick(() => this._repositionAnnotationPopup());
                return;
            }
            const hasSelected = this.popupRelationshipTargetOptions.some(
                (ent) => ent.id === this.popup.relationshipTarget
            );
            if (!hasSelected) {
                this.popup.relationshipTarget = this.popupRelationshipTargetOptions[0]?.id || '';
            }
            this.$nextTick(() => this._repositionAnnotationPopup());
        },

        parsePopupAttributeRef(ref) {
            const rawRef = String(ref || '').trim();
            const dotIdx = rawRef.indexOf('.');
            if (dotIdx < 0) {
                return { attribute: '', relationshipTarget: '' };
            }

            const rawAttribute = rawRef.substring(dotIdx + 1).trim();
            if (!rawAttribute) {
                return { attribute: '', relationshipTarget: '' };
            }

            if (rawAttribute.startsWith('relationship.')) {
                const target = rawAttribute.substring('relationship.'.length).trim();
                return { attribute: 'relationship', relationshipTarget: target };
            }

            return { attribute: rawAttribute, relationshipTarget: '' };
        },

        applyAnnotation() {
            if (!this.canApplyPopupAnnotation) return;
            if (this.popup.questionIdx !== undefined && !this.canEditQuestions) return;
            if (this.popup.questionIdx === undefined && !this.canEditDocumentAnnotations) return;

            const entityId = this._resolvePopupEntityId();
            if (!entityId) return;

            let attribute = String(this.popup.attribute || '').trim();
            if (this.popupNeedsRelationshipTarget) {
                attribute = `relationship.${String(this.popup.relationshipTarget || '').trim()}`;
            }
            const ref = `${entityId}.${attribute}`;

            this.pushUndo();

            // Check if this is a question/answer annotation
            if (this.popup.questionIdx !== undefined && this.popup.questionField) {
                const question = this.docData.questions[this.popup.questionIdx];
                if (!question) return;

                const field = this.popup.questionField; // 'question' or 'answer'
                const currentText = field === 'question' ? question.question : String(question.answer);

                let annotatedText = '';
                if (this.popup.editing && this.popup.editRef) {
                    // Update existing question/answer annotation
                    let newText = removeAnnotation(currentText, this.popup.rawStart, this.popup.rawEnd);
                    newText = insertAnnotation(newText, this.popup.rawStart, this.popup.rawStart + this.popup.selectedText.length, ref);
                    annotatedText = newText;
                } else {
                    // Apply new annotation to question/answer text
                    annotatedText = insertAnnotation(currentText, this.popup.rawStart, this.popup.rawEnd, ref);
                }

                if (field === 'question') {
                    question.question = annotatedText;
                } else {
                    question.answer = annotatedText;
                }

                showToast(`Annotation ${this.popup.editing ? 'updated' : 'added'} in ${field}`, 'success');
                this.closePopup();
                this.refreshEntities();
                this.markDirty();
                this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
                return;
            }

            // Otherwise, it's a document annotation
            const raw = this.docData.document_to_annotate;

            if (this.popup.editing && this.popup.editRef) {
                // Update existing annotation: remove old and insert new
                const oldStart = this.popup.rawStart;
                const oldEnd = this.popup.rawEnd;
                const text = this.popup.selectedText;
                
                // Remove the old annotation
                let newText = removeAnnotation(raw, oldStart, oldEnd);
                
                // Re-insert at the same position with updated reference
                newText = insertAnnotation(newText, oldStart, oldStart + text.length, ref);
                this.docData.document_to_annotate = newText;
                
                showToast('Annotation updated', 'success');
            } else {
                // New annotation
                this.docData.document_to_annotate = insertAnnotation(raw, this.popup.rawStart, this.popup.rawEnd, ref);
                showToast('Annotation added', 'success');
            }

            this.closePopup();
            this.refreshEntities();
            this.markDirty();
        },

        closePopup() {
            this.popup.show = false;
            this.popup.editing = false;
            this.popup.editRef = null;
            this.popup.relationshipTarget = '';
            this.popup.resizingSpan = false;
            this.popup.questionIdx = undefined;
            this.popup.questionField = null;
        },

        // --- Undo/Redo ---
        pushUndo() {
            this.undoStack.push(this.docData.document_to_annotate);
            this.redoStack = [];
            if (this.undoStack.length > 50) this.undoStack.shift();
        },

        undo() {
            if (this.undoStack.length === 0) return;
            this.redoStack.push(this.docData.document_to_annotate);
            this.docData.document_to_annotate = this.undoStack.pop();
            this.refreshEntities();
            this.markDirty();
        },

        redo() {
            if (this.redoStack.length === 0) return;
            this.undoStack.push(this.docData.document_to_annotate);
            this.docData.document_to_annotate = this.redoStack.pop();
            this.refreshEntities();
            this.markDirty();
        },

        // --- Source view ---
        onSourceViewToggle() {
            if (this.sourceView) {
                this.syncSourceEditorTextFromDoc();
            }
        },

        syncSourceEditorTextFromDoc() {
            const raw = String(this.docData?.document_to_annotate || '');
            if (this.agreementWorkspace.active && this.agreementWorkspace.resolveMode) {
                this.sourceEditorText = String(this._parseAgreementAnnotatedText(raw).plainText || '');
                return;
            }
            this.sourceEditorText = raw;
        },

        _applyAgreementSourcePlainTextChange(nextPlainText) {
            const parsed = this._parseAgreementAnnotatedText(this.docData?.document_to_annotate || '');
            const basePlain = String(parsed.plainText || '');
            const targetPlain = String(nextPlainText || '');
            if (basePlain === targetPlain) return false;

            let plain = basePlain;
            let annotations = (parsed.annotations || []).map((ann) => ({
                start: Number(ann.start || 0),
                end: Math.max(Number(ann.start || 0), Number(ann.end || ann.start || 0)),
                ref: String(ann.ref || '').trim(),
            }));

            const edits = this._computeAgreementTextEdits(basePlain, targetPlain)
                .slice()
                .sort((a, b) => (Number(b.baseStart || 0) - Number(a.baseStart || 0)) || (Number(b.baseEnd || 0) - Number(a.baseEnd || 0)));

            for (const edit of edits) {
                const start = Math.max(0, Math.min(Number(edit.baseStart || 0), plain.length));
                const end = Math.max(start, Math.min(Number(edit.baseEnd || start), plain.length));
                const targetStart = Math.max(0, Math.min(Number(edit.targetStart || 0), targetPlain.length));
                const targetEnd = Math.max(targetStart, Math.min(Number(edit.targetEnd || targetStart), targetPlain.length));
                const replacement = targetPlain.substring(targetStart, targetEnd);
                const delta = replacement.length - (end - start);

                const shifted = [];
                for (const ann of annotations) {
                    const annStart = Number(ann.start || 0);
                    const annEnd = Math.max(annStart, Number(ann.end || annStart));
                    if (annEnd <= start) {
                        shifted.push({ ...ann, start: annStart, end: annEnd });
                        continue;
                    }
                    if (annStart >= end) {
                        shifted.push({ ...ann, start: annStart + delta, end: annEnd + delta });
                        continue;
                    }

                    const leftStart = annStart;
                    const leftEnd = Math.min(annEnd, start);
                    if (leftEnd > leftStart) {
                        shifted.push({ ...ann, start: leftStart, end: leftEnd });
                    }

                    const rightStart = Math.max(annStart, end);
                    const rightEnd = annEnd;
                    if (rightEnd > rightStart) {
                        shifted.push({
                            ...ann,
                            start: rightStart + delta,
                            end: rightEnd + delta,
                        });
                    }
                }

                plain = `${plain.substring(0, start)}${replacement}${plain.substring(end)}`;
                annotations = shifted;
            }

            this.docData.document_to_annotate = this._rebuildAnnotatedTextFromPlain(plain, annotations);
            this.agreementWorkspace.loadedVariant = 'editor';
            return true;
        },

        onSourceChange() {
            if (!this.docData) return;
            if (this.agreementWorkspace.active && this.agreementWorkspace.resolveMode) {
                const changed = this._applyAgreementSourcePlainTextChange(this.sourceEditorText);
                if (!changed) return;
            } else {
                this.docData.document_to_annotate = String(this.sourceEditorText || '');
            }
            this.refreshEntities();
            this.markDirty();
        },

        // --- Questions (read-only by default, click pencil to edit) ---
        handleEntityHover(event) {
            if (this.lockedHighlight) return;
            
            // Find the annotation element (might be target or parent)
            const annElement = event.target.classList.contains('ann') 
                ? event.target 
                : event.target.closest('.ann');
            
            if (!annElement) return;
            
            const entityId = annElement.dataset.entityId;
            console.log('Hover on entity:', entityId, 'element:', annElement);
            if (entityId && window.highlightEntity) {
                window.highlightEntity(entityId, annElement);
            }
        },

        handleEntityLeave(event) {
            console.log('handleEntityLeave, locked:', this.lockedHighlight);
            if (this.lockedHighlight) {
                console.log('Locked highlight active, not clearing');
                return;
            }
            
            // Only clear if we're actually leaving an annotation
            const annElement = event.target.classList.contains('ann') 
                ? event.target 
                : event.target.closest('.ann');
            
            if (!annElement) return;
            
            console.log('Clearing highlight on leave');
            if (window.clearEntityHighlight) {
                window.clearEntityHighlight();
            }
        },

        handleEntityClick(event) {
            console.log('handleEntityClick called', event.target);
            
            // Find the annotation element
            const annElement = event.target.classList.contains('ann') 
                ? event.target 
                : event.target.closest('.ann');
            
            console.log('Found annElement:', annElement);
            
            if (!annElement) return;
            
            // Ignore clicks on interactive children
            if (event.target.classList.contains('resize-handle') || 
                event.target.classList.contains('ann-delete-btn')) {
                console.log('Ignoring click on interactive child');
                return;
            }
            
            const entityId = annElement.dataset.entityId;
            console.log('Entity ID:', entityId, 'Current locked:', this.lockedHighlight);
            
            if (!entityId) return;

            if (
                this.canEditQuestions
                && this.editingQuestion !== null
                && Number.isInteger(this.qaInsertionTarget?.questionIdx)
                && ['question', 'answer'].includes(String(this.qaInsertionTarget?.field || ''))
            ) {
                const entityRef = String(annElement.dataset.ref || '').trim();
                const displayText = this._annotationDisplayText(annElement);
                if (entityRef && displayText) {
                    this.insertEntityAnnotationIntoActiveQuestion(entityRef, displayText);
                }
            }
            
            if (this.lockedHighlight === entityId) {
                console.log('Unlocking highlight');
                this.lockedHighlight = null;
                if (window.clearEntityHighlight) {
                    window.clearEntityHighlight();
                }
            } else {
                console.log('Locking highlight to:', entityId);
                this.lockedHighlight = entityId;
                if (window.highlightEntity) {
                    window.highlightEntity(entityId, annElement);
                }
            }
        },

        toggleQuestionEdit(qIdx) {
            this.editingQuestion = this.editingQuestion === qIdx ? null : qIdx;
            if (this.editingQuestion === null) {
                this.clearQaInsertionTarget();
            }
            // Re-attach listeners when toggling
            this.$nextTick(() => this.attachQuestionAnnotationHoverListeners());
        },

        setQaInsertionTarget(questionIdx, field = 'question') {
            this.qaInsertionTarget = {
                questionIdx: Number.isInteger(questionIdx) ? questionIdx : null,
                field: String(field || 'question').trim().toLowerCase() || 'question',
            };
        },

        clearQaInsertionTarget() {
            this.qaInsertionTarget = { questionIdx: null, field: 'question' };
        },

        insertEntityAnnotationIntoActiveQuestion(entityRef, displayText) {
            const targetIdx = Number(this.qaInsertionTarget?.questionIdx);
            if (!Number.isInteger(targetIdx) || targetIdx < 0) return;
            const question = this.docData?.questions?.[targetIdx];
            const targetField = String(this.qaInsertionTarget?.field || 'question').trim().toLowerCase();
            if (!question || !['question', 'answer'].includes(targetField)) return;

            const safeDisplay = String(displayText || '').trim();
            const safeRef = String(entityRef || '').trim();
            if (!safeDisplay || !safeRef) return;
            const token = targetField === 'answer'
                ? safeRef
                : `[${safeDisplay}; ${safeRef}]`;

            const fieldSelector = targetField === 'answer'
                ? `input[data-answer-input-idx="${targetIdx}"]`
                : `textarea[data-question-input-idx="${targetIdx}"]`;
            const inputEl = this.$root?.querySelector(fieldSelector);
            const currentValue = String(question[targetField] || '');

            if (!inputEl) {
                const separator = currentValue && !/\s$/.test(currentValue) ? ' ' : '';
                question[targetField] = `${currentValue}${separator}${token}`;
                this.markDirty();
                return;
            }

            inputEl.focus();
            const start = Number(inputEl.selectionStart ?? inputEl.value.length);
            const end = Number(inputEl.selectionEnd ?? start);
            const current = currentValue;
            const prefix = current.slice(0, start);
            const suffix = current.slice(end);
            const needsLeadingSpace = prefix.length > 0 && !/\s$/.test(prefix);
            const needsTrailingSpace = suffix.length > 0 && !/^\s/.test(suffix);
            const inserted = `${needsLeadingSpace ? ' ' : ''}${token}${needsTrailingSpace ? ' ' : ''}`;
            question[targetField] = `${prefix}${inserted}${suffix}`;
            this.markDirty();

            this.$nextTick(() => {
                const nextPos = start + inserted.length;
                inputEl.focus();
                inputEl.setSelectionRange(nextPos, nextPos);
            });
            const insertedLabel = targetField === 'answer' ? safeRef : `[value; ref]`;
            showToast(`Inserted ${insertedLabel} into ${targetField}`, 'success', 1400);
        },

        addQuestion() {
            if (!this.canEditQuestions) return;
            if (!this.docData) return;
            if (!this.docData.questions) this.docData.questions = [];
            const usedIds = new Set((this.docData.questions || []).map(q => String(q?.question_id || '')));
            const userTag = (!isPowerUser && currentUsername)
                ? currentUsername.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
                : '';
            const idPrefix = userTag ? `${this.doc_id}_${userTag}_w` : `${this.doc_id}_w`;
            let nextIdx = 1;
            while (usedIds.has(`${idPrefix}${nextIdx}`)) {
                nextIdx += 1;
            }
            const newQId = `${idPrefix}${nextIdx}`;
            this.docData.questions.unshift({
                question_id: newQId,
                question: '',
                question_type: 'extractive',
                answer: '',
                answer_type: 'variant',
                reasoning_chain: [],
                reasoning_chain_text: '',
            });
            this.editingQuestion = 0;
            this.markDirty();
        },

        onQuestionAnswerTypeChanged(question) {
            if (!question || typeof question !== 'object') return;
            const answerType = this._normalizeAnswerType(question.answer_type, question.is_answer_invariant);
            question.answer_type = answerType;
            if (answerType === 'refusal') {
                question.answer = refusalAnswerLiteral;
            }
            this.markDirty();
        },

        onQuestionReasoningChainChanged(question) {
            if (!question || typeof question !== 'object') return;
            question.reasoning_chain = this._normalizeReasoningChain(
                question.reasoning_chain,
                question.reasoning_chain_text
            );
            this.markDirty();
        },

        deleteQuestion(idx) {
            if (!this.canEditQuestions) return;
            if (!confirm('Delete this question?')) return;
            const removedQuestion = this.docData?.questions?.[idx] || null;
            const removedKey = removedQuestion
                ? this._questionConflictKeyFromQuestion(removedQuestion, idx)
                : '';
            this.docData.questions.splice(idx, 1);
            if (this.qaInsertionTarget.questionIdx === idx) {
                this.clearQaInsertionTarget();
            } else if (Number.isInteger(this.qaInsertionTarget.questionIdx) && this.qaInsertionTarget.questionIdx > idx) {
                this.qaInsertionTarget.questionIdx -= 1;
            }
            if (this.editingQuestion === idx) {
                this.editingQuestion = null;
            } else if (typeof this.editingQuestion === 'number' && this.editingQuestion > idx) {
                this.editingQuestion -= 1;
            }
            if (this.agreementWorkspace.active && removedKey) {
                const conflict = this._questionConflictForKey(removedKey, { includeResolved: true });
                const conflictId = String(conflict?.id || '');
                if (conflictId) {
                    this.agreementWorkspace.structured.decisions[conflictId] = 'manual';
                    this.agreementWorkspace.structured.manualResolutions[conflictId] = '__none__';
                }
            }
            this.markDirty();
        },

        handleQuestionTextSelection(event, questionIdx) {
            if (!this.canEditQuestions) return;
            const selection = window.getSelection();
            if (!selection || selection.isCollapsed) return;

            const selectedText = selection.toString().trim();
            if (!selectedText) return;

            const question = this.docData.questions[questionIdx];
            if (!question) return;

            // Calculate offsets within the question text
            const container = event.currentTarget;
            const range = selection.getRangeAt(0);
            const preRange = document.createRange();
            preRange.setStart(container, 0);
            preRange.setEnd(range.startContainer, range.startOffset);
            const renderedOffset = this.getVisibleTextOffset(preRange, container);
            const currentRaw = String(question.question || '');
            const rawStart = this.renderedToRawOffset(renderedOffset, currentRaw);
            const rawEnd = this.renderedToRawOffset(renderedOffset + selectedText.length, currentRaw);

            // Show popup for annotation
            this.showQuestionAnnotationPopup(event, selectedText, rawStart, rawEnd, questionIdx, 'question');
            selection.removeAllRanges();
        },

        handleAnswerTextSelection(event, questionIdx) {
            if (!this.canEditQuestions) return;
            const selection = window.getSelection();
            if (!selection || selection.isCollapsed) return;

            const selectedText = selection.toString().trim();
            if (!selectedText) return;

            const question = this.docData.questions[questionIdx];
            if (!question) return;

            const container = event.currentTarget;
            const range = selection.getRangeAt(0);
            const preRange = document.createRange();
            preRange.setStart(container, 0);
            preRange.setEnd(range.startContainer, range.startOffset);
            const renderedOffset = this.getVisibleTextOffset(preRange, container);
            const currentRaw = String(question.answer || '');
            const rawStart = this.renderedToRawOffset(renderedOffset, currentRaw);
            const rawEnd = this.renderedToRawOffset(renderedOffset + selectedText.length, currentRaw);

            this.showQuestionAnnotationPopup(event, selectedText, rawStart, rawEnd, questionIdx, 'answer');
            selection.removeAllRanges();
        },

        showQuestionAnnotationPopup(event, text, rawStart, rawEnd, questionIdx, field) {
            if (!this.canEditQuestions) return;
            const sel = window.getSelection();
            const rect = sel && sel.rangeCount ? sel.getRangeAt(0).getBoundingClientRect() : null;
            let x = rect ? rect.left : event.clientX;
            let y = rect ? rect.bottom + 8 : event.clientY;

            const popupWidth = 350;
            const popupHeight = 400;
            const sidebarWidth = 220;

            // Smart positioning
            if (y + popupHeight > window.innerHeight) {
                y = (rect ? rect.top : event.clientY) - popupHeight - 10;
            }
            if (y < 10) y = 10;

            const minX = sidebarWidth + 30;
            if (x < minX) x = minX;

            const maxX = window.innerWidth - popupWidth - 20;
            if (x > maxX) x = maxX;

            // Store question context
            this.popup = {
                show: true,
                x,
                y,
                selectedText: text,
                rawStart,
                rawEnd,
                entityType: '',
                entityId: '',
                attribute: '',
                relationshipTarget: '',
                editing: false,
                editRef: null,
                resizingSpan: false,
                questionIdx,
                questionField: field
            };
        },

        // --- Rules ---
        addRule() {
            if (!this.canEditRules) return;
            if (!this.docData.rules) this.docData.rules = [];
            this.docData.rules.push('');
            this.editingRule = this.docData.rules.length - 1;
            this.markDirty();
            this.$nextTick(() => {
                this.attachRuleReferenceHoverListeners();
            });
        },

        deleteRule(idx) {
            if (!this.canEditRules) return;
            this.docData.rules.splice(idx, 1);
            if (this.editingRule === idx) {
                this.editingRule = null;
            } else if (typeof this.editingRule === 'number' && this.editingRule > idx) {
                this.editingRule -= 1;
            }
            this.markDirty();
            this.$nextTick(() => {
                this.attachRuleReferenceHoverListeners();
            });
        },

        // --- Rule Composer Methods ---
        updateRuleInputContent() {
            const input = this.$refs.ruleInput;
            this.ruleInputContent = input ? input.textContent.trim() : '';
        },

        insertRuleChip(ref, type) {
            const input = this.$refs.ruleInput;
            if (!input) return;
            
            const chip = document.createElement('span');
            chip.className = 'rule-inline-chip rule-chip-' + type;
            chip.textContent = ref;
            chip.contentEditable = 'false';
            chip.dataset.value = ref;

            input.focus();
            const sel = window.getSelection();
            if (sel.rangeCount) {
                const range = sel.getRangeAt(0);
                range.deleteContents();
                range.insertNode(chip);
                range.setStartAfter(chip);
                range.setEndAfter(chip);
                sel.removeAllRanges();
                sel.addRange(range);
            } else {
                input.appendChild(chip);
            }
            const space = document.createTextNode(' ');
            chip.after(space);
            const r = document.createRange();
            r.setStartAfter(space);
            r.setEndAfter(space);
            sel.removeAllRanges();
            sel.addRange(r);
            this.updateRuleInputContent();
        },

        insertRuleText(text) {
            const input = this.$refs.ruleInput;
            if (!input) return;
            
            input.focus();
            const sel = window.getSelection();
            if (sel.rangeCount) {
                const range = sel.getRangeAt(0);
                range.deleteContents();
                const node = document.createTextNode(text);
                range.insertNode(node);
                range.setStartAfter(node);
                range.setEndAfter(node);
                sel.removeAllRanges();
                sel.addRange(range);
            } else {
                input.appendChild(document.createTextNode(text));
            }
            this.updateRuleInputContent();
        },

        ruleInputHasContent() {
            return this.ruleInputContent.length > 0;
        },

        getRuleText() {
            const input = this.$refs.ruleInput;
            if (!input) return '';
            let text = '';
            for (const node of input.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    text += node.textContent;
                } else if (node.dataset && node.dataset.value) {
                    text += node.dataset.value;
                } else {
                    text += node.textContent;
                }
            }
            return text.trim();
        },

        commitRule() {
            if (!this.canEditRules) return;
            const rule = this.getRuleText();
            if (!rule) return;
            if (!this.docData.rules) this.docData.rules = [];
            this.docData.rules.push(rule);
            const input = this.$refs.ruleInput;
            if (input) input.innerHTML = '';
            this.ruleInputContent = '';
            this.markDirty();
            this.$nextTick(() => {
                this.attachRuleReferenceHoverListeners();
            });
        },

        // --- Save, Validate, Finish ---
        markDirty() {
            this.dirty = true;
            if (this.agreementWorkspace.active) {
                this._recomputeAgreementConflictStates();
                this._recomputeAgreementStructuredConflictStates();
            }
        },

        async save() {
            if (!this.docData || this.saving) return;
            if (!this.canSaveWorkspace) {
                showToast('This review workspace is read-only outside agreement mode', 'info');
                return;
            }
            this.saving = true;
            try {
                if (this.reviewTarget === 'questions') {
                    this._prepareQuestionPayloadForPersistence();
                } else {
                    this.docData.num_questions = this.docData.questions?.length || 0;
                }
                if (this.reviewTarget && this.agreementWorkspace.active && isPowerUser) {
                    await API.setAdminReviewFinalFromEditor(this.reviewTarget, this.theme, this.doc_id, this.docData, 'admin_draft');
                    this.agreementWorkspace.versions.final.available = true;
                    this.agreementWorkspace.versions.final.document_to_annotate = String(this.docData.document_to_annotate || '');
                    this.agreementWorkspace.versions.final.editable_document = this._normalizeEditableDocument(this.docData);
                    this.agreementWorkspace.versions.final.username = 'admin';
                    this.currentStatus = 'in_progress';
                } else if (this.reviewTarget && isPowerUser) {
                    await API.saveDocument(this.theme, this.doc_id, this.docData, {
                        reviewTarget: this.reviewTarget,
                    });
                } else if (this.reviewTarget) {
                    await API.saveReviewDocument(this.reviewTarget, this.theme, this.doc_id, this.docData);
                    this.currentStatus = 'in_progress';
                } else {
                    await API.saveDocument(this.theme, this.doc_id, this.docData);
                }
                this.dirty = false;
                showToast('Document saved', 'success');
                if (!(this.reviewTarget && this.agreementWorkspace.active && isPowerUser)) {
                    this.reloadHistory();
                }
            } catch (e) {
                showToast('Save failed: ' + e.message, 'error');
            } finally {
                this.saving = false;
            }
        },

        async markAgreementFinished() {
            if (!this.agreementWorkspace.active) return;
            if (!this.agreementWorkspace.runId) {
                showToast('No active agreement run found for this document', 'error');
                return;
            }
            if (this.agreementWorkspace.finalizing) return;

            if (this.dirty) {
                await this.save();
                if (this.dirty) return;
            }

            this._recomputeAgreementConflictStates();
            this._recomputeAgreementStructuredConflictStates();
            if (this.agreementHasUnresolvedConflicts()) {
                showToast('Resolve all remaining conflicts before finishing agreement', 'warning');
                return;
            }
            if (this.reviewTarget === 'questions') {
                const coverage = this.qaCoverageValidationForSubmit();
                if (!coverage.valid) {
                    const validationMessages = [...coverage.errors];
                    if (coverage.missingRows.length) {
                        const missingLabels = coverage.missingRows
                            .map((row) => `${this.qaCoverageQuestionTypeLabel(row.question_type)} + ${this.qaCoverageAnswerTypeLabel(row.answer_type)}`)
                            .join(', ');
                        validationMessages.push(
                            `Missing QA coverage: ${missingLabels}. Add replacement questions or an ignore justification before finishing agreement.`
                        );
                    }
                    this.panelState.questions = true;
                    showToast(validationMessages.join(' '), 'warning', 9000);
                    return;
                }
            }

            if (!confirm('Mark this document as Agreement finished?')) return;

            this.agreementWorkspace.finalizing = true;
            try {
                this.docData = this._normalizeEditableDocument(this.docData || {});
                const resolution = (this.reviewTarget && isPowerUser)
                    ? await (async () => {
                        await API.setAdminReviewFinalFromEditor(this.reviewTarget, this.theme, this.doc_id, this.docData, 'admin_agreement');
                        return API.resolveReviewAgreementPacket(this.reviewTarget, this.agreementWorkspace.runId, this.theme, this.doc_id, null, 'agreement');
                    })()
                    : await (async () => {
                        await API.setAdminFinalFromEditor(this.theme, this.doc_id, this.docData, 'admin_agreement');
                        return API.resolveAgreementPacket(this.agreementWorkspace.runId, this.theme, this.doc_id);
                    })();

                this.agreementWorkspace.status = String(resolution?.agreement_status || 'resolved');
                this.agreementWorkspace.loadedVariant = 'final';
                this.agreementWorkspace.versions.final.available = true;
                this.agreementWorkspace.versions.final.document_to_annotate = String(this.docData.document_to_annotate || '');
                this.agreementWorkspace.versions.final.editable_document = this._normalizeEditableDocument(this.docData);
                this.agreementWorkspace.versions.final.username = 'admin';
                const waitingAcceptance = Boolean(resolution?.awaiting_reviewer_acceptance);
                this.agreementWorkspace.awaitingReviewerAcceptance = waitingAcceptance;
                this.currentStatus = waitingAcceptance ? 'in_progress' : 'completed';

                if (waitingAcceptance) {
                    showToast('Agreement saved. Waiting for both annotators to accept final annotations.', 'success');
                } else {
                    showToast('Agreement finished and final annotation saved', 'success');
                }
                if (!(this.reviewTarget && isPowerUser)) {
                    await this.reloadHistory();
                }
            } catch (e) {
                showToast('Failed to finish agreement: ' + e.message, 'error');
            } finally {
                this.agreementWorkspace.finalizing = false;
            }
        },

        async markAgreementCompletedDirectly() {
            if (!this.agreementWorkspace.active) return;
            if (!this.agreementWorkspace.runId) {
                showToast('No active agreement run found for this document', 'error');
                return;
            }
            if (this.agreementWorkspace.finalizing) return;

            if (this.dirty) {
                await this.save();
                if (this.dirty) return;
            }

            this._recomputeAgreementConflictStates();
            this._recomputeAgreementStructuredConflictStates();
            if (this.agreementHasUnresolvedConflicts()) {
                showToast('Resolve all remaining conflicts before completing directly', 'warning');
                return;
            }
            if (this.reviewTarget === 'questions') {
                const coverage = this.qaCoverageValidationForSubmit();
                if (!coverage.valid) {
                    const validationMessages = [...coverage.errors];
                    if (coverage.missingRows.length) {
                        const missingLabels = coverage.missingRows
                            .map((row) => `${this.qaCoverageQuestionTypeLabel(row.question_type)} + ${this.qaCoverageAnswerTypeLabel(row.answer_type)}`)
                            .join(', ');
                        validationMessages.push(
                            `Missing QA coverage: ${missingLabels}. Add replacement questions or an ignore justification before completing directly.`
                        );
                    }
                    this.panelState.questions = true;
                    showToast(validationMessages.join(' '), 'warning', 9000);
                    return;
                }
            }

            if (!confirm('Mark this document as completed without sending it back to reviewers?')) return;

            this.agreementWorkspace.finalizing = true;
            try {
                this.docData = this._normalizeEditableDocument(this.docData || {});
                const resolution = (this.reviewTarget && isPowerUser)
                    ? await (async () => {
                        await API.setAdminReviewFinalFromEditor(this.reviewTarget, this.theme, this.doc_id, this.docData, 'admin_completed');
                        return API.resolveReviewAgreementPacket(this.reviewTarget, this.agreementWorkspace.runId, this.theme, this.doc_id, null, 'complete');
                    })()
                    : await (async () => {
                        await API.setAdminFinalFromEditor(this.theme, this.doc_id, this.docData, 'admin_completed');
                        return API.resolveAgreementPacket(this.agreementWorkspace.runId, this.theme, this.doc_id);
                    })();

                this.agreementWorkspace.status = String(resolution?.agreement_status || 'resolved');
                this.agreementWorkspace.awaitingReviewerAcceptance = false;
                this.agreementWorkspace.loadedVariant = 'final';
                this.agreementWorkspace.versions.final.available = true;
                this.agreementWorkspace.versions.final.document_to_annotate = String(this.docData.document_to_annotate || '');
                this.agreementWorkspace.versions.final.editable_document = this._normalizeEditableDocument(this.docData);
                this.agreementWorkspace.versions.final.username = 'admin';
                this.currentStatus = 'completed';
                showToast('Agreement completed without reviewer validation', 'success');
            } catch (e) {
                showToast('Failed to complete agreement: ' + e.message, 'error');
            } finally {
                this.agreementWorkspace.finalizing = false;
            }
        },

        async validate() {
            try {
                const result = await API.validateDocument(this.theme, this.doc_id);
                if (result.valid) {
                    showToast('All annotations valid', 'success');
                } else {
                    showToast(`${result.errors.length} validation error(s)`, 'error', 5000);
                    console.error('Validation errors:', result.errors);
                }
            } catch (e) {
                showToast('Validation failed: ' + e.message, 'error');
            }
        },

        async finish() {
            if (!this.canFinishWorkspace) {
                showToast('This review workspace is read-only outside agreement mode', 'info');
                return;
            }
            if (this.dirty) {
                showToast('Save your changes before finishing', 'warning');
                return;
            }
            if (this.reviewTarget === 'questions') {
                const coverage = this.qaCoverageValidationForSubmit();
                if (!coverage.valid) {
                    const validationMessages = [...coverage.errors];
                    if (coverage.missingRows.length) {
                        const missingLabels = coverage.missingRows
                            .map((row) => `${this.qaCoverageQuestionTypeLabel(row.question_type)} + ${this.qaCoverageAnswerTypeLabel(row.answer_type)}`)
                            .join(', ');
                        validationMessages.push(
                            `Missing QA coverage: ${missingLabels}. Add questions or add an ignore justification for each missing combination.`
                        );
                    }
                    this.panelState.questions = true;
                    showToast(
                        validationMessages.join(' '),
                        'warning',
                        9000
                    );
                    return;
                }
            }
            try {
                if (this.reviewTarget) {
                    if (isPowerUser) {
                        await API.completeAdminReviewFromEditor(
                            this.reviewTarget,
                            this.theme,
                            this.doc_id,
                            this.docData,
                            'admin_completed'
                        );
                        this.currentStatus = 'completed';
                        showToast(
                            this.reviewTarget === 'rules' ? 'Rules review marked as finished' : 'Questions review marked as finished',
                            'success'
                        );
                        await this.reloadHistory();
                    } else {
                        await API.finishReviewDocument(this.reviewTarget, this.theme, this.doc_id);
                        this.currentStatus = 'completed';
                        showToast(
                            this.reviewTarget === 'rules' ? 'Rules review submitted' : 'Questions review submitted',
                            'success'
                        );
                        setTimeout(() => {
                            window.location.href = '/';
                        }, 500);
                    }
                    return;
                }
                await API.finishDocument(this.theme, this.doc_id);
                if (isPowerUser) {
                    this.currentStatus = 'completed';
                    showToast('Document marked as finished', 'success');
                } else {
                    this.currentStatus = 'in_progress';
                    showToast('Annotations submitted', 'success');
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 500);
                }
                this.reloadHistory();
            } catch (e) {
                showToast('Finish failed: ' + e.message, 'error');
            }
        },

        async submitAnnotations() {
            let confirmationMessage = 'Submit your annotations for this document?';
            if (this.reviewTarget === 'rules') {
                confirmationMessage = 'Submit your rules review for this document?';
            } else if (this.reviewTarget === 'questions') {
                confirmationMessage = 'Submit your questions review for this document?';
            }
            if (!confirm(confirmationMessage)) return;
            await this.finish();
        },

        async unfinish() {
            if (!confirm('Mark this document as incomplete?')) return;
            try {
                // Change status back to in_progress
                await API.unvalidateStatus(this.theme, this.doc_id);
                this.currentStatus = 'in_progress';
                showToast('Document marked as incomplete', 'success');
                this.reloadHistory();
            } catch (e) {
                showToast('Unfinish failed: ' + e.message, 'error');
            }
        },

        async unvalidate() {
            if (!confirm('Reopen this document for editing?')) return;
            try {
                await API.unvalidateStatus(this.theme, this.doc_id);
                this.currentStatus = 'in_progress';
                showToast('Document reopened', 'success');
                this.reloadHistory();
            } catch (e) {
                showToast('Reopen failed: ' + e.message, 'error');
            }
        },

        async reloadHistory() {
            try {
                if (this.reviewTarget && this.agreementWorkspace.active && isPowerUser) {
                    return;
                }
                if (this.reviewTarget && isPowerUser) {
                    const [doc, histData] = await Promise.all([
                        API.loadDocument(this.theme, this.doc_id, { reviewTarget: this.reviewTarget }),
                        API.getDocumentHistory(this.theme, this.doc_id, { reviewType: this.reviewTarget }),
                    ]);
                    this.docData = this._normalizeEditableDocument(doc);
                    this.historyEntries = histData.entries || [];
                    this.annotationVersions = histData.annotation_versions || [];
                    this.currentStatus = this.getCurrentReviewStatus(doc);
                    this.lastEditor = histData.last_editor || null;
                    return;
                }
                if (this.reviewTarget) {
                    const doc = await API.loadReviewDocument(this.reviewTarget, this.theme, this.doc_id);
                    this.docData = this._normalizeEditableDocument(doc);
                    this.currentStatus = this.getCurrentReviewStatus(doc);
                    return;
                }
                const histData = await API.getDocumentHistory(this.theme, this.doc_id);
                this.historyEntries = histData.entries || [];
                this.annotationVersions = histData.annotation_versions || [];
                this.currentStatus = histData.current_status || 'draft';
                this.lastEditor = histData.last_editor || null;
            } catch (e) { /* ignore */ }
        },

        async viewHistorySnapshot(historyId) {
            try {
                const snapshot = await API.getHistorySnapshot(this.theme, this.doc_id, historyId);
                this.historySnapshot = snapshot;
                this.showHistoryModal = true;
            } catch (e) {
                showToast('Failed to load snapshot: ' + e.message, 'error');
            }
        },

        async viewAnnotationVersion(versionKey) {
            try {
                const snapshot = await API.getAnnotationVersion(this.theme, this.doc_id, versionKey, {
                    reviewType: isPowerUser ? this.reviewTarget : null,
                });
                this.historySnapshot = snapshot;
                this.showHistoryModal = true;
            } catch (e) {
                showToast('Failed to load version: ' + e.message, 'error');
            }
        },

        toggleVersionDetails(versionKey) {
            this.openVersionDetails[versionKey] = !this.openVersionDetails[versionKey];
        },

        isVersionDetailsOpen(versionKey) {
            return !!this.openVersionDetails[versionKey];
        },

        closeHistoryModal() {
            this.showHistoryModal = false;
            this.historySnapshot = null;
        },

        async restoreFromHistory() {
            if (!this.historySnapshot) return;
            if (!this.canSaveWorkspace) {
                showToast('This workspace is read-only', 'info');
                return;
            }
            if (this.historySnapshot.content_format === 'markdown') {
                showToast('This version is read-only and cannot be restored directly', 'warning');
                return;
            }
            if (!confirm('Restore this version? Current changes will be overwritten.')) return;
            
            try {
                this.pushUndo();
                
                // Restore document text
                this.docData.document_to_annotate = this.historySnapshot.document;
                
                // Restore questions if available
                if (this.historySnapshot.questions) {
                    this.docData.questions = this._normalizeEditableDocument({
                        questions: this.historySnapshot.questions,
                    }).questions;
                }
                
                // Restore rules if available
                if (this.historySnapshot.rules) {
                    this.docData.rules = this.historySnapshot.rules;
                }
                
                this.refreshEntities();
                this.markDirty();
                this.closeHistoryModal();
                showToast('Version restored! Click Save to keep changes.', 'success');
            } catch (e) {
                showToast('Failed to restore version: ' + e.message, 'error');
            }
        },

        getSummaryKeywords(entry) {
            // Parse details_json if available
            if (!entry.details_json) return '';
            try {
                const details = JSON.parse(entry.details_json);
                const parts = [];
                if (details.questions_count !== undefined) parts.push(`${details.questions_count} Q`);
                if (details.rules_count !== undefined) parts.push(`${details.rules_count} R`);
                return parts.join(', ');
            } catch {
                return '';
            }
        },

        // --- Keyboard shortcuts ---
        handleGlobalKeydown(event) {
            const key = event.key;
            const ctrl = event.ctrlKey || event.metaKey;
            const shift = event.shiftKey;
            const inInput = ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName);
            const inEditable = !!event.target?.closest?.('input, textarea, select, [contenteditable="true"]');

            if (ctrl && key === 's') { event.preventDefault(); this.save(); return; }
            if (ctrl && !shift && key === 'z' && !inEditable) { event.preventDefault(); this.undo(); return; }
            if (ctrl && shift && key === 'z' && !inEditable) { event.preventDefault(); this.redo(); return; }

            if (key === '?' && !inEditable && !this.popup.show) {
                event.preventDefault();
                this.showShortcuts = !this.showShortcuts;
                return;
            }

            // Popup shortcuts
            if (this.popup.show && !inEditable) {
                const typeMap = {
                    '1': 'person',
                    '2': 'place',
                    '3': 'event',
                    '4': 'military_org',
                    '5': 'entreprise_org',
                    '6': 'ngo',
                    '7': 'government_org',
                    '8': 'educational_org',
                    '9': 'media_org',
                    'q': 'temporal',
                    'w': 'number',
                    'e': 'award',
                    'r': 'legal',
                    't': 'product',
                };
                const popupShortcut = String(key || '').toLowerCase();
                if (typeMap[popupShortcut] && (this.entityTypes || []).includes(typeMap[popupShortcut])) {
                    event.preventDefault();
                    this.selectEntityType(typeMap[popupShortcut]);
                    return;
                }
                if (key === 'Enter') { event.preventDefault(); this.applyAnnotation(); return; }
            }

            // Agreement resolution shortcuts
            if (
                this.agreementWorkspace.active
                && this.agreementWorkspace.resolveMode
                && !this.showEntityReferences
                && !inEditable
                && !this.popup.show
            ) {
                if (key === 'j' || key === 'J') {
                    event.preventDefault();
                    this.selectAdjacentAgreementConflict(1, true);
                    return;
                }
                if (key === 'k' || key === 'K') {
                    event.preventDefault();
                    this.selectAdjacentAgreementConflict(-1, true);
                    return;
                }
                if (key === '1') {
                    event.preventDefault();
                    const selected = this.getAgreementSelectedConflict();
                    if (selected) this.applyAgreementConflictChoice(selected.id, 'a');
                    return;
                }
                if (key === '2') {
                    event.preventDefault();
                    const selected = this.getAgreementSelectedConflict();
                    if (selected) this.applyAgreementConflictChoice(selected.id, 'b');
                    return;
                }
                if (key === 'm' || key === 'M') {
                    event.preventDefault();
                    this.openSelectedConflictInEditor();
                    return;
                }
                if (key === 'u' || key === 'U') {
                    event.preventDefault();
                    this.useCurrentForSelectedConflict();
                    return;
                }
            }

            // A: annotate selection
            if (key === 'a' && !inEditable && !this.popup.show && this.canEditDocumentAnnotations && !this.showEntityReferences) {
                const selection = window.getSelection();
                if (selection && !selection.isCollapsed) {
                    event.preventDefault();
                    this.handleTextSelection(event);
                }
                return;
            }

            // D: delete annotation under cursor
            if (key === 'd' && !inEditable && !this.popup.show && this.canEditDocumentAnnotations && !this.showEntityReferences) {
                const hoveredAnn = document.querySelector('.ann:hover');
                if (hoveredAnn) {
                    event.preventDefault();
                    const start = parseInt(hoveredAnn.dataset.start);
                    const end = parseInt(hoveredAnn.dataset.end);
                    this.pushUndo();
                    this.docData.document_to_annotate = removeAnnotation(this.docData.document_to_annotate, start, end);
                    this.refreshEntities();
                    this.markDirty();
                }
                return;
            }

            // E: edit annotation under cursor
            if (key === 'e' && !inEditable && !this.popup.show && this.canEditDocumentAnnotations && !this.showEntityReferences) {
                const hoveredAnn = document.querySelector('.ann:hover');
                if (hoveredAnn) {
                    event.preventDefault();
                    const start = parseInt(hoveredAnn.dataset.start);
                    const end = parseInt(hoveredAnn.dataset.end);
                    const annotations = parseAnnotations(this.docData.document_to_annotate);
                    const ann = annotations.find(a => a.start === start);
                    if (ann) {
                        const parsedAttr = this.parsePopupAttributeRef(ann.ref || '');
                        this.popup = {
                            show: true,
                            x: hoveredAnn.getBoundingClientRect().left,
                            y: hoveredAnn.getBoundingClientRect().bottom + 8,
                            selectedText: ann.text,
                            rawStart: start,
                            rawEnd: end,
                            entityType: ann.entityType,
                            entityId: ann.entityId,
                            attribute: parsedAttr.attribute,
                            relationshipTarget: parsedAttr.relationshipTarget,
                            editing: true,
                            editRef: ann.ref,
                        };
                    }
                }
                return;
            }

            if (key === 'Escape') {
                if (this.popup.show) { this.closePopup(); return; }
                if (this.showShortcuts) { this.showShortcuts = false; return; }
                if (this.paintEntity) { this.paintEntity = null; return; }
                if (this.highlightedType) { this.highlightedType = null; clearTypeHighlight(); return; }
                if (this.lockedHighlight) { this.lockedHighlight = null; clearEntityHighlight(); return; }
            }
        },

        // Format timestamp for display
        formatTime(ts) {
            if (!ts) return '';
            try {
                const d = new Date(ts + 'Z');
                return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
            } catch { return ts; }
        },

        async _loadGlobalDocumentIndex() {
            this.navigationLoading = true;
            try {
                if (this.referenceMode) {
                    this.allDocuments = [{
                        theme: this.theme,
                        theme_label: this.docData?.document_theme || this.theme,
                        doc_id: this.doc_id,
                    }];
                    this.currentDocGlobalIndex = 0;
                    return;
                }
                if (!isPowerUser) {
                    const queue = await API.getMyQueue().catch(() => ({ current_tasks: [] }));
                    const flattened = (queue.current_tasks || []).map((task) => ({
                        theme: task.theme,
                        theme_label: task.theme,
                        doc_id: task.doc_id,
                    }));
                    let currentIdx = flattened.findIndex(
                        (d) => d.theme === this.theme && d.doc_id === this.doc_id
                    );
                    if (currentIdx === -1) {
                        flattened.unshift({
                            theme: this.theme,
                            theme_label: this.docData?.document_theme || this.theme,
                            doc_id: this.doc_id,
                        });
                        currentIdx = 0;
                    }
                    this.allDocuments = flattened;
                    this.currentDocGlobalIndex = currentIdx;
                    return;
                }

                const themes = await API.listThemes();
                const perTheme = await Promise.all(
                    (themes || []).map(async (t) => {
                        const docs = await API.listThemeDocuments(t.theme_id);
                        return {
                            theme_id: t.theme_id,
                            theme_label: t.theme_label || t.theme_id,
                            docs: docs || [],
                        };
                    })
                );

                const flattened = [];
                for (const group of perTheme) {
                    const docs = [...group.docs].sort((a, b) =>
                        String(a.doc_id || '').localeCompare(String(b.doc_id || ''), undefined, { numeric: true })
                    );
                    for (const d of docs) {
                        flattened.push({
                            theme: group.theme_id,
                            theme_label: group.theme_label,
                            doc_id: d.doc_id,
                        });
                    }
                }

                let currentIdx = flattened.findIndex(
                    (d) => d.theme === this.theme && d.doc_id === this.doc_id
                );

                if (currentIdx === -1) {
                    flattened.push({
                        theme: this.theme,
                        theme_label: this.docData?.document_theme || this.theme,
                        doc_id: this.doc_id,
                    });
                    currentIdx = flattened.length - 1;
                }

                this.allDocuments = flattened;
                this.currentDocGlobalIndex = currentIdx;
            } catch (e) {
                console.error('Failed to load global document index:', e);
                this.allDocuments = [];
                this.currentDocGlobalIndex = -1;
            } finally {
                this.navigationLoading = false;
            }
        },

        _confirmNavigateAwayIfDirty() {
            if (!this.dirty) return true;
            return confirm('You have unsaved changes. Continue without saving?');
        },

        _editorRouteHref(theme, docId) {
            const base = `/editor/${encodeURIComponent(String(theme || ''))}/${encodeURIComponent(String(docId || ''))}`;
            const params = new URLSearchParams();
            if (this.reviewTarget === 'rules' || this.reviewTarget === 'questions') {
                params.set('review_target', this.reviewTarget);
            }
            if (this.agreementWorkspace?.active) {
                params.set('agreement_mode', '1');
                const variant = String(this.agreementWorkspace?.contestVariant || '').trim().toLowerCase();
                if (variant === 'reviewer_a' || variant === 'reviewer_b') {
                    params.set('contest_variant', variant);
                }
            }
            const suffix = params.toString();
            return suffix ? `${base}?${suffix}` : base;
        },

        navigateToPreviousDocument() {
            if (!this.hasPrevDocument) return;
            if (!this._confirmNavigateAwayIfDirty()) return;
            const prev = this.allDocuments[this.currentDocGlobalIndex - 1];
            window.location.href = this._editorRouteHref(prev.theme, prev.doc_id);
        },

        navigateToNextDocument() {
            if (!this.hasNextDocument) return;
            if (!this._confirmNavigateAwayIfDirty()) return;
            const next = this.allDocuments[this.currentDocGlobalIndex + 1];
            window.location.href = this._editorRouteHref(next.theme, next.doc_id);
        },

        // --- Groq Playground ---
        groqPlaygroundStorageKey() {
            return `groq-playground:${this.theme}:${this.doc_id}`;
        },

        _stripGroqPlaygroundAnnotations(text) {
            return String(text || '').replace(/\[([^\]]+);\s*[^\]]+\]/g, '$1');
        },

        _nextGroqQuestionId(existing = []) {
            const used = new Set((existing || []).map((item) => String(item?.question_id || '').trim()).filter(Boolean));
            let index = 1;
            while (used.has(`q${index}`)) index += 1;
            return `q${index}`;
        },

        createGroqPlaygroundQuestion(questionText = '', questionId = '') {
            const current = Array.isArray(this.groqPlayground?.questions) ? this.groqPlayground.questions : [];
            const sanitizedId = String(questionId || '').trim() || this._nextGroqQuestionId(current);
            return {
                client_id: `gpq_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
                question_id: sanitizedId,
                question_text: this._stripGroqPlaygroundAnnotations(questionText).trim(),
            };
        },

        restoreGroqPlaygroundState() {
            const gp = this.groqPlayground;
            let restored = null;
            try {
                const raw = localStorage.getItem(this.groqPlaygroundStorageKey());
                restored = raw ? JSON.parse(raw) : null;
            } catch (e) {
                console.warn('Failed to restore Groq playground state:', e);
            }

            gp.documentSource = restored?.documentSource === 'original' ? 'original' : 'current';
            if (restored?.documentSource === 'fictional') {
                gp.documentSource = 'fictional';
            }
            gp.fictionalVersionIndex = Number.isFinite(Number(restored?.fictionalVersionIndex))
                ? Number(restored.fictionalVersionIndex)
                : 0;
            gp.selectedModels = Array.isArray(restored?.selectedModels)
                ? restored.selectedModels.map((item) => String(item || '').trim()).filter(Boolean)
                : [];
            gp.questions = Array.isArray(restored?.questions)
                ? restored.questions
                    .map((item) => this.createGroqPlaygroundQuestion(item?.question_text || '', item?.question_id || ''))
                    .filter((item) => item.question_id || item.question_text)
                : [];
            if (gp.questions.length === 0) {
                gp.questions = [this.createGroqPlaygroundQuestion()];
            }
            gp.systemPrompt = String(restored?.systemPrompt || '').trim() || defaultGroqSystemPrompt;
            gp.temperature = Number.isFinite(Number(restored?.temperature)) ? Number(restored.temperature) : 0;
            gp.seed = Number.isFinite(Number(restored?.seed)) ? Number(restored.seed) : 23;
            gp.maxTokens = Number.isFinite(Number(restored?.maxTokens)) ? Number(restored.maxTokens) : 512;
            gp.showAdvanced = !!restored?.showAdvanced;
        },

        persistGroqPlaygroundState() {
            if (!this.isPowerUser) return;
            const gp = this.groqPlayground;
            const payload = {
                documentSource: gp.documentSource,
                fictionalVersionIndex: Number(gp.fictionalVersionIndex || 0),
                selectedModels: Array.isArray(gp.selectedModels) ? gp.selectedModels : [],
                questions: (gp.questions || []).map((item) => ({
                    question_id: String(item?.question_id || '').trim(),
                    question_text: String(item?.question_text || ''),
                })),
                systemPrompt: String(gp.systemPrompt || ''),
                temperature: Number(gp.temperature || 0),
                seed: Number(gp.seed || 23),
                maxTokens: Number(gp.maxTokens || 512),
                showAdvanced: !!gp.showAdvanced,
            };
            try {
                localStorage.setItem(this.groqPlaygroundStorageKey(), JSON.stringify(payload));
            } catch (e) {
                console.warn('Failed to persist Groq playground state:', e);
            }
        },

        selectGroqDocumentSource(source) {
            const nextSource = ['current', 'original', 'fictional'].includes(source) ? source : 'current';
            if (nextSource === 'fictional' && (!Array.isArray(this.fictionalVersions) || this.fictionalVersions.length === 0)) {
                return;
            }
            this.groqPlayground.documentSource = nextSource;
            this.persistGroqPlaygroundState();
        },

        async activateGroqFictionalSource() {
            if (this.generatingFictional) return;
            if (Array.isArray(this.fictionalVersions) && this.fictionalVersions.length > 0) {
                this.selectGroqDocumentSource('fictional');
                return;
            }
            await this.generateFictionalForGroq();
        },

        groqSelectedFictionalVersion() {
            const versions = Array.isArray(this.fictionalVersions) ? this.fictionalVersions : [];
            if (versions.length === 0) return null;
            const rawIndex = Number(this.groqPlayground?.fictionalVersionIndex || 0);
            const index = Math.min(Math.max(rawIndex, 0), versions.length - 1);
            return versions[index] || null;
        },

        groqSelectedFictionalVersionLabel() {
            const versions = Array.isArray(this.fictionalVersions) ? this.fictionalVersions : [];
            if (versions.length === 0) return 'No fictional version loaded';
            const rawIndex = Number(this.groqPlayground?.fictionalVersionIndex || 0);
            const index = Math.min(Math.max(rawIndex, 0), versions.length - 1);
            const version = versions[index] || {};
            const seed = version?.seed ?? '—';
            return `Version ${index + 1} · seed ${seed}`;
        },

        onGroqFictionalVersionChange() {
            const versions = Array.isArray(this.fictionalVersions) ? this.fictionalVersions : [];
            if (versions.length === 0) {
                this.groqPlayground.fictionalVersionIndex = 0;
            } else {
                const rawIndex = Number(this.groqPlayground.fictionalVersionIndex || 0);
                this.groqPlayground.fictionalVersionIndex = Math.min(Math.max(rawIndex, 0), versions.length - 1);
            }
            this.persistGroqPlaygroundState();
        },

        openGroqSelectedFictionalPreview() {
            const versions = Array.isArray(this.fictionalVersions) ? this.fictionalVersions : [];
            if (versions.length === 0) return;
            const rawIndex = Number(this.groqPlayground?.fictionalVersionIndex || 0);
            this.currentVersionIndex = Math.min(Math.max(rawIndex, 0), versions.length - 1);
            this.showFictionalModal = true;
        },

        async loadGroqPlaygroundModels(forceRefresh = false) {
            const gp = this.groqPlayground;
            if (gp.loadingModels) return;
            gp.loadingModels = true;
            gp.error = '';
            try {
                const payload = await API.getGroqPlaygroundModels();
                gp.configured = payload?.configured !== false;
                gp.availableModels = Array.isArray(payload?.models) ? payload.models : [];
                const availableIds = new Set(gp.availableModels.map((item) => String(item?.id || '').trim()).filter(Boolean));
                gp.selectedModels = (gp.selectedModels || []).filter((item) => availableIds.has(item));
                if (gp.selectedModels.length === 0 && gp.availableModels.length > 0) {
                    const preferred = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b'];
                    const defaults = preferred.filter((item) => availableIds.has(item));
                    gp.selectedModels = defaults.length ? defaults : [gp.availableModels[0].id];
                }
                if (!gp.configured) {
                    gp.error = 'GROQ_API_KEY is not configured on this deployment.';
                } else if (gp.availableModels.length === 0 && forceRefresh) {
                    gp.error = 'No Groq chat models are currently available.';
                }
                this.persistGroqPlaygroundState();
            } catch (e) {
                gp.configured = false;
                gp.error = e.message || 'Failed to load Groq models.';
            } finally {
                gp.loadingModels = false;
            }
        },

        async openGroqPlayground() {
            const gp = this.groqPlayground;
            if (!gp.initialized) {
                this.restoreGroqPlaygroundState();
                gp.initialized = true;
            }
            if (gp.documentSource === 'fictional' && (!Array.isArray(this.fictionalVersions) || this.fictionalVersions.length === 0)) {
                gp.documentSource = 'current';
            }
            if (Array.isArray(this.fictionalVersions) && this.fictionalVersions.length > 0) {
                const rawIndex = Number(gp.fictionalVersionIndex || 0);
                gp.fictionalVersionIndex = Math.min(Math.max(rawIndex, 0), this.fictionalVersions.length - 1);
            }
            this.showGroqPlaygroundModal = true;
            if (gp.availableModels.length === 0 && !gp.loadingModels) {
                await this.loadGroqPlaygroundModels();
            }
        },

        closeGroqPlayground() {
            this.showGroqPlaygroundModal = false;
        },

        filteredGroqPlaygroundModels() {
            const filterValue = String(this.groqPlayground?.modelFilter || '').trim().toLowerCase();
            const models = Array.isArray(this.groqPlayground?.availableModels) ? this.groqPlayground.availableModels.slice() : [];
            models.sort((a, b) => {
                const aSelected = (this.groqPlayground.selectedModels || []).includes(a.id) ? 0 : 1;
                const bSelected = (this.groqPlayground.selectedModels || []).includes(b.id) ? 0 : 1;
                if (aSelected !== bSelected) return aSelected - bSelected;
                return String(a?.id || '').localeCompare(String(b?.id || ''));
            });
            if (!filterValue) return models;
            return models.filter((item) => String(item?.id || '').toLowerCase().includes(filterValue));
        },

        isGroqModelSelected(modelId) {
            return (this.groqPlayground.selectedModels || []).includes(modelId);
        },

        toggleGroqModel(modelId) {
            const gp = this.groqPlayground;
            const current = Array.isArray(gp.selectedModels) ? gp.selectedModels.slice() : [];
            const index = current.indexOf(modelId);
            if (index >= 0) {
                current.splice(index, 1);
            } else {
                current.push(modelId);
            }
            gp.selectedModels = current;
            this.persistGroqPlaygroundState();
        },

        selectAllVisibleGroqModels() {
            const visible = this.filteredGroqPlaygroundModels().map((item) => item.id);
            const combined = new Set([...(this.groqPlayground.selectedModels || []), ...visible]);
            this.groqPlayground.selectedModels = Array.from(combined);
            this.persistGroqPlaygroundState();
        },

        clearGroqModelSelection() {
            this.groqPlayground.selectedModels = [];
            this.persistGroqPlaygroundState();
        },

        addGroqPlaygroundQuestion(questionText = '', questionId = '') {
            if (!Array.isArray(this.groqPlayground.questions)) {
                this.groqPlayground.questions = [];
            }
            this.groqPlayground.questions.push(this.createGroqPlaygroundQuestion(questionText, questionId));
            this.persistGroqPlaygroundState();
        },

        removeGroqPlaygroundQuestion(index) {
            if (!Array.isArray(this.groqPlayground.questions)) return;
            this.groqPlayground.questions.splice(index, 1);
            if (this.groqPlayground.questions.length === 0) {
                this.groqPlayground.questions = [this.createGroqPlaygroundQuestion()];
            }
            this.persistGroqPlaygroundState();
        },

        clearGroqPlaygroundQuestions() {
            this.groqPlayground.questions = [this.createGroqPlaygroundQuestion()];
            this.persistGroqPlaygroundState();
        },

        importAnnotatedQuestionsToGroqPlayground() {
            const sourceQuestions = Array.isArray(this.docData?.questions) ? this.docData.questions : [];
            if (sourceQuestions.length === 0) {
                showToast('This document has no annotated questions to import.', 'warning');
                return;
            }
            const seenTexts = new Set(
                (this.groqPlayground.questions || [])
                    .map((item) => this._stripGroqPlaygroundAnnotations(item?.question_text || '').trim().toLowerCase())
                    .filter(Boolean)
            );
            const imported = [];
            for (const question of sourceQuestions) {
                const questionText = this._stripGroqPlaygroundAnnotations(question?.question || '').trim();
                if (!questionText) continue;
                const normalizedText = questionText.toLowerCase();
                if (seenTexts.has(normalizedText)) continue;
                seenTexts.add(normalizedText);
                imported.push(this.createGroqPlaygroundQuestion(questionText, question?.question_id || ''));
            }
            if (imported.length === 0) {
                showToast('All annotated questions are already in the playground.', 'info');
                return;
            }
            const onlyBlankStarter = (this.groqPlayground.questions || []).length === 1
                && !String(this.groqPlayground.questions[0]?.question_text || '').trim();
            this.groqPlayground.questions = onlyBlankStarter
                ? imported
                : [...(this.groqPlayground.questions || []), ...imported];
            this.persistGroqPlaygroundState();
            showToast(`Imported ${imported.length} question${imported.length === 1 ? '' : 's'} into the playground.`, 'success');
        },

        resetGroqPlaygroundPrompt() {
            this.groqPlayground.systemPrompt = defaultGroqSystemPrompt;
            this.persistGroqPlaygroundState();
        },

        groqPlaygroundDocumentText() {
            if (this.groqPlayground.documentSource === 'fictional') {
                const version = this.groqSelectedFictionalVersion();
                return String(version?.generated_document || '').trim();
            }
            const source = this.groqPlayground.documentSource === 'original' ? 'original' : 'current';
            const raw = source === 'original'
                ? String(this.docData?.original_document || this.docData?.document_to_annotate || '')
                : String(this.docData?.document_to_annotate || '');
            return this._stripGroqPlaygroundAnnotations(raw).trim();
        },

        async runGroqPlayground() {
            const gp = this.groqPlayground;
            if (gp.running) return;

            const questions = (gp.questions || [])
                .map((item, index) => ({
                    question_id: String(item?.question_id || '').trim() || `q${index + 1}`,
                    question_text: this._stripGroqPlaygroundAnnotations(item?.question_text || '').trim(),
                }))
                .filter((item) => item.question_text);
            if (questions.length === 0) {
                showToast('Add at least one question before running the playground.', 'warning');
                return;
            }

            if (!Array.isArray(gp.selectedModels) || gp.selectedModels.length === 0) {
                showToast('Select at least one Groq model.', 'warning');
                return;
            }

            const documentText = this.groqPlaygroundDocumentText();
            if (!documentText) {
                showToast('This document has no text to send to Groq.', 'error');
                return;
            }

            gp.running = true;
            gp.error = '';
            gp.results = [];
            this.persistGroqPlaygroundState();
            try {
                const response = await API.runGroqPlayground(this.theme, this.doc_id, {
                    document_source: gp.documentSource,
                    document_text: documentText,
                    questions,
                    models: gp.selectedModels,
                    system_prompt: String(gp.systemPrompt || '').trim() || defaultGroqSystemPrompt,
                    temperature: Number(gp.temperature || 0),
                    seed: Number(gp.seed || 23),
                    max_tokens: Number(gp.maxTokens || 512),
                });
                gp.configured = response?.configured !== false;
                gp.results = Array.isArray(response?.results) ? response.results : [];
                gp.lastRunQuestions = Array.isArray(response?.questions) ? response.questions : questions;
                gp.lastDocumentCharCount = Number(response?.document_char_count || documentText.length || 0);
                gp.lastCompletedAt = new Date().toISOString().replace(/Z$/, '');

                const failedCount = gp.results.filter((item) => String(item?.error || '').trim()).length;
                const partialCount = gp.results.filter((item) => !String(item?.error || '').trim() && Number(item?.failed_questions || 0) > 0).length;
                if (failedCount === 0 && partialCount === 0) {
                    showToast(`Ran ${gp.results.length} Groq model${gp.results.length === 1 ? '' : 's'}.`, 'success');
                } else if (failedCount === gp.results.length) {
                    gp.error = 'All selected models failed.';
                    showToast('All selected Groq models failed.', 'error');
                } else if (failedCount === 0) {
                    showToast(`${partialCount} Groq model${partialCount === 1 ? '' : 's'} had question-level failures; partial results are shown.`, 'warning');
                } else {
                    showToast(`${failedCount} Groq model${failedCount === 1 ? '' : 's'} failed and ${partialCount} returned partial results.`, 'warning');
                }
            } catch (e) {
                gp.error = e.message || 'Groq playground run failed.';
                showToast('Groq playground failed: ' + gp.error, 'error');
            } finally {
                gp.running = false;
                this.persistGroqPlaygroundState();
            }
        },

        // --- Fictional Document Generation ---
        fictionalPreview: null,
        fictionalVersions: [],
        currentVersionIndex: 0,
        showFictionalModal: false,
        generatingFictional: false,
        fictionalHighlight: true,

        async _generateFictionalVersions({ showModal = false, sourceAfterGenerate = '' } = {}) {
            if (this.generatingFictional) return;
            this.generatingFictional = true;
            this.fictionalVersions = [];
            try {
                const targetVersionCount = 10;
                const baseSeed = 23;
                const seedStride = 1_000_003;
                const maxAttempts = 60;
                const batch = await API.generateFictionalBatch(this.theme, this.doc_id, {
                    seed: baseSeed,
                    targetVersionCount,
                    maxAttempts,
                    seedStride,
                });
                if (Array.isArray(batch?.versions)) {
                    this.fictionalVersions = batch.versions;
                }
                if (this.fictionalVersions.length === 0) {
                    throw new Error('No fictional preview could be generated.');
                }
                if (this.fictionalVersions.length < targetVersionCount) {
                    showToast(
                        `Generated ${this.fictionalVersions.length} unique versions (pool constraints limited diversity).`,
                        'warning'
                    );
                }
                this.currentVersionIndex = 0;
                this.groqPlayground.fictionalVersionIndex = 0;
                if (sourceAfterGenerate === 'fictional') {
                    this.groqPlayground.documentSource = 'fictional';
                }
                this.persistGroqPlaygroundState();
                if (showModal) {
                    this.showFictionalModal = true;
                } else {
                    showToast(
                        `Generated ${this.fictionalVersions.length} fictional version${this.fictionalVersions.length === 1 ? '' : 's'}.`,
                        'success'
                    );
                }
            } catch (e) {
                showToast('Generation failed: ' + e.message, 'error');
            } finally {
                this.generatingFictional = false;
            }
        },

        async generateFictional() {
            await this._generateFictionalVersions({ showModal: true });
        },

        async generateFictionalForGroq() {
            await this._generateFictionalVersions({ showModal: false, sourceAfterGenerate: 'fictional' });
        },

        get fictionalPreview() {
            return this.fictionalVersions[this.currentVersionIndex] || null;
        },

        nextVersion() {
            if (this.currentVersionIndex < this.fictionalVersions.length - 1) {
                this.currentVersionIndex++;
            }
        },

        prevVersion() {
            if (this.currentVersionIndex > 0) {
                this.currentVersionIndex--;
            }
        },

        closeFictionalModal() {
            this.showFictionalModal = false;
            this.fictionalVersions = [];
            this.currentVersionIndex = 0;
        },

        _fictionalVersionSignature(preview) {
            if (!preview || !Array.isArray(preview.entity_mapping)) {
                return JSON.stringify(preview || {});
            }
            const parts = [];
            for (const row of preview.entity_mapping) {
                const entityId = String(row?.entity_id || '');
                const attr = String(row?.attribute || '');
                const fictional = String(row?.fictional || '');
                parts.push(`${entityId}.${attr}=${fictional}`);
            }
            parts.sort();
            return parts.join('|');
        },

        get fictionalDocumentHtml() {
            if (!this.fictionalPreview) return '';
            const doc = this.fictionalPreview.generated_document;
            if (!this.fictionalHighlight) {
                return this._escHtml(doc);
            }
            return this._buildDiffHtml(
                doc,
                this.fictionalPreview.annotations,
                this.fictionalPreview.entity_mapping
            );
        },

        fictionalQuestionHtml(idx) {
            if (!this.fictionalPreview) return '';
            const q = this.fictionalPreview.generated_questions[idx];
            if (!q) return '';
            if (!this.fictionalHighlight) {
                return this._escHtml(q.question);
            }
            const qAnns = (this.fictionalPreview.question_annotations || [])[idx] || [];
            return this._buildDiffHtml(
                q.question,
                qAnns,
                this.fictionalPreview.entity_mapping
            );
        },

        fictionalAnswerExpressionHtml(q) {
            const answerExpr = String((q && (q.answer_expression || q.answer)) || '');
            if (!answerExpr) return '';
            if (typeof window.renderAnswerExpression === 'function') {
                return window.renderAnswerExpression(answerExpr);
            }
            return this._escHtml(answerExpr);
        },

        formatReferencedValue(value) {
            if (value === null || value === undefined || value === '') return '—';
            return String(value);
        },

        computeReferencedDelta(factual, fictional) {
            const f = Number(factual);
            const fx = Number(fictional);
            if (Number.isFinite(f) && Number.isFinite(fx)) {
                const d = fx - f;
                if (d > 0) return `+${d}`;
                return String(d);
            }
            return String(factual) === String(fictional) ? 'same' : 'changed';
        },

        _buildDiffHtml(generatedText, sourceAnnotations, entityMapping) {
            const generated = String(generatedText || '');

            // Build a lookup: entity_id.attribute -> {factual, fictional}
            const lookup = {};
            for (const m of entityMapping) {
                const key = m.entity_id + '.' + m.attribute;
                lookup[key] = m;
            }

            // Collect all replacements with their fictional values
            const replacements = [];
            for (const ann of sourceAnnotations) {
                const hasInlineFictional = ann && ann.fictional_text !== undefined && ann.fictional_text !== null;
                const key = ann.entity_id + '.' + (ann.attribute || '');
                const mapping = lookup[key];
                const fictionalValue = hasInlineFictional ? ann.fictional_text : (mapping ? mapping.fictional : '');
                if (!fictionalValue) continue;
                replacements.push({
                    factual: ann.factual_text,
                    fictional: fictionalValue,
                    entityId: ann.entity_id,
                    attribute: ann.attribute || '',
                });
            }

            // Find and highlight fictional values in the generated text.
            // Important: enforce token boundaries for alnum values to avoid
            // false matches inside other words (e.g. "she" inside "finished").
            const isWordChar = (ch) => /[A-Za-z0-9]/.test(ch || '');
            const overlapsExisting = (start, end, existing) => {
                return existing.some((s) => start < s.end && end > s.start);
            };

            const spans = [];
            for (const rep of replacements) {
                const fictional = String(rep.fictional || '');
                if (!fictional) continue;

                let startIdx = 0;
                while (true) {
                    const pos = generated.indexOf(fictional, startIdx);
                    if (pos === -1) break;

                    const end = pos + fictional.length;
                    const before = pos > 0 ? generated[pos - 1] : '';
                    const after = end < generated.length ? generated[end] : '';
                    const needsLeftBoundary = isWordChar(fictional[0]);
                    const needsRightBoundary = isWordChar(fictional[fictional.length - 1]);
                    const boundaryConflict =
                        (needsLeftBoundary && isWordChar(before)) ||
                        (needsRightBoundary && isWordChar(after));

                    // Check not already covered
                    const overlaps = overlapsExisting(pos, end, spans);
                    if (!boundaryConflict && !overlaps) {
                        spans.push({
                            start: pos,
                            end,
                            factual: String(rep.factual || ''),
                            fictional,
                            entityId: rep.entityId,
                            attribute: rep.attribute,
                        });
                        break; // Only match first occurrence per annotation
                    }
                    startIdx = pos + 1;
                }
            }

            spans.sort((a, b) => a.start - b.start);
            const chunks = [];
            let cursor = 0;
            for (const sp of spans) {
                if (sp.start < cursor) continue;
                chunks.push(this._escHtml(generated.slice(cursor, sp.start)));
                chunks.push(
                    '<span class="diff-replacement" title="' + this._escHtml(sp.entityId + '.' + sp.attribute) + '">' +
                        '<span class="diff-factual">' + this._escHtml(sp.factual) + '</span>' +
                        '<span class="diff-fictional">' + this._escHtml(sp.fictional) + '</span>' +
                    '</span>'
                );
                cursor = sp.end;
            }
            chunks.push(this._escHtml(generated.slice(cursor)));
            return chunks.join('');
        },

        _escHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        },
    };
}

// Warn before leaving with unsaved changes
window.addEventListener('beforeunload', function(e) {
    const app = document.querySelector('[x-data]')?.__x?.$data;
    if (app?.dirty) {
        e.preventDefault();
        e.returnValue = '';
    }
});
