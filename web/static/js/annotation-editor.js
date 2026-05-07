/**
 * Annotation parsing and rendering utilities (v2).
 * Regex mirrors Python AnnotationParser: \[([^\]]+);\s*([^\]]+)\]
 */

const ANNOTATION_REGEX = /\[([^\]]+);\s*([^\]]+)\]/g;
const DEFAULT_ENTITY_TYPES = [
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
const LEGACY_ORGANIZATION_TYPE_ALIASES = {
    organisation: 'organization',
    military_organization: 'military_org',
    entreprise_organization: 'entreprise_org',
    ong: 'ngo',
    government_organization: 'government_org',
    educational_organization: 'educational_org',
    media_organization: 'media_org',
};
const LEGACY_ORGANIZATION_ATTRIBUTE_TO_TYPE = {
    is_education: 'educational_org',
    is_journalism: 'media_org',
    is_government: 'government_org',
    is_military: 'military_org',
    is_transportation: 'entreprise_org',
    is_technology: 'entreprise_org',
    is_finance: 'entreprise_org',
    is_medical: 'ngo',
    is_sport: 'ngo',
    is_military_org: 'military_org',
    is_entreprise_org: 'entreprise_org',
    is_ngo: 'ngo',
    is_government_org: 'government_org',
    is_educational_org: 'educational_org',
    is_media_org: 'media_org',
    is_military_organization: 'military_org',
    is_entreprise_organization: 'entreprise_org',
    is_ong: 'ngo',
    is_government_organization: 'government_org',
    is_educational_organization: 'educational_org',
    is_media_organization: 'media_org',
};

function parseEntityId(entityId) {
    const match = String(entityId || '').trim().match(/^([a-z_]+)_(\d+)$/);
    if (!match) return { entityType: null, entityIndex: null };
    return {
        entityType: match[1],
        entityIndex: Number.parseInt(match[2], 10),
    };
}

function canonicalOrganizationType(entityType) {
    const cleaned = String(entityType || '').trim();
    if (!cleaned) return null;
    if ([
        'organization',
        'military_org',
        'entreprise_org',
        'ngo',
        'government_org',
        'educational_org',
        'media_org',
    ].includes(cleaned)) {
        return cleaned;
    }
    return LEGACY_ORGANIZATION_TYPE_ALIASES[cleaned] || null;
}

function inferOrganizationType(entityType, attribute) {
    const canonicalType = canonicalOrganizationType(entityType);
    if (canonicalType && canonicalType !== 'organization') {
        return canonicalType;
    }
    const cleanedAttribute = String(attribute || '').trim();
    if (cleanedAttribute && LEGACY_ORGANIZATION_ATTRIBUTE_TO_TYPE[cleanedAttribute]) {
        return LEGACY_ORGANIZATION_ATTRIBUTE_TO_TYPE[cleanedAttribute];
    }
    return canonicalType;
}

function canonicalizeEntityReference(entityId, attribute) {
    const { entityType, entityIndex } = parseEntityId(String(entityId || '').replace(/^organisation_/, 'organization_'));
    if (!entityType || !entityIndex) {
        return {
            entityId: String(entityId || '').replace(/^organisation_/, 'organization_'),
            attribute: attribute || null,
            entityType: entityType || String(entityId || ''),
            ref: attribute ? `${String(entityId || '')}.${attribute}` : String(entityId || ''),
        };
    }

    const organizationType = inferOrganizationType(entityType, attribute);
    const canonicalType = organizationType || entityType;
    const canonicalEntityId = `${canonicalType}_${entityIndex}`;
    let canonicalAttribute = attribute || null;
    if (organizationType) {
        canonicalAttribute = 'name';
    } else if (canonicalOrganizationType(entityType)) {
        canonicalAttribute = canonicalAttribute || 'name';
    }

    return {
        entityId: canonicalEntityId,
        attribute: canonicalAttribute,
        entityType: canonicalType,
        ref: canonicalAttribute ? `${canonicalEntityId}.${canonicalAttribute}` : canonicalEntityId,
    };
}

/**
 * Check if a position range would overlap with existing annotations.
 * @param {string} text - Raw YAML text with annotations
 * @param {number} newStart - Proposed start position
 * @param {number} newEnd - Proposed end position
 * @param {number} excludeStart - Start of annotation to exclude from check (the one being moved)
 * @param {number} excludeEnd - End of annotation to exclude from check
 * @returns {boolean} - True if position is valid (no collision), false if collision detected
 */
function isValidAnnotationPosition(text, newStart, newEnd, excludeStart, excludeEnd) {
    const annotations = parseAnnotations(text);
    
    for (const ann of annotations) {
        // Skip the annotation we're moving
        if (ann.start === excludeStart && ann.end === excludeEnd) continue;
        
        // Check for overlap: [newStart, newEnd) overlaps with [ann.start, ann.end)
        // Overlap occurs if: newStart < ann.end AND newEnd > ann.start
        if (newStart < ann.end && newEnd > ann.start) {
            return false; // Collision detected
        }
    }
    
    return true; // No collision
}

/**
 * Parse all annotations from annotated text.
 */
function parseAnnotations(text) {
    if (!text) return [];
    const annotations = [];
    let match;
    const re = new RegExp(ANNOTATION_REGEX.source, 'g');
    while ((match = re.exec(text)) !== null) {
        const originalText = match[1];
        const entityRef = match[2].trim();
        let entityId, attribute;
        if (entityRef.includes('.')) {
            const dotIdx = entityRef.indexOf('.');
            entityId = entityRef.substring(0, dotIdx);
            attribute = entityRef.substring(dotIdx + 1);
        } else {
            entityId = entityRef;
            attribute = null;
        }
        const canonical = canonicalizeEntityReference(entityId, attribute);

        annotations.push({
            start: match.index,
            end: match.index + match[0].length,
            fullMatch: match[0],
            text: originalText,
            entityId: canonical.entityId,
            attribute: canonical.attribute,
            entityType: canonical.entityType,
            ref: canonical.ref,
        });
    }
    return annotations;
}

/**
 * Render annotated text to HTML with colored highlights and data attributes.
 * Tooltips are rendered via CSS ::after using data-ref, so they don't add
 * text nodes to the DOM and won't pollute Range.toString() offsets.
 * @param {string} text - Raw annotated text
 * @param {string|null} annotatedBy - Model/user name to show in the tooltip (optional)
 * @param {object} options - Rendering options
 * @param {'text'|'ref'} options.displayMode - Which label to show inside annotated spans
 * @param {boolean} options.showControls - Whether resize/delete controls should be rendered
 */
function renderAnnotatedHtml(text, annotatedBy = null, options = {}) {
    if (!text) return '';
    const annotations = parseAnnotations(text);
    if (annotations.length === 0) return escapeHtml(text);
    const displayMode = options?.displayMode === 'ref' ? 'ref' : 'text';
    const showControls = options?.showControls !== false;

    let html = '';
    let lastEnd = 0;

    for (const ann of annotations) {
        html += escapeHtml(text.substring(lastEnd, ann.start));
        const typeClass = `ann-${ann.entityType}`;
        const titleParts = [
            `Entity: ${ann.entityId}`,
            `Type: ${ann.entityType}`,
        ];
        if (ann.attribute) titleParts.push(`Attribute: ${ann.attribute}`);
        if (annotatedBy) titleParts.push(`By: ${annotatedBy}`);
        const hoverTitle = titleParts.join(' | ');
        const displayText = displayMode === 'ref' && ann.ref ? ann.ref : ann.text;
        const classes = ['ann', typeClass];
        if (displayMode === 'ref') {
            classes.push('ann-ref-view');
        }
        html += `<span class="${classes.join(' ')}"` +
            ` data-ref="${escapeAttr(ann.ref)}"` +
            ` data-entity-id="${escapeAttr(ann.entityId)}"` +
            ` data-entity-type="${escapeAttr(ann.entityType)}"` +
            ` data-start="${ann.start}" data-end="${ann.end}"` +
            ` data-display-mode="${escapeAttr(displayMode)}"` +
            ` title="${escapeAttr(hoverTitle)}"` +
            (annotatedBy ? ` data-annotated-by="${escapeAttr(annotatedBy)}"` : '') +
            `>`;
        if (showControls) {
            html += `<span class="resize-handle left" data-side="left"></span>`;
            html += `<button class="ann-delete-btn" title="Delete this annotation">×</button>`;
        }
        html += escapeHtml(displayText);
        if (showControls) {
            html += `<span class="resize-handle right" data-side="right"></span>`;
        }
        html += '</span>';
        lastEnd = ann.end;
    }
    html += escapeHtml(text.substring(lastEnd));
    return html;
}

/**
 * Extract unique entities from annotations, grouped by type.
 */
function extractEntitiesFromText(text, entityTypes = DEFAULT_ENTITY_TYPES) {
    const annotations = parseAnnotations(text);
    const entityMap = {};

    for (const ann of annotations) {
        if (!entityMap[ann.entityId]) {
            entityMap[ann.entityId] = { type: ann.entityType, attrs: {}, count: 0 };
        }
        entityMap[ann.entityId].count++;
        if (ann.attribute && ann.text) {
            entityMap[ann.entityId].attrs[ann.attribute] = ann.text.trim();
        }
    }

    const groups = {};
    for (const type of entityTypes || []) groups[type] = [];

    for (const [id, data] of Object.entries(entityMap)) {
        const type = data.type;
        if (!groups[type]) groups[type] = [];

        const attrs = Object.entries(data.attrs).map(([key, value]) => ({ key, value }));
        const preview = data.attrs.full_name || data.attrs.name || data.attrs.city ||
                       data.attrs.country || data.attrs.demonym || data.attrs.reference_code ||
                       data.attrs.day || data.attrs.date || data.attrs.int ||
                       data.attrs.float || data.attrs.percent || data.attrs.proportion ||
                       data.attrs.str || attrs[0]?.value || '';

        groups[type].push({ id, type, preview: truncate(preview, 22), attrs, count: data.count });
    }

    for (const type of Object.keys(groups)) {
        groups[type].sort((a, b) => {
            const numA = parseInt(a.id.split('_').pop()) || 0;
            const numB = parseInt(b.id.split('_').pop()) || 0;
            return numA - numB;
        });
    }

    return groups;
}

/**
 * Highlight all annotation spans matching a given entity ID.
 * @param {string} entityId - The entity to highlight (e.g., "person_1")
 * @param {HTMLElement|null} hoveredElement - The specific element being hovered (gets strong highlight)
 */
function highlightEntity(entityId, hoveredElement = null) {
    if (!entityId) return;
    document.querySelectorAll('.ann').forEach(el => {
        if (el.dataset.entityId === entityId) {
            if (el === hoveredElement) {
                el.classList.add('glow-strong');
                el.classList.remove('glow');
            } else {
                el.classList.add('glow');
                el.classList.remove('glow-strong');
            }
        }
    });
}

/**
 * Remove all entity highlights.
 */
function clearEntityHighlight() {
    document.querySelectorAll('.ann.glow, .ann.glow-strong, .ann.glow-ref, .ann.glow-dim').forEach(el => {
        el.classList.remove('glow', 'glow-strong', 'glow-ref', 'glow-dim');
    });
}

/**
 * Highlight all annotations of a given entity type, dim others.
 */
function highlightEntityType(entityType) {
    document.querySelectorAll('.ann').forEach(el => {
        if (el.dataset.entityType === entityType) {
            el.classList.add('type-highlight');
            el.classList.remove('dimmed');
        } else {
            el.classList.add('dimmed');
            el.classList.remove('type-highlight');
        }
    });
}

/**
 * Clear all type highlights and dimming.
 */
function clearTypeHighlight() {
    document.querySelectorAll('.ann').forEach(el => {
        el.classList.remove('type-highlight', 'dimmed');
    });
}

/**
 * Insert an annotation into raw text at a given character range.
 */
function insertAnnotation(text, start, end, entityRef) {
    const selectedText = text.substring(start, end);
    const annotation = `[${selectedText}; ${entityRef}]`;
    return text.substring(0, start) + annotation + text.substring(end);
}

/**
 * Remove an annotation from text (unwrap brackets, keep original text).
 */
function removeAnnotation(text, annStart, annEnd) {
    const fullMatch = text.substring(annStart, annEnd);
    const match = fullMatch.match(/^\[([^\]]+);\s*[^\]]+\]$/);
    if (match) {
        return text.substring(0, annStart) + match[1] + text.substring(annEnd);
    }
    return text;
}

/**
 * Resize an annotation span: change which text portion is annotated
 * while keeping the same entity reference.
 */
function resizeAnnotation(text, annStart, annEnd, newTextStart, newTextEnd) {
    const fullMatch = text.substring(annStart, annEnd);
    const refMatch = fullMatch.match(/^\[([^\]]+);\s*([^\]]+)\]$/);
    if (!refMatch) return text;

    const entityRef = refMatch[2].trim();
    const oldText = refMatch[1];

    // Remove the old annotation, keeping only its visible text.
    const result = text.substring(0, annStart) + oldText + text.substring(annEnd);

    // Raw layout inside the original annotation: "[" + oldText + "; ref]"
    const textRawStart = annStart + 1;
    const textRawEnd = textRawStart + oldText.length;
    const offset = oldText.length - (annEnd - annStart); // negative markup delta

    // Map a position from annotated-raw coordinates -> unwrapped coordinates.
    const mapRawToUnwrapped = (pos) => {
        if (pos <= annStart) return pos;
        if (pos >= annEnd) return pos + offset;
        // Inside the annotation payload.
        if (pos <= textRawStart) return annStart;
        if (pos <= textRawEnd) return annStart + (pos - textRawStart);
        // Inside metadata ("; entity.attr]") => clamp to end of visible text.
        return annStart + oldText.length;
    };

    let adjStart = mapRawToUnwrapped(newTextStart);
    let adjEnd = mapRawToUnwrapped(newTextEnd);

    // Normalize and clamp.
    if (adjEnd < adjStart) {
        const tmp = adjStart;
        adjStart = adjEnd;
        adjEnd = tmp;
    }
    adjStart = Math.max(0, Math.min(adjStart, result.length));
    adjEnd = Math.max(0, Math.min(adjEnd, result.length));

    // Keep at least one character in the annotation.
    if (adjEnd <= adjStart) {
        adjEnd = Math.min(result.length, adjStart + 1);
    }
    if (adjEnd <= adjStart) {
        return text;
    }

    const newText = result.substring(adjStart, adjEnd);
    const annotation = `[${newText}; ${entityRef}]`;
    return result.substring(0, adjStart) + annotation + result.substring(adjEnd);
}

// --- Utilities ---

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len) + '...' : str;
}

/**
 * Render answer expression (like "number_6.int - number_7.int") with entity references as colored spans.
 * This is different from renderAnnotatedHtml which expects [text; entity_id.attr] syntax.
 */
function renderAnswerExpression(answerText) {
    if (!answerText) return '';
    
    // Match entity references like: person_1.age, number_6.int, place_2.city
    // Pattern: word_number.attribute or just word_number
    const entityRefPattern = /(\w+_\d+(?:\.\w+)?)/g;
    
    let html = '';
    let lastIndex = 0;
    let match;
    
    while ((match = entityRefPattern.exec(answerText)) !== null) {
        const ref = match[1];
        const startIdx = match.index;
        
        // Add text before this match
        html += escapeHtml(answerText.substring(lastIndex, startIdx));
        
        // Parse entity ID and determine type
        let entityId, attribute;
        if (ref.includes('.')) {
            const dotIdx = ref.indexOf('.');
            entityId = ref.substring(0, dotIdx);
            attribute = ref.substring(dotIdx + 1);
        } else {
            entityId = ref;
            attribute = null;
        }
        
        const underscoreIdx = entityId.lastIndexOf('_');
        const entityType = underscoreIdx > 0 ? entityId.substring(0, underscoreIdx) : entityId;
        const typeClass = `ann-${entityType}`;
        const titleParts = [
            `Entity: ${entityId}`,
            `Type: ${entityType}`,
        ];
        if (attribute) titleParts.push(`Attribute: ${attribute}`);
        const hoverTitle = titleParts.join(' | ');
        
        // Create colored span for entity reference
        html += `<span class="ann ${typeClass}"` +
            ` data-ref="${escapeAttr(ref)}"` +
            ` data-entity-id="${escapeAttr(entityId)}"` +
            ` data-entity-type="${escapeAttr(entityType)}"` +
            ` title="${escapeAttr(hoverTitle)}">`;
        html += escapeHtml(ref);
        html += '</span>';
        
        lastIndex = startIdx + ref.length;
    }
    
    // Add remaining text
    html += escapeHtml(answerText.substring(lastIndex));
    
    return html;
}

// Expose functions to window for Alpine.js access
window.highlightEntity = highlightEntity;
window.clearEntityHighlight = clearEntityHighlight;
window.highlightEntityType = highlightEntityType;
window.clearTypeHighlight = clearTypeHighlight;
window.renderAnswerExpression = renderAnswerExpression;
window.DEFAULT_ENTITY_TYPES = DEFAULT_ENTITY_TYPES;
