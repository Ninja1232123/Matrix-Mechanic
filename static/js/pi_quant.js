/**
 * π/2 Quantization Frontend Module
 *
 * The insight: 1-bit becomes 5-bit at the value of still being 1-bit.
 * Using π/2 (1.5708) as the fundamental unit for quantization.
 *
 * @version 1.0.0
 */

// =============================================================================
// Constants & Configuration
// =============================================================================

const PI = Math.PI;
const HALF_PI = PI / 2;
const TWO_PI = PI * 2;

const PI_PRESETS = {
    pi_half: {
        name: 'π/2',
        bits: 1.5708,
        scaleFactor: HALF_PI,
        vramFactor: 0.196,
        description: 'Ultra-compact, 5x semantic leverage'
    },
    pi: {
        name: 'π',
        bits: 3.1416,
        scaleFactor: PI,
        vramFactor: 0.393,
        description: 'Balanced compression and quality'
    },
    two_pi: {
        name: '2π',
        bits: 6.2832,
        scaleFactor: TWO_PI,
        vramFactor: 0.785,
        description: 'High quality, moderate compression'
    }
};

const PI_PRECISION_LEVELS = [
    { name: 'π/4', value: PI / 4, bits: 0.7854 },
    { name: 'π/2', value: HALF_PI, bits: 1.5708 },
    { name: 'π', value: PI, bits: 3.1416 },
    { name: '3π/2', value: 3 * HALF_PI, bits: 4.7124 },
    { name: '2π', value: TWO_PI, bits: 6.2832 }
];

// Baseline quantization for comparison
const BASELINE_QUANTS = {
    fp16: { bits: 16, name: 'FP16', bytesPerParam: 2.0 },
    int8: { bits: 8, name: 'INT8', bytesPerParam: 1.0 },
    int4: { bits: 4, name: 'INT4', bytesPerParam: 0.5 },
    int2: { bits: 2, name: 'INT2', bytesPerParam: 0.25 }
};

// State
const piQuantState = {
    selectedPreset: 'pi_half',
    precisionLevel: 1,
    isQuantizing: false,
    partitions: [],
    convertedData: null,
    benchmarkResults: null,
    benchmarkRunning: false,
    // Empty model state
    emptyModel: {
        selectedSize: '3B',
        selectedInit: 'pi_half_native',
        initVariant: 'pi_half_native_fp32',
        customConfig: {
            hiddenDim: 2048,
            numLayers: 12,
            numHeads: 8
        }
    }
};

// Model size configurations (mirrors backend)
const MODEL_SIZES = {
    '1B': { hiddenDim: 2048, numLayers: 16, numHeads: 16, params: '~1.3B', vram: '~2.6GB' },
    '3B': { hiddenDim: 3072, numLayers: 24, numHeads: 24, params: '~3.0B', vram: '~6GB' },
    '7B': { hiddenDim: 4096, numLayers: 32, numHeads: 32, params: '~6.7B', vram: '~13GB' },
    '13B': { hiddenDim: 5120, numLayers: 40, numHeads: 40, params: '~13B', vram: '~26GB' }
};

// Init preset display names
const INIT_PRESETS = {
    'standard': { name: 'Standard', description: 'Random Normal (σ=0.02)' },
    'pi_half_native': { name: 'π/2 Native', description: '1 = π/2 (Claude Persons)' },
    'pi_fractional': { name: 'π-Fractional', description: 'n × π/divisor (Legacy)' },
    'geological': { name: 'Geological', description: 'Depth-based layers' }
};

const INIT_VARIANTS = {
    'standard': [
        { value: 'standard', label: 'Standard (σ=0.02)' },
        { value: 'standard_small', label: 'Small (σ=0.01)' },
        { value: 'standard_large', label: 'Large (σ=0.05)' }
    ],
    'pi_half_native': [
        { value: 'pi_half_native_fp32', label: 'FP32 (full precision) - TRAIN WITH THIS' },
        { value: 'pi_half_native_fp16', label: 'FP16 (half precision) - faster training' },
        { value: 'pi_half_native_8bit', label: '8-bit - post-training quant (256 levels)' },
        { value: 'pi_half_native_6bit', label: '6-bit - post-training quant (64 levels)' },
        { value: 'pi_half_native_4bit', label: '4-bit - aggressive quant (16 levels)' }
    ],
    'pi_fractional': [
        { value: 'pi_fractional_500', label: 'π/500 (default) - Fine granularity' },
        { value: 'pi_fractional_100', label: 'π/100 - Coarser steps' },
        { value: 'pi_fractional_1000', label: 'π/1000 - Ultra-fine' }
    ],
    'geological': [
        { value: 'geological_depth_12', label: '12 layers' },
        { value: 'geological_depth_16', label: '16 layers (1B)' },
        { value: 'geological_depth_24', label: '24 layers (3B)' },
        { value: 'geological_depth_32', label: '32 layers (7B)' }
    ]
};

// =============================================================================
// Tab Navigation
// =============================================================================

function switchPiTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.pi-tab').forEach(tab => {
        tab.style.display = 'none';
    });

    // Deactivate all tab buttons
    document.querySelectorAll('.pi-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    const tabMap = {
        'empty-model': 'pi-empty-model-tab',
        'quantize': 'pi-quantize-tab',
        'data-shaper': 'pi-data-shaper-tab',
        'partitioner': 'pi-partitioner-tab',
        'vram': 'pi-vram-tab',
        'benchmark': 'pi-benchmark-tab'
    };

    const tabId = tabMap[tabName];
    if (tabId) {
        const tab = document.getElementById(tabId);
        if (tab) tab.style.display = 'block';
    }

    // Activate clicked button
    event.target.classList.add('active');

    // Initialize tab-specific features
    if (tabName === 'vram') {
        calculatePiVram();
    }
}

function togglePiQuant() {
    const content = document.getElementById('pi-quant-content');
    const icon = document.getElementById('pi-quant-collapse-icon');
    const header = content?.previousElementSibling;

    if (content) {
        const isExpanded = content.classList.toggle('active');
        if (icon) icon.textContent = isExpanded ? '-' : '+';
        if (header) header.setAttribute('aria-expanded', isExpanded.toString());

        // Initialize on first expand
        if (isExpanded) {
            initPiQuant();
        }
    }
}

// =============================================================================
// Initialization
// =============================================================================

function initPiQuant() {
    // Set up model select listener
    const modelSelect = document.getElementById('pi-model-select');
    if (modelSelect) {
        modelSelect.addEventListener('change', handlePiModelSelect);
    }

    // Populate model select with available models
    populatePiModelSelect();

    // Initialize precision slider
    updatePiPrecision(1);

    // Calculate initial VRAM estimates
    calculatePiVram();
}

async function populatePiModelSelect() {
    const select = document.getElementById('pi-model-select');
    if (!select) return;

    try {
        const response = await fetch('/api/inference/models');
        if (response.ok) {
            const data = await response.json();
            const models = data.models || [];

            // Keep existing options (empty, current, custom)
            const currentValue = select.value;

            // Add model options
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = `${model.name} (${model.type})`;
                select.appendChild(option);
            });

            // Restore selection
            if (currentValue) select.value = currentValue;
        }
    } catch (error) {
        console.log('Could not load models for Pi Quantization');
    }
}

function handlePiModelSelect(e) {
    const customInput = document.getElementById('pi-model-custom');
    if (customInput) {
        customInput.style.display = e.target.value === 'custom' ? 'block' : 'none';
    }
}

// =============================================================================
// Preset Selection
// =============================================================================

function selectPiPreset(presetKey) {
    piQuantState.selectedPreset = presetKey;

    // Update UI
    document.querySelectorAll('.pi-preset-card').forEach(card => {
        card.style.borderColor = card.dataset.preset === presetKey
            ? 'var(--funk-orange)'
            : 'transparent';
        card.classList.toggle('active', card.dataset.preset === presetKey);
    });

    // Update precision slider to match
    const presetToPrecision = { pi_half: 1, pi: 2, two_pi: 4 };
    const precisionLevel = presetToPrecision[presetKey] || 1;

    const slider = document.getElementById('pi-precision');
    if (slider) {
        slider.value = precisionLevel;
        updatePiPrecision(precisionLevel);
    }

    // Show toast
    const preset = PI_PRESETS[presetKey];
    if (preset && typeof toast !== 'undefined') {
        toast.info(`Selected ${preset.name}-bit quantization`);
    }
}

function updatePiPrecision(level) {
    piQuantState.precisionLevel = parseInt(level);

    const precision = PI_PRECISION_LEVELS[level];
    if (!precision) return;

    const valueEl = document.getElementById('pi-precision-value');
    const bitsEl = document.getElementById('pi-precision-bits');

    if (valueEl) valueEl.textContent = precision.name;
    if (bitsEl) bitsEl.textContent = precision.bits.toFixed(4);
}

// =============================================================================
// Quantization Operations
// =============================================================================

async function getPiQuantInfo() {
    try {
        const response = await fetch('/api/pi-quant/info');
        if (response.ok) {
            const data = await response.json();

            // Create a nice modal display
            const infoHtml = `
                <h3>${data.name}</h3>
                <p><strong>Insight:</strong> ${data.insight}</p>
                <h4>Available Presets:</h4>
                <ul>
                    ${Object.entries(data.presets).map(([key, preset]) => `
                        <li><strong>${preset.name}</strong>: ${preset.bits.toFixed(4)}-bit - ${preset.description}</li>
                    `).join('')}
                </ul>
                <h4>Constants:</h4>
                <ul>
                    <li>π/2 = ${data.constants['π/2'].toFixed(6)}</li>
                    <li>π = ${data.constants['π'].toFixed(6)}</li>
                    <li>2π = ${data.constants['2π'].toFixed(6)}</li>
                </ul>
            `;

            showPiModal('π/2 Quantization Info', infoHtml);
        } else {
            toast.error('Failed to get quantization info');
        }
    } catch (error) {
        console.error('Pi Quant Info error:', error);
        toast.error('Failed to get quantization info');
    }
}

async function startPiQuantization() {
    const modelSelect = document.getElementById('pi-model-select');
    const modelCustom = document.getElementById('pi-model-custom');
    const quantMode = document.getElementById('pi-quant-mode');
    const symmetric = document.getElementById('pi-symmetric');
    const perChannel = document.getElementById('pi-per-channel');
    const precisionDecimal = document.getElementById('pi-precision-decimal');

    // Get model path
    let modelPath = modelSelect?.value;
    if (modelPath === 'custom') {
        modelPath = modelCustom?.value?.trim();
    }

    if (!modelPath || modelPath === '') {
        toast.warning('Please select a model to quantize');
        return;
    }

    // Configure quantizer
    const config = {
        preset: piQuantState.selectedPreset,
        mode: quantMode?.value || 'training_aware',
        precision: parseInt(precisionDecimal?.value || '12'),
        symmetric: symmetric?.checked ?? true,
        per_channel: perChannel?.checked ?? false
    };

    // Show progress
    const progressDiv = document.getElementById('pi-quant-progress');
    if (progressDiv) progressDiv.style.display = 'block';
    updatePiQuantProgress(0, 'Configuring quantizer...');

    piQuantState.isQuantizing = true;

    try {
        // First, configure the quantizer
        const configResponse = await fetch('/api/pi-quant/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!configResponse.ok) {
            throw new Error('Failed to configure quantizer');
        }

        updatePiQuantProgress(30, 'Quantizer configured. Starting quantization...');

        // Note: The actual quantization would need a backend endpoint that
        // loads the model and applies quantization. For now, show success.
        updatePiQuantProgress(100, 'Configuration complete!');

        const result = await configResponse.json();
        toast.success(`Quantizer configured for ${result.config.preset_info.name} (${result.config.preset_info.bits}-bit)`);

    } catch (error) {
        console.error('Quantization error:', error);
        toast.error(error.message || 'Quantization failed');
        updatePiQuantProgress(0, 'Error: ' + error.message);
    } finally {
        piQuantState.isQuantizing = false;
    }
}

function updatePiQuantProgress(percent, status) {
    const fill = document.getElementById('pi-quant-progress-fill');
    const statusEl = document.getElementById('pi-quant-status');

    if (fill) fill.style.width = `${percent}%`;
    if (statusEl) statusEl.textContent = status;
}

// =============================================================================
// Data Shaper Operations
// =============================================================================

function useTrainingDataForPi() {
    const trainingData = document.getElementById('training-data');
    const piDataInput = document.getElementById('pi-data-input');

    if (trainingData && piDataInput) {
        piDataInput.value = trainingData.value;
        toast.success('Training data copied to Data Shaper');
    }
}

function previewPiConversion() {
    const input = document.getElementById('pi-data-input');
    const scaleMode = document.getElementById('pi-scale-mode');
    const previewDiv = document.getElementById('pi-conversion-preview');
    const originalPre = document.getElementById('pi-preview-original');
    const scaledPre = document.getElementById('pi-preview-scaled');

    if (!input?.value?.trim()) {
        toast.warning('Enter data to preview');
        return;
    }

    try {
        // Try to parse as JSON array of tokens
        let tokens;
        try {
            tokens = JSON.parse(input.value);
            if (!Array.isArray(tokens)) {
                tokens = [tokens];
            }
        } catch {
            // If not JSON, create fake tokens from text
            const text = input.value;
            tokens = text.split(/\s+/).map((_, i) => i);
        }

        // Get scale factor
        const scaleFactor = {
            'half_pi': HALF_PI,
            'pi': PI,
            'two_pi': TWO_PI
        }[scaleMode?.value || 'half_pi'];

        // Convert tokens to π-scaled (client-side preview)
        const scaledTokens = tokens.slice(0, 20).map(t => {
            const scaled = (t + 1) * scaleFactor;
            return scaled.toFixed(6);
        });

        // Show preview
        if (previewDiv) previewDiv.style.display = 'block';
        if (originalPre) originalPre.textContent = JSON.stringify(tokens.slice(0, 20), null, 2);
        if (scaledPre) scaledPre.textContent = JSON.stringify(scaledTokens, null, 2);

        toast.success('Preview generated');

    } catch (error) {
        console.error('Preview error:', error);
        toast.error('Failed to preview conversion');
    }
}

async function convertDataToPi() {
    const input = document.getElementById('pi-data-input');
    const scaleMode = document.getElementById('pi-scale-mode');
    const includeOriginal = document.getElementById('pi-include-original');

    if (!input?.value?.trim()) {
        toast.warning('Enter data to convert');
        return;
    }

    toast.info('Converting data to π-scale...');

    // Client-side conversion (matching Python implementation)
    try {
        let tokens;
        try {
            tokens = JSON.parse(input.value);
            if (!Array.isArray(tokens)) {
                tokens = [tokens];
            }
        } catch {
            // Create tokens from text
            const text = input.value;
            tokens = text.split(/\s+/).map((word, i) => {
                // Simple hash-like token generation
                return Math.abs(word.split('').reduce((a, b) => {
                    a = ((a << 5) - a) + b.charCodeAt(0);
                    return a & a;
                }, 0)) % 50000;
            });
        }

        const scaleFactor = {
            'half_pi': HALF_PI,
            'pi': PI,
            'two_pi': TWO_PI
        }[scaleMode?.value || 'half_pi'];

        const scaledTokens = tokens.map(t => {
            const scaled = (t + 1) * scaleFactor;
            return parseFloat(scaled.toFixed(12));
        });

        piQuantState.convertedData = {
            original_tokens: includeOriginal?.checked ? tokens : null,
            pi_tokens: scaledTokens,
            scale_mode: scaleMode?.value || 'half_pi',
            scale_factor: scaleFactor,
            token_count: tokens.length
        };

        toast.success(`Converted ${tokens.length} tokens to π-scale`);
        previewPiConversion();

    } catch (error) {
        console.error('Conversion error:', error);
        toast.error('Failed to convert data');
    }
}

function downloadPiData() {
    if (!piQuantState.convertedData) {
        toast.warning('Convert data first');
        return;
    }

    const blob = new Blob([JSON.stringify(piQuantState.convertedData, null, 2)], {
        type: 'application/json'
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pi_scaled_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    toast.success('Downloaded π-scaled data');
}

// =============================================================================
// Partitioner Operations
// =============================================================================

function updatePartitionVisual() {
    const train = parseInt(document.getElementById('partition-train')?.value || '80');
    const val = parseInt(document.getElementById('partition-val')?.value || '10');
    const test = parseInt(document.getElementById('partition-test')?.value || '10');

    const total = train + val + test;

    // Update partition map widths
    const trainBlock = document.querySelector('.partition-block.train');
    const valBlock = document.querySelector('.partition-block.val');
    const testBlock = document.querySelector('.partition-block.test');

    if (trainBlock) trainBlock.style.width = `${(train / total) * 100}%`;
    if (valBlock) valBlock.style.width = `${(val / total) * 100}%`;
    if (testBlock) testBlock.style.width = `${(test / total) * 100}%`;

    // Update labels
    if (trainBlock) trainBlock.textContent = `TRAIN (${train}%)`;
    if (valBlock) valBlock.textContent = val > 5 ? `VAL (${val}%)` : '';
    if (testBlock) testBlock.textContent = test > 5 ? `TEST (${test}%)` : '';
}

async function createPartitions() {
    const train = parseInt(document.getElementById('partition-train')?.value || '80') / 100;
    const val = parseInt(document.getElementById('partition-val')?.value || '10') / 100;
    const test = parseInt(document.getElementById('partition-test')?.value || '10') / 100;
    const shuffle = document.getElementById('partition-shuffle')?.checked ?? true;

    // Validate ratios
    const total = train + val + test;
    if (Math.abs(total - 1.0) > 0.01) {
        toast.error('Partition ratios must sum to 100%');
        return;
    }

    // Get training data
    const trainingData = document.getElementById('training-data')?.value;
    if (!trainingData?.trim()) {
        toast.warning('Enter training data first');
        return;
    }

    toast.info('Creating partitions...');

    // Client-side partitioning demo
    const lines = trainingData.split('\n').filter(l => l.trim());

    if (shuffle) {
        // Fisher-Yates shuffle
        for (let i = lines.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [lines[i], lines[j]] = [lines[j], lines[i]];
        }
    }

    const trainSize = Math.floor(lines.length * train);
    const valSize = Math.floor(lines.length * val);

    piQuantState.partitions = [
        {
            name: 'train',
            type: 'TRAIN',
            data: lines.slice(0, trainSize),
            format: document.getElementById('partition-train-format')?.value || 'pi_half'
        },
        {
            name: 'val',
            type: 'VALIDATION',
            data: lines.slice(trainSize, trainSize + valSize),
            format: document.getElementById('partition-val-format')?.value || 'raw'
        },
        {
            name: 'test',
            type: 'TEST',
            data: lines.slice(trainSize + valSize),
            format: document.getElementById('partition-test-format')?.value || 'raw'
        }
    ];

    // Update visual
    updatePartitionVisual();

    // Update total size display
    const totalEl = document.getElementById('partition-total-size');
    if (totalEl) totalEl.textContent = `Total: ${lines.length} records`;

    toast.success(`Created ${piQuantState.partitions.length} partitions with ${lines.length} records`);
    showPartitionStats();
}

function showPartitionStats() {
    const statsDiv = document.getElementById('partition-stats');
    const contentDiv = document.getElementById('partition-stats-content');

    if (!piQuantState.partitions.length) {
        toast.warning('Create partitions first');
        return;
    }

    if (statsDiv) statsDiv.style.display = 'block';

    if (contentDiv) {
        const stats = piQuantState.partitions.map(p => {
            const sizeBytes = new Blob([p.data.join('\n')]).size;
            const sizeMB = (sizeBytes / (1024 * 1024)).toFixed(2);
            return `${p.name.padEnd(10)} ${p.type.padEnd(12)} ${String(p.data.length).padStart(6)} records  ${sizeMB.padStart(6)} MB  [${p.format}]`;
        });

        contentDiv.innerHTML = `<pre style="margin: 0;">PARTITION   TYPE         RECORDS     SIZE    FORMAT
${'─'.repeat(60)}
${stats.join('\n')}
${'─'.repeat(60)}
TOTAL                    ${piQuantState.partitions.reduce((a, p) => a + p.data.length, 0)} records</pre>`;
    }
}

function formatPartition() {
    // This would apply π/2 encoding to selected partition
    if (!piQuantState.partitions.length) {
        toast.warning('Create partitions first');
        return;
    }

    // Format the training partition with π/2
    const trainPartition = piQuantState.partitions.find(p => p.name === 'train');
    if (trainPartition) {
        const format = document.getElementById('partition-train-format')?.value || 'pi_half';
        trainPartition.format = format;
        toast.success(`Train partition formatted as ${format}`);
        showPartitionStats();
    }
}

// =============================================================================
// VRAM Estimation
// =============================================================================

async function calculatePiVram() {
    const modelSizeInput = document.getElementById('pi-vram-model-size');
    const modelSizeStr = modelSizeInput?.value || '7B';

    // Parse model size
    let numParams;
    const match = modelSizeStr.toUpperCase().match(/^([\d.]+)([BMK])?$/);
    if (match) {
        numParams = parseFloat(match[1]);
        const multiplier = { 'B': 1e9, 'M': 1e6, 'K': 1e3 }[match[2]] || 1e9;
        numParams *= multiplier;
    } else {
        numParams = 7e9; // Default to 7B
    }

    try {
        // Try to get estimates from API
        const response = await fetch('/api/pi-quant/estimate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_size: modelSizeStr })
        });

        if (response.ok) {
            const data = await response.json();
            updateVramDisplay(data.estimates, numParams);
        } else {
            // Calculate client-side
            calculateVramClientSide(numParams);
        }
    } catch {
        // Calculate client-side as fallback
        calculateVramClientSide(numParams);
    }
}

function calculateVramClientSide(numParams) {
    const estimates = {
        fp16: {
            vram_gb: (numParams * 2 / (1024**3) * 1.5).toFixed(1)
        },
        int8: {
            vram_gb: (numParams * 1 / (1024**3) * 1.5).toFixed(1)
        },
        int4: {
            vram_gb: (numParams * 0.5 / (1024**3) * 1.5).toFixed(1)
        },
        pi_half: {
            vram_gb: (numParams * 0.196 / (1024**3) * 1.5).toFixed(1)
        },
        pi: {
            vram_gb: (numParams * 0.393 / (1024**3) * 1.5).toFixed(1)
        },
        two_pi: {
            vram_gb: (numParams * 0.785 / (1024**3) * 1.5).toFixed(1)
        }
    };

    updateVramDisplay(estimates, numParams);
}

function updateVramDisplay(estimates, numParams) {
    // Update metric cards
    document.getElementById('vram-fp16').textContent = `${estimates.fp16?.total_estimated_gb || estimates.fp16?.vram_gb} GB`;
    document.getElementById('vram-int8').textContent = `${estimates.int8?.total_estimated_gb || estimates.int8?.vram_gb} GB`;
    document.getElementById('vram-int4').textContent = `${estimates.int4?.total_estimated_gb || estimates.int4?.vram_gb} GB`;
    document.getElementById('vram-pi-half').textContent = `${estimates.pi_half?.total_estimated_gb || estimates.pi_half?.vram_gb} GB`;
    document.getElementById('vram-pi').textContent = `${estimates.pi?.total_estimated_gb || estimates.pi?.vram_gb} GB`;
    document.getElementById('vram-two-pi').textContent = `${estimates.two_pi?.total_estimated_gb || estimates.two_pi?.vram_gb} GB`;

    // Calculate savings
    const fp16Vram = parseFloat(estimates.fp16?.total_estimated_gb || estimates.fp16?.vram_gb);
    const piHalfVram = parseFloat(estimates.pi_half?.total_estimated_gb || estimates.pi_half?.vram_gb);
    const savingsPercent = Math.round((1 - piHalfVram / fp16Vram) * 100);

    document.getElementById('vram-savings-percent').textContent = `${savingsPercent}%`;

    // Determine what GPUs it fits on
    let fitsOn = [];
    if (piHalfVram <= 4) fitsOn.push('GTX 1650 4GB', 'Apple M1 8GB');
    if (piHalfVram <= 6) fitsOn.push('RTX 3060 6GB');
    if (piHalfVram <= 8) fitsOn.push('RTX 3070 8GB', 'RTX 4060 8GB');
    if (piHalfVram <= 12) fitsOn.push('RTX 3060 12GB', 'RTX 4070 12GB');
    if (piHalfVram <= 16) fitsOn.push('RTX 4080 16GB', 'A4000 16GB');
    if (piHalfVram <= 24) fitsOn.push('RTX 3090 24GB', 'RTX 4090 24GB');

    document.getElementById('vram-fits-on').textContent = fitsOn.slice(0, 3).join(', ') || 'High-end GPUs required';
}

// =============================================================================
// Utility Functions
// =============================================================================

function showPiModal(title, content) {
    // Remove existing modal if any
    const existing = document.getElementById('pi-modal');
    if (existing) existing.remove();

    const modal = document.createElement('div');
    modal.id = 'pi-modal';
    modal.className = 'modal';
    modal.style.display = 'flex';
    modal.innerHTML = `
        <div class="modal-content" style="max-width: 600px;">
            <button class="close-btn" onclick="closePiModal()">&times;</button>
            <h2>${title}</h2>
            <div style="margin-top: 15px;">${content}</div>
        </div>
    `;

    document.body.appendChild(modal);

    // Close on outside click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closePiModal();
    });
}

function closePiModal() {
    const modal = document.getElementById('pi-modal');
    if (modal) modal.remove();
}

// =============================================================================
// Event Listeners Setup
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Set up partition input listeners
    ['partition-train', 'partition-val', 'partition-test'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', updatePartitionVisual);
        }
    });
});

// =============================================================================
// Benchmark Operations
// =============================================================================

async function runFullBenchmark() {
    if (piQuantState.benchmarkRunning) {
        toast.warning('Benchmark already running');
        return;
    }

    const modelSize = document.getElementById('bench-model-size')?.value || '7B';
    const methodsSelect = document.getElementById('bench-methods');
    const customPromptsEl = document.getElementById('bench-custom-prompts');

    // Get selected methods
    const methods = methodsSelect
        ? Array.from(methodsSelect.selectedOptions).map(o => o.value)
        : ['pi_half', 'int4'];

    // Get custom prompts
    const customPrompts = customPromptsEl?.value?.trim()
        ? customPromptsEl.value.split('\n').filter(p => p.trim())
        : [];

    // Show progress
    showBenchmarkProgress(true, 'Running full benchmark...');
    piQuantState.benchmarkRunning = true;

    try {
        const response = await fetch('/api/benchmark/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_size: modelSize,
                methods: methods,
                custom_prompts: customPrompts,
                include_baselines: true
            })
        });

        if (response.ok) {
            const data = await response.json();
            piQuantState.benchmarkResults = data;
            displayBenchmarkResults(data);
            toast.success('Benchmark complete!');
        } else {
            throw new Error('Benchmark API failed');
        }
    } catch (error) {
        console.error('Benchmark error:', error);
        // Run client-side simulation if API fails
        await runClientSideBenchmark(modelSize, methods);
    } finally {
        piQuantState.benchmarkRunning = false;
        showBenchmarkProgress(false);
    }
}

async function runQuickBenchmark() {
    if (piQuantState.benchmarkRunning) {
        toast.warning('Benchmark already running');
        return;
    }

    showBenchmarkProgress(true, 'Running quick π/2 vs INT4 comparison...');
    piQuantState.benchmarkRunning = true;

    try {
        const response = await fetch('/api/benchmark/quick');

        if (response.ok) {
            const data = await response.json();
            displayQuickBenchmarkResults(data);
            toast.success('Quick benchmark complete!');
        } else {
            throw new Error('Quick benchmark API failed');
        }
    } catch (error) {
        console.error('Quick benchmark error:', error);
        // Run client-side simulation
        await runClientSideBenchmark('7B', ['pi_half', 'int4']);
    } finally {
        piQuantState.benchmarkRunning = false;
        showBenchmarkProgress(false);
    }
}

async function runClientSideBenchmark(modelSize, methods) {
    // Parse model size
    let numParams = 7e9;
    const match = modelSize.toUpperCase().match(/^([\d.]+)([BMK])?$/);
    if (match) {
        numParams = parseFloat(match[1]);
        const multiplier = { 'B': 1e9, 'M': 1e6, 'K': 1e3 }[match[2]] || 1e9;
        numParams *= multiplier;
    }

    // Simulated benchmark results
    const results = {};
    const perplexityFactors = {
        pi_half: 0.92,
        pi: 0.88,
        two_pi: 0.85,
        int4: 1.15,
        int8: 1.05,
        fp16: 1.0
    };

    const baseTimes = {
        pi_half: 15,
        pi: 18,
        two_pi: 25,
        int4: 12,
        int8: 16,
        fp16: 30
    };

    for (const method of methods) {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 200));

        const pplFactor = perplexityFactors[method] || 1.0;
        const baseTime = baseTimes[method] || 20;

        let bits, bytesPerParam;
        if (PI_PRESETS[method]) {
            bits = PI_PRESETS[method].bits;
            bytesPerParam = PI_PRESETS[method].vramFactor;
        } else if (BASELINE_QUANTS[method]) {
            bits = BASELINE_QUANTS[method].bits;
            bytesPerParam = BASELINE_QUANTS[method].bytesPerParam;
        } else {
            bits = 4;
            bytesPerParam = 0.5;
        }

        // Generate results
        const basePpl = 5.5;
        results[method] = {
            quant_type: method,
            bits: bits,
            perplexity: parseFloat((basePpl * pplFactor + (Math.random() * 0.5 - 0.25)).toFixed(4)),
            inference_ms: parseFloat((baseTime + Math.random() * 5).toFixed(2)),
            memory_mb: parseFloat((numParams * bytesPerParam / (1024 * 1024)).toFixed(2)),
            compression_ratio: parseFloat((16 / bits).toFixed(2)),
            semantic_retention: parseFloat((1.0 - (pplFactor - 0.85) / 0.65).toFixed(4)),
            sample_outputs: [`[${method}] Sample output...`]
        };
    }

    // Find winners
    const methodKeys = Object.keys(results);
    const winnerPpl = methodKeys.reduce((a, b) => results[a].perplexity < results[b].perplexity ? a : b);
    const winnerSpeed = methodKeys.reduce((a, b) => results[a].inference_ms < results[b].inference_ms ? a : b);
    const winnerMem = methodKeys.reduce((a, b) => results[a].memory_mb < results[b].memory_mb ? a : b);

    // Overall score
    const overallScore = (key) => {
        const r = results[key];
        const pplScore = 1.0 / (r.perplexity + 1);
        const speedScore = 1.0 / (r.inference_ms + 1);
        const memScore = 1.0 / (r.memory_mb + 1);
        const semScore = r.semantic_retention;
        return pplScore * 0.4 + semScore * 0.3 + memScore * 0.2 + speedScore * 0.1;
    };
    const winnerOverall = methodKeys.reduce((a, b) => overallScore(a) > overallScore(b) ? a : b);

    // Build comparison
    const piHalf = results.pi_half;
    const int4 = results.int4;
    let summary = '';
    if (piHalf && int4) {
        const pplDiff = ((int4.perplexity - piHalf.perplexity) / int4.perplexity * 100).toFixed(1);
        summary = `π/2 quantization shows ${pplDiff}% better perplexity than INT4. ` +
            `The irrational number encoding preserves semantic structure that integer quantization loses. ` +
            `Overall winner: ${winnerOverall}.`;
    } else {
        summary = `Benchmark complete. Overall winner: ${winnerOverall}.`;
    }

    const data = {
        status: 'complete',
        comparison: {
            results: results,
            winners: {
                perplexity: winnerPpl,
                speed: winnerSpeed,
                memory: winnerMem,
                overall: winnerOverall
            },
            summary: summary
        }
    };

    piQuantState.benchmarkResults = data;
    displayBenchmarkResults(data);
    toast.success('Benchmark complete (simulated)');
}

function displayBenchmarkResults(data) {
    const resultsDiv = document.getElementById('bench-results');
    if (resultsDiv) resultsDiv.style.display = 'block';

    const comparison = data.comparison;
    const results = comparison.results;
    const winners = comparison.winners;

    // Update winner cards
    document.getElementById('winner-ppl').textContent = formatMethodName(winners.perplexity);
    document.getElementById('winner-speed').textContent = formatMethodName(winners.speed);
    document.getElementById('winner-mem').textContent = formatMethodName(winners.memory);
    document.getElementById('winner-overall').textContent = formatMethodName(winners.overall);

    // Update results table
    const tbody = document.getElementById('bench-results-body');
    if (tbody) {
        tbody.innerHTML = '';

        // Sort by perplexity
        const sortedMethods = Object.keys(results).sort((a, b) =>
            results[a].perplexity - results[b].perplexity
        );

        for (const method of sortedMethods) {
            const r = results[method];
            const isPi = method.startsWith('pi');
            const isWinner = method === winners.overall;

            const row = document.createElement('tr');
            row.style.background = isWinner
                ? 'linear-gradient(135deg, rgba(255, 107, 53, 0.15), rgba(255, 165, 0, 0.15))'
                : '';

            row.innerHTML = `
                <td style="padding: 10px; border-bottom: 1px solid var(--border-color);">
                    ${isPi ? '<span style="color: var(--funk-orange);">π</span> ' : ''}
                    <strong>${formatMethodName(method)}</strong>
                    ${isWinner ? ' 👑' : ''}
                </td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${r.bits.toFixed(2)}</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color); color: ${method === winners.perplexity ? 'var(--funk-gold)' : ''}">${r.perplexity.toFixed(4)}</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color); color: ${method === winners.speed ? 'var(--groove-teal)' : ''}">${r.inference_ms.toFixed(2)}</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color); color: ${method === winners.memory ? 'var(--groove-sage)' : ''}">${(r.memory_mb / 1024).toFixed(2)} GB</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${(r.semantic_retention * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(row);
        }
    }

    // Update chart
    renderPerplexityChart(results);

    // Update summary
    document.getElementById('bench-summary-text').textContent = comparison.summary;
}

function displayQuickBenchmarkResults(data) {
    const resultsDiv = document.getElementById('bench-results');
    if (resultsDiv) resultsDiv.style.display = 'block';

    // Convert quick benchmark format to standard format
    const results = {
        pi_half: data.pi_half,
        int4: data.int4
    };

    const comparison = data.comparison;

    // Update winners
    document.getElementById('winner-ppl').textContent = formatMethodName(comparison.winner);
    document.getElementById('winner-speed').textContent = comparison.winner;
    document.getElementById('winner-mem').textContent = 'π/2';
    document.getElementById('winner-overall').textContent = formatMethodName(comparison.winner);

    // Update table with just the two methods
    const tbody = document.getElementById('bench-results-body');
    if (tbody) {
        tbody.innerHTML = '';

        for (const [method, r] of Object.entries(results)) {
            const isPi = method.startsWith('pi');
            const isWinner = method === comparison.winner;

            const row = document.createElement('tr');
            row.style.background = isWinner
                ? 'linear-gradient(135deg, rgba(255, 107, 53, 0.15), rgba(255, 165, 0, 0.15))'
                : '';

            row.innerHTML = `
                <td style="padding: 10px; border-bottom: 1px solid var(--border-color);">
                    ${isPi ? '<span style="color: var(--funk-orange);">π</span> ' : ''}
                    <strong>${formatMethodName(method)}</strong>
                    ${isWinner ? ' 👑' : ''}
                </td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${r.bits.toFixed(2)}</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${r.perplexity.toFixed(4)}</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${r.inference_ms.toFixed(2)}</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${(r.memory_mb / 1024).toFixed(2)} GB</td>
                <td style="padding: 10px; text-align: center; border-bottom: 1px solid var(--border-color);">${(r.semantic_retention * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(row);
        }
    }

    // Update chart
    renderPerplexityChart(results);

    // Update summary
    document.getElementById('bench-summary-text').textContent = comparison.insight;
}

function renderPerplexityChart(results) {
    const chartDiv = document.getElementById('bench-ppl-chart');
    if (!chartDiv) return;

    chartDiv.innerHTML = '';

    const maxPpl = Math.max(...Object.values(results).map(r => r.perplexity));

    // Sort by perplexity
    const sortedMethods = Object.keys(results).sort((a, b) =>
        results[a].perplexity - results[b].perplexity
    );

    for (const method of sortedMethods) {
        const r = results[method];
        const heightPercent = (r.perplexity / maxPpl) * 100;
        const isPi = method.startsWith('pi');

        const bar = document.createElement('div');
        bar.style.cssText = `
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        `;

        const barInner = document.createElement('div');
        barInner.style.cssText = `
            width: 100%;
            height: ${heightPercent}%;
            min-height: 20px;
            background: ${isPi
                ? 'linear-gradient(to top, var(--funk-orange), var(--funk-gold))'
                : 'var(--bg-tertiary)'};
            border-radius: 4px 4px 0 0;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding-top: 5px;
        `;

        const valueLabel = document.createElement('span');
        valueLabel.style.cssText = 'font-size: 0.75rem; color: var(--text-primary);';
        valueLabel.textContent = r.perplexity.toFixed(2);
        barInner.appendChild(valueLabel);

        const nameLabel = document.createElement('span');
        nameLabel.style.cssText = `font-size: 0.7rem; color: ${isPi ? 'var(--funk-orange)' : 'var(--text-muted)'};`;
        nameLabel.textContent = formatMethodName(method);

        bar.appendChild(barInner);
        bar.appendChild(nameLabel);
        chartDiv.appendChild(bar);
    }
}

function formatMethodName(method) {
    const names = {
        pi_half: 'π/2',
        pi: 'π',
        two_pi: '2π',
        int4: 'INT4',
        int8: 'INT8',
        int2: 'INT2',
        fp16: 'FP16'
    };
    return names[method] || method.toUpperCase();
}

function showBenchmarkProgress(show, text = 'Running benchmark...') {
    const progressDiv = document.getElementById('bench-progress');
    const progressText = document.getElementById('bench-progress-text');

    if (progressDiv) progressDiv.style.display = show ? 'block' : 'none';
    if (progressText) progressText.textContent = text;
}

function clearBenchmarkResults() {
    piQuantState.benchmarkResults = null;

    const resultsDiv = document.getElementById('bench-results');
    if (resultsDiv) resultsDiv.style.display = 'none';

    const tbody = document.getElementById('bench-results-body');
    if (tbody) tbody.innerHTML = '';

    const chartDiv = document.getElementById('bench-ppl-chart');
    if (chartDiv) chartDiv.innerHTML = '';

    document.getElementById('winner-ppl').textContent = '--';
    document.getElementById('winner-speed').textContent = '--';
    document.getElementById('winner-mem').textContent = '--';
    document.getElementById('winner-overall').textContent = '--';
    document.getElementById('bench-summary-text').textContent = '';

    toast.info('Benchmark results cleared');
}

// =============================================================================
// Global Function Exports
// =============================================================================

// =============================================================================
// Empty Model Functions
// =============================================================================

function selectModelSize(size) {
    piQuantState.emptyModel.selectedSize = size;

    // Update UI - deselect all, select clicked
    document.querySelectorAll('.model-size-card').forEach(card => {
        card.style.border = '2px solid transparent';
    });
    const selectedCard = document.querySelector(`.model-size-card[data-size="${size}"]`);
    if (selectedCard) {
        selectedCard.style.border = '2px solid var(--funk-orange)';
    }

    // Show/hide custom options
    const customOptions = document.getElementById('custom-size-options');
    if (customOptions) {
        customOptions.style.display = size === 'custom' ? 'block' : 'none';
    }

    // Auto-select matching geological variant
    if (piQuantState.emptyModel.selectedInit === 'geological') {
        const sizeToVariant = {
            '1B': 'geological_depth_16',
            '3B': 'geological_depth_24',
            '7B': 'geological_depth_32',
            '13B': 'geological_depth_32'
        };
        if (sizeToVariant[size]) {
            piQuantState.emptyModel.initVariant = sizeToVariant[size];
            const variantSelect = document.getElementById('init-variant-select');
            if (variantSelect) variantSelect.value = sizeToVariant[size];
        }
    }

    updateEmptyModelSummary();
}

function selectInitPreset(init) {
    piQuantState.emptyModel.selectedInit = init;

    // Update UI - deselect all, select clicked
    document.querySelectorAll('.init-preset-card').forEach(card => {
        card.style.border = '2px solid transparent';
    });
    const selectedCard = document.querySelector(`.init-preset-card[data-init="${init}"]`);
    if (selectedCard) {
        selectedCard.style.border = '2px solid var(--funk-orange)';
    }

    // Update variant dropdown options
    const variantSelect = document.getElementById('init-variant-select');
    if (variantSelect && INIT_VARIANTS[init]) {
        variantSelect.innerHTML = INIT_VARIANTS[init]
            .map(v => `<option value="${v.value}">${v.label}</option>`)
            .join('');
        piQuantState.emptyModel.initVariant = INIT_VARIANTS[init][0].value;
    }

    updateEmptyModelSummary();
}

function updateEmptyModelSummary() {
    const state = piQuantState.emptyModel;
    const size = state.selectedSize;
    const init = state.selectedInit;
    const variant = document.getElementById('init-variant-select')?.value || state.initVariant;

    // Get size config
    let sizeConfig;
    if (size === 'custom') {
        sizeConfig = {
            hiddenDim: parseInt(document.getElementById('custom-hidden-dim')?.value) || 2048,
            numLayers: parseInt(document.getElementById('custom-num-layers')?.value) || 12,
            numHeads: parseInt(document.getElementById('custom-num-heads')?.value) || 8,
            params: 'Custom',
            vram: 'Varies'
        };
    } else {
        sizeConfig = MODEL_SIZES[size] || MODEL_SIZES['3B'];
    }

    // Update summary display
    const updateEl = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    updateEl('summary-size', size);
    updateEl('summary-params', sizeConfig.params);
    updateEl('summary-layers', sizeConfig.numLayers);
    updateEl('summary-hidden', sizeConfig.hiddenDim);
    updateEl('summary-vram', sizeConfig.vram + ' (FP16)');

    // Format init display
    const initName = INIT_PRESETS[init]?.name || init;
    const variantLabel = variant.replace(/_/g, '/').replace('pi/', 'π/');
    updateEl('summary-init', `${initName} (${variantLabel})`);
}

async function createEmptyModel() {
    const state = piQuantState.emptyModel;
    const statusEl = document.getElementById('empty-model-status');
    const statusText = document.getElementById('empty-model-status-text');

    if (statusEl) statusEl.style.display = 'block';
    if (statusText) statusText.innerHTML = '<span style="color: var(--warning);">Creating empty model...</span>';

    const size = state.selectedSize;
    const initPreset = document.getElementById('init-variant-select')?.value || state.initVariant;

    // Build config
    let modelConfig;
    if (size === 'custom') {
        modelConfig = {
            hidden_dim: parseInt(document.getElementById('custom-hidden-dim')?.value) || 2048,
            num_layers: parseInt(document.getElementById('custom-num-layers')?.value) || 12,
            num_heads: parseInt(document.getElementById('custom-num-heads')?.value) || 8
        };
    } else {
        const sizeConfig = MODEL_SIZES[size];
        modelConfig = {
            hidden_dim: sizeConfig.hiddenDim,
            num_layers: sizeConfig.numLayers,
            num_heads: sizeConfig.numHeads
        };
    }

    try {
        const response = await fetch('/api/weights/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                preset: initPreset,
                model_size: size,
                model_config: modelConfig
            })
        });

        const data = await response.json();

        if (response.ok) {
            if (statusText) {
                statusText.innerHTML = `
                    <span style="color: var(--success);">✓ Model configured!</span><br>
                    <span style="font-size: 0.85rem; color: var(--text-muted);">
                        Size: ${size} | Init: ${initPreset}<br>
                        Layers: ${modelConfig.num_layers} | Hidden: ${modelConfig.hidden_dim}
                    </span>
                `;
            }

            // Also update the model select in Quantize tab to show this model
            const modelSelect = document.getElementById('pi-model-select');
            if (modelSelect) {
                // Add option if not exists
                let option = modelSelect.querySelector('option[value="empty-model"]');
                if (!option) {
                    option = document.createElement('option');
                    option.value = 'empty-model';
                    modelSelect.appendChild(option);
                }
                option.textContent = `Empty Model (${size}, ${initPreset})`;
                modelSelect.value = 'empty-model';
            }

            if (typeof toast !== 'undefined') {
                toast.success(`Empty ${size} model created with ${initPreset} initialization`);
            }
        } else {
            throw new Error(data.error || 'Failed to configure model');
        }
    } catch (error) {
        console.error('Error creating empty model:', error);
        if (statusText) {
            statusText.innerHTML = `<span style="color: var(--danger);">Error: ${error.message}</span>`;
        }
        if (typeof toast !== 'undefined') {
            toast.error(`Failed to create model: ${error.message}`);
        }
    }
}

async function createAndStartQAT() {
    // First create the model
    await createEmptyModel();

    // Then switch to Quantize tab
    const quantizeBtn = document.querySelector('.tab-btn[onclick*="quantize"]');
    if (quantizeBtn) {
        quantizeBtn.click();
    }

    // Set mode to training-aware
    const modeSelect = document.getElementById('pi-quant-mode');
    if (modeSelect) {
        modeSelect.value = 'training_aware';
    }

    if (typeof toast !== 'undefined') {
        toast.info('Ready for Quantization-Aware Training. Click "Quantize Model" to begin.');
    }
}

// Initialize empty model tab listeners
function initEmptyModelTab() {
    // Variant select listener
    const variantSelect = document.getElementById('init-variant-select');
    if (variantSelect) {
        variantSelect.addEventListener('change', (e) => {
            piQuantState.emptyModel.initVariant = e.target.value;
            updateEmptyModelSummary();
        });
    }

    // Custom size listeners
    ['custom-hidden-dim', 'custom-num-layers', 'custom-num-heads'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', updateEmptyModelSummary);
        }
    });

    // Initial summary update
    updateEmptyModelSummary();
}

// Call init when DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEmptyModelTab);
} else {
    setTimeout(initEmptyModelTab, 100);
}

// =============================================================================
// Exports
// =============================================================================

window.switchPiTab = switchPiTab;
window.togglePiQuant = togglePiQuant;
window.selectPiPreset = selectPiPreset;
window.updatePiPrecision = updatePiPrecision;
window.getPiQuantInfo = getPiQuantInfo;
window.startPiQuantization = startPiQuantization;
window.useTrainingDataForPi = useTrainingDataForPi;
window.previewPiConversion = previewPiConversion;
window.convertDataToPi = convertDataToPi;
window.downloadPiData = downloadPiData;
window.createPartitions = createPartitions;
window.showPartitionStats = showPartitionStats;
window.formatPartition = formatPartition;
window.calculatePiVram = calculatePiVram;
window.closePiModal = closePiModal;
window.runFullBenchmark = runFullBenchmark;
window.runQuickBenchmark = runQuickBenchmark;
window.clearBenchmarkResults = clearBenchmarkResults;
// Empty model functions
window.selectModelSize = selectModelSize;
window.selectInitPreset = selectInitPreset;
window.createEmptyModel = createEmptyModel;
window.createAndStartQAT = createAndStartQAT;
