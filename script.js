// Logistic Regression Solver (Frontend-only)

function parseCSV(text) {
    // Accepts CSV with x[,x2],y per line, robust to invalid lines
    return text.trim().split(/\n|\r/).map(line => {
        if (!line.trim()) return null;
        const parts = line.split(',').map(s => s.trim());
        if (parts.length < 2) return null;
        const nums = parts.map(Number);
        if (nums.some(n => isNaN(n))) return null;
        return { x: nums.slice(0, -1), y: nums[nums.length - 1] };
    }).filter(Boolean);
}

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

function runLogisticRegression(data, lr, epochs, k) {
    if (!Array.isArray(data) || data.length === 0) return 'No data.';
    const nFeatures = data[0].x.length;
    let weights = Array(nFeatures).fill(0);
    let bias = 0;
    // Header: w0,w1,w2,...
    let results = [bias,...weights].map(w=>w.toFixed(5)).join(',') + '\n';
    for (let epoch = 1; epoch <= epochs; epoch++) {
        let dw = Array(nFeatures).fill(0);
        let db = 0;
        for (const row of data) {
            if (!row || !Array.isArray(row.x) || typeof row.y !== 'number') continue;
            const z = weights.reduce((sum, w, i) => sum + w * row.x[i], bias);
            const y_pred = sigmoid(z);
            if (isNaN(y_pred)) return 'Computation error: check your data.';
            const error = y_pred - row.y;
            for (let i = 0; i < nFeatures; i++) {
                dw[i] += error * row.x[i];
            }
            db += error;
        }
        for (let i = 0; i < nFeatures; i++) {
            if (!isFinite(dw[i])) return 'Computation error: check your data.';
            weights[i] -= lr * dw[i] / data.length;
        }
        if (!isFinite(db)) return 'Computation error: check your data.';
        bias -= lr * db / data.length;
        // Output every k epochs, and always the last epoch
        if (epoch % k === 0 || epoch === epochs) {
            if (![bias,...weights].every(Number.isFinite)) return 'Computation error: check your data.';
            results += [bias,...weights].map(w=>w.toFixed(5)).join(',') + '\n';
        }
    }
    return results;
}

document.getElementById('run-btn').onclick = function() {
    const csv = document.getElementById('csv-input').value;
    const lr = parseFloat(document.getElementById('learning-rate').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    const k = parseInt(document.getElementById('k-epochs').value);
    const data = parseCSV(csv);
    const results = runLogisticRegression(data, lr, epochs, k);
    document.getElementById('results').value = results;
};

document.getElementById('clear-btn').onclick = function() {
    document.getElementById('csv-input').value = '';
    document.getElementById('results').value = '';
};

document.getElementById('copy-btn').onclick = function() {
    const results = document.getElementById('results').value;
    if (!results) return;
    navigator.clipboard.writeText(results).then(() => {
        document.getElementById('copy-btn').textContent = 'Copied!';
        setTimeout(()=>{
            document.getElementById('copy-btn').textContent = 'Copy';
        }, 1200);
    });
};
