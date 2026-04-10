// ================================================================
// API KEYS & CONFIGURATION
// ================================================================
// API keys are loaded from config.js
// In production, these should come from environment variables
// ================================================================

var personDataUri = null;
var garmentDataUri = null;
var personB64 = null;
var garmentB64 = null;
var resultUrl = null;

// ---- LOG ----
function log(msg, cls) {
  var box = document.getElementById('logContent');
  var d = document.createElement('div');
  if (cls) d.className = cls;
  d.textContent = new Date().toLocaleTimeString() + ' | ' + msg;
  box.appendChild(d);
  box.scrollTop = 999999;
  console.log('[VTON] ' + msg);
}

function setStatus(html, cls) {
  document.getElementById('status').innerHTML = html;
  document.getElementById('status').className = cls || '';
}

// ---- MODEL ----
function onModelChange() {
  var v = document.getElementById('selModel').value;
  var labels = {catvton:'CatVTON',kling:'Kling Kolors v1.5',nbp:'Nano Banana Pro',qwen:'Qwen Image Max',fashn:'FASHN v1.5'};
  var prices = {catvton:'~$0.01',kling:'$0.07',nbp:'$0.15',qwen:'$0.075',fashn:'Your GPU'};
  document.getElementById('mLabel').textContent = labels[v];
  document.getElementById('mPrice').textContent = prices[v];
}

// ---- FILE SELECT ----
function onFile(input, type) {
  var file = input.files[0];
  if (!file) return;
  log('Selected ' + type + ': ' + file.name + ' (' + Math.round(file.size/1024) + 'KB)', 'b');
  var reader = new FileReader();
  reader.onload = function(e) {
    var dataUri = e.target.result;
    var b64 = dataUri.split(',')[1];
    if (type === 'person') {
      personDataUri = dataUri;
      personB64 = b64;
      document.getElementById('prevPerson').src = dataUri;
      document.getElementById('cardPerson').classList.add('has');
    } else {
      garmentDataUri = dataUri;
      garmentB64 = b64;
      document.getElementById('prevGarment').src = dataUri;
      document.getElementById('cardGarment').classList.add('has');
    }
    log(type + ' image ready', 'g');
  };
  reader.readAsDataURL(file);
}

// ---- SHOW RESULT ----
function showResult(url) {
  resultUrl = url;
  document.getElementById('resultImg').src = url;
  document.getElementById('resultImg').style.display = 'block';
  document.getElementById('resultEmpty').style.display = 'none';
  document.getElementById('btnDl').style.display = 'inline-block';
  setStatus('Done!', 'ok');
  log('Result displayed', 'g');
}

function doDownload() {
  if (!resultUrl) return;
  var a = document.createElement('a');
  a.href = resultUrl; a.download = 'tryon_result.png';
  if (resultUrl.startsWith('http')) a.target = '_blank';
  a.click();
}

// ============================================================
// FAL.AI API (plain fetch, queue-based, from the official docs)
// Docs: https://fal.ai/models/fal-ai/cat-vton/api
//
// Submit:  POST https://queue.fal.run/{endpoint}
// Status:  GET  https://queue.fal.run/{endpoint}/requests/{id}/status
// Result:  GET  https://queue.fal.run/{endpoint}/requests/{id}
// Auth:    Authorization: Key {FAL_KEY}
// Body:    JSON input directly (NOT wrapped in {input:})
// Files:   Data URIs accepted as image URLs
// ============================================================

// Submit to queue — returns { request_id, status_url, response_url }
function falSubmit(endpoint, body) {
  log('POST queue.fal.run/' + endpoint);
  return fetch('https://queue.fal.run/' + endpoint, {
    method: 'POST',
    headers: {
      'Authorization': 'Key ' + FAL_KEY,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  }).then(function(r) {
    if (!r.ok) return r.text().then(function(t) { throw new Error('fal submit ' + r.status + ': ' + t); });
    return r.json();
  });
}

// Poll using the status_url from submit response (NOT manually constructed)
function falPoll(statusUrl, responseUrl, attempt) {
  if (attempt > 120) throw new Error('Timeout after 4 min');
  return new Promise(function(res) { setTimeout(res, 2000); })
  .then(function() {
    return fetch(statusUrl, {
      headers: { 'Authorization': 'Key ' + FAL_KEY }
    }).then(function(r) {
      if (!r.ok) return r.text().then(function(t) { throw new Error('fal status ' + r.status + ': ' + t); });
      return r.json();
    });
  })
  .then(function(s) {
    log('  status: ' + s.status + ' (poll #' + attempt + ')');
    if (s.status === 'COMPLETED') {
      // Fetch the actual result
      return fetch(responseUrl, {
        headers: { 'Authorization': 'Key ' + FAL_KEY }
      }).then(function(r) {
        if (!r.ok) return r.text().then(function(t) { throw new Error('fal result ' + r.status + ': ' + t); });
        return r.json();
      });
    }
    if (s.status === 'FAILED') throw new Error('fal job FAILED: ' + JSON.stringify(s));
    if (s.status === 'IN_QUEUE') setStatus('<span class="sp"></span> In queue...', '');
    if (s.status === 'IN_PROGRESS') setStatus('<span class="sp"></span> Generating...', '');
    return falPoll(statusUrl, responseUrl, attempt + 1);
  });
}

function runFalModel(endpoint, body) {
  setStatus('<span class="sp"></span> Submitting...', '');
  return falSubmit(endpoint, body)
  .then(function(data) {
    log('Job: ' + data.request_id, 'b');
    log('Status URL: ' + data.status_url, 'b');
    log('Response URL: ' + data.response_url, 'b');
    setStatus('<span class="sp"></span> Processing...', '');
    // Use the server-provided URLs instead of building them manually
    return falPoll(data.status_url, data.response_url, 0);
  });
}

// ============================================================
// RUNPOD API (for FASHN)
// ============================================================
function runRunpod() {
  var steps = parseInt(document.getElementById('selSteps').value);
  var seed = parseInt(document.getElementById('selSeed').value);
  var cat = document.getElementById('selCat').value;

  setStatus('<span class="sp"></span> Submitting to RunPod...', '');
  log('POST api.runpod.ai/v2/' + RP_EP + '/run');

  return fetch('https://api.runpod.ai/v2/' + RP_EP + '/run', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + RP_KEY, 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: {
      person_image: personB64, garment_image: garmentB64,
      category: cat, garment_type: 'flat-lay', steps: steps, seed: seed
    }})
  })
  .then(function(r) {
    if (!r.ok) return r.text().then(function(t) { throw new Error('RunPod: ' + t); });
    return r.json();
  })
  .then(function(job) {
    log('RunPod job: ' + job.id, 'b');
    return pollRunpod(job.id, 0);
  });
}

function pollRunpod(jobId, attempt) {
  if (attempt > 90) throw new Error('RunPod timeout');
  return new Promise(function(r) { setTimeout(r, 3000); })
  .then(function() {
    return fetch('https://api.runpod.ai/v2/' + RP_EP + '/status/' + jobId, {
      headers: { 'Authorization': 'Bearer ' + RP_KEY }
    }).then(function(r) { return r.json(); });
  })
  .then(function(s) {
    log('  RunPod: ' + s.status);
    if (s.status === 'COMPLETED') {
      if (s.output && s.output.image) return 'data:image/png;base64,' + s.output.image;
      if (s.output && s.output.error) throw new Error(s.output.error);
      throw new Error('No image in RunPod output');
    }
    if (s.status === 'IN_QUEUE') setStatus('<span class="sp"></span> Worker starting (' + (attempt*3) + 's)...', '');
    if (s.status === 'IN_PROGRESS') setStatus('<span class="sp"></span> Generating (' + (attempt*3) + 's)...', '');
    if (s.status === 'FAILED' || s.status === 'TIMED_OUT') throw new Error('RunPod ' + s.status);
    return pollRunpod(jobId, attempt + 1);
  });
}

// ============================================================
// MAIN
// ============================================================
function doTryOn() {
  if (!personDataUri) { setStatus('Upload a person photo first.', 'err'); return; }
  if (!garmentDataUri) { setStatus('Upload a garment image first.', 'err'); return; }

  var mk = document.getElementById('selModel').value;
  var steps = parseInt(document.getElementById('selSteps').value);
  var seed = parseInt(document.getElementById('selSeed').value);
  var cat = document.getElementById('selCat').value;
  var btn = document.getElementById('btnGo');

  btn.disabled = true; btn.textContent = 'Processing...';
  document.getElementById('resultImg').style.display = 'none';
  document.getElementById('resultEmpty').style.display = 'block';
  document.getElementById('btnDl').style.display = 'none';

  log('=== ' + mk + ' | steps=' + steps + ' seed=' + seed + ' cat=' + cat + ' ===', 'y');

  var promise;

  if (mk === 'fashn') {
    if (RP_EP === 'YOUR_RUNPOD_ENDPOINT_ID') { setStatus('Set RUNPOD_ENDPOINT_ID', 'err'); btn.disabled=false; btn.textContent='Try On'; return; }
    promise = runRunpod().then(function(url) { showResult(url); });
  }
  else {
    if (FAL_KEY === 'YOUR_FAL_KEY') { setStatus('Set FAL_KEY', 'err'); btn.disabled=false; btn.textContent='Try On'; return; }

    var endpoint, body, outputType;

    if (mk === 'catvton') {
      endpoint = 'fal-ai/cat-vton';
      body = {
        human_image_url: personDataUri,
        garment_image_url: garmentDataUri,
        cloth_type: {tops:'upper',bottoms:'lower',dresses:'overall'}[cat] || 'upper',
        num_inference_steps: steps,
        guidance_scale: 2.5,
        seed: seed
      };
      outputType = 'single'; // result.image.url
    }
    else if (mk === 'kling') {
      endpoint = 'fal-ai/kling/v1-5/kolors-virtual-try-on';
      body = {
        human_image_url: personDataUri,
        garment_image_url: garmentDataUri
      };
      outputType = 'single';
    }
    else if (mk === 'nbp') {
      endpoint = 'fal-ai/nano-banana-pro/edit';
      body = {
        prompt: "Put the garment from image 2 onto the person in image 1. Keep the person's face, body shape, pose, and background exactly the same. Only replace their clothing.",
        image_urls: [personDataUri, garmentDataUri],
        num_images: 1,
        aspect_ratio: '3:4',
        output_format: 'png',
        resolution: '1K',
        seed: seed
      };
      outputType = 'array'; // result.images[0].url
    }
    else if (mk === 'qwen') {
      endpoint = 'fal-ai/qwen-image-max/edit';
      body = {
        prompt: "Replace the clothing on the person in image 1 with the garment shown in image 2. Keep the person's face, body, pose, and background unchanged.",
        image_urls: [personDataUri, garmentDataUri],
        num_images: 1,
        output_format: 'png',
        seed: seed
      };
      outputType = 'array';
    }

    log('Endpoint: ' + endpoint);
    log('Body keys: ' + Object.keys(body).join(', '));

    promise = runFalModel(endpoint, body).then(function(result) {
      log('Result keys: ' + Object.keys(result).join(', '), 'b');
      var url;
      if (outputType === 'single') {
        url = result.image.url;
      } else {
        url = result.images[0].url;
      }
      log('Image URL: ' + url, 'g');
      showResult(url);
    });
  }

  promise.catch(function(err) {
    log('ERROR: ' + (err.message || err), 'r');
    setStatus(String(err.message || err).substring(0, 150), 'err');
  }).finally(function() {
    btn.disabled = false;
    btn.textContent = 'Try On';
  });
}

// ---- INIT ----
log('Page loaded', 'g');
onModelChange();

// Check if running from file:// (will cause CORS errors)
if (window.location.protocol === 'file:') {
  document.getElementById('fileWarning').style.display = 'block';
  log('WARNING: Opened from file://. API calls will fail due to CORS.', 'r');
  log('Run: cd VTON && python3 -m http.server 8080', 'y');
  log('Then open: http://localhost:8080/public/', 'y');
  setStatus('Serve this file from localhost — see warning above.', 'err');
}

if (FAL_KEY === 'YOUR_FAL_KEY') {
  log('FAL_KEY not set. Check .env file or config.js', 'y');
  setStatus('Configure API keys first. See README.md', 'err');
}
