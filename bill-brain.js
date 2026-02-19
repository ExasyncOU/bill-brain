/**
 * BILL - B-Intelligent Neural System
 * Real Neural Network Engine v5.0
 * Particle System | Bloom | Hebbian Learning | Cascade Inference
 * (c) 2026 Exasync OUe
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

// ============================================================
// CONFIG
// ============================================================
const SUPABASE_URL = 'https://crslpxgwxjmovrhyxiim.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNyc2xweGd3eGptb3ZyaHl4aWltIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzAzMjAyNTQsImV4cCI6MjA4NTg5NjI1NH0.LMFDBk3a6IaDGw7-GYBkwONJPDPFnMnGa4N_cFpsTKg';
const EDGE_FN_URL = `${SUPABASE_URL}/functions/v1/bill-brain-api`;

const REGION_HUES = {
  prefrontal:     { h: 0.82, s: 0.90, l: 0.58 },  // magenta-purple (strategy)
  frontal:        { h: 0.60, s: 0.95, l: 0.52 },  // blue (execution)
  temporal_left:  { h: 0.48, s: 0.85, l: 0.50 },  // teal (language)
  temporal_right: { h: 0.44, s: 0.80, l: 0.48 },  // cyan-green (analysis)
  parietal:       { h: 0.72, s: 0.85, l: 0.55 },  // indigo (integration)
  occipital:      { h: 0.78, s: 0.75, l: 0.52 },  // violet (visualization)
  thalamus:       { h: 0.12, s: 1.00, l: 0.55 },  // gold (central router)
  hippocampus:    { h: 0.36, s: 0.80, l: 0.48 },  // green (memory)
  amygdala:       { h: 0.02, s: 0.85, l: 0.55 },  // red-orange (risk)
  cerebellum:     { h: 0.30, s: 0.75, l: 0.45 },  // lime (optimization)
  brainstem:      { h: 0.58, s: 0.35, l: 0.38 },  // steel blue (infrastructure)
  motor_cortex:   { h: 0.55, s: 1.00, l: 0.55 }   // electric blue (actions)
};

// Agent name → brain region mapping (for live activity visualization)
const AGENT_REGION_MAP = {
  'Atlas': 'prefrontal', 'Hermes': 'frontal', 'Plutus': 'temporal_right',
  'Apollo': 'temporal_left', 'Prometheus': 'parietal', 'Hephaestus': 'motor_cortex',
  'Athena': 'occipital', 'Hera': 'thalamus', 'Pheme': 'hippocampus',
  'Nike': 'prefrontal', 'Eirene': 'prefrontal', 'Arete': 'prefrontal',
  'Chrysos': 'temporal_right', 'Eunomia': 'temporal_right', 'Ploutos': 'temporal_right',
  'Peitho': 'temporal_left', 'Kalliope': 'temporal_left', 'Techne': 'motor_cortex',
  'Metis': 'parietal', 'Kairos': 'parietal', 'Daedalus': 'motor_cortex',
  'Harmonia': 'frontal', 'Automatos': 'frontal', 'Eupraxia': 'frontal',
  'Morpheus': 'cerebellum', 'Tyche': 'amygdala', 'Themis': 'amygdala',
  'Philotes': 'hippocampus', 'Homonoia': 'thalamus'
};

// ============================================================
// STATE
// ============================================================
let scene, camera, renderer, controls, composer;
let brainGroup;
let neuronMeshes = {};
let synapseMeshes = {}, synapseGlows = {};
let energyParticles = [];
let regionMeshes = {}, labelSprites = [];
let regions = [], neurons = [], synapses = [];
let autoRotate = false, showLabels = true;
let raycaster, mouse;
let clock = new THREE.Clock();
let fireRate = 0, fireCount = 0, lastFireCheck = 0;
let glowTexture;
let hoveredNeuron = null;
let hoverLabel = null;

// Live agent activity state
let lastActivityTime = new Date().toISOString();
let liveConnected = false;
let liveActivityCount = 0;

// SOM attraction + heat visualization
let neuronHeat = {};
let ripples = [];
const BRAIN_RADIUS = 2.7;

// Cross-section state
let crossSectionActive = false;
let hemispheresSplit = false;
let clipPlane = null;
let clipAxis = new THREE.Vector3(0, 0, -1);
let clipOffset = 4.0;
let internalGroup = null;
let crossSectionDisc = null;
let thalamusRoutes = [];
let thalamusGlowSprite = null;
let hemiAnimating = false;

// ============================================================
// WEBSOCKET LIVE MODE STATE
// ============================================================
let wsLiveMode = false;       // Toggle: true = live data from PyTorch, false = demo/simulation
let wsConnection = null;      // WebSocket instance
let wsReconnectTimer = null;  // Reconnect timer
let wsLastState = null;       // Last received state from server
let wsLiveBuilt = false;      // Whether live neurons have been built
let wsLiveNeuronMeshes = {};  // Separate meshes for live neurons (legacy, used for < 1024)
let wsLiveSynapseMeshes = []; // Live synapse lines
let wsLiveSynapseGlows = [];  // Live synapse glow sprites
let wsLiveParticles = [];     // Live energy particles
let wsDemoHidden = false;     // Whether demo objects are hidden
const WS_URL = 'ws://localhost:8765/ws/brain';
const WS_RECONNECT_DELAY = 3000;  // 3 seconds

// ============================================================
// INSTANCED RENDERING (for 1K+ neurons)
// ============================================================
let wsInstancedMesh = null;        // THREE.InstancedMesh for all neurons
let wsInstancedCount = 0;          // Current neuron count
let wsInstancedColors = null;      // Float32Array for per-instance colors
let wsInstancedScales = null;      // Float32Array for per-instance scales
let wsInstancedNeuronData = [];    // Cached neuron metadata
let wsSynapseLineSegments = null;  // Single THREE.LineSegments for all synapses
let wsSynapsePositionBuffer = null;// Float32Array for synapse vertex positions
let wsSynapseColorBuffer = null;   // Float32Array for synapse vertex colors
const WS_INSTANCED_THRESHOLD = 512; // Use instanced rendering above this count
const _tmpMatrix = new THREE.Matrix4();
const _tmpColor = new THREE.Color();
const _tmpVec = new THREE.Vector3();

// ============================================================
// WEBSOCKET CLIENT
// ============================================================

/**
 * Connect to the PyTorch backend WebSocket server.
 * In "Live Mode", neuron positions and activations come from the real neural network.
 * On disconnect, automatically falls back to "Demo Mode".
 */
function wsConnect() {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) return;

  try {
    wsConnection = new WebSocket(WS_URL);

    wsConnection.onopen = () => {
      console.log('[Bill WS] Connected to PyTorch backend');
      dbg('LIVE MODE: Connected to PyTorch');
      wsLiveMode = true;
      wsUpdateToggleUI();
    };

    wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'state') {
          wsLastState = data;
          if (wsLiveMode) {
            wsApplyState(data);
          }
        } else if (data.type === 'heartbeat') {
          console.log('[Bill WS] Heartbeat, clients:', data.clients);
        }
      } catch (e) {
        console.warn('[Bill WS] Parse error:', e);
      }
    };

    wsConnection.onclose = () => {
      console.log('[Bill WS] Disconnected');
      wsLiveMode = false;
      wsConnection = null;
      wsUpdateToggleUI();
      dbg('DEMO MODE: WebSocket disconnected');
      // Auto-reconnect
      if (!wsReconnectTimer) {
        wsReconnectTimer = setTimeout(() => {
          wsReconnectTimer = null;
          wsConnect();
        }, WS_RECONNECT_DELAY);
      }
    };

    wsConnection.onerror = (err) => {
      console.warn('[Bill WS] Error:', err);
      // onclose will fire after onerror
    };
  } catch (e) {
    console.warn('[Bill WS] Connection failed:', e);
    wsLiveMode = false;
    wsUpdateToggleUI();
  }
}

/**
 * Apply live state from PyTorch backend to Three.js scene.
 * Automatically chooses instanced rendering for 1K+ neurons.
 */
function wsApplyState(state) {
  if (!state.neurons || !Array.isArray(state.neurons)) return;

  const useInstanced = state.neurons.length >= WS_INSTANCED_THRESHOLD;

  // --- First call: build live neurons ---
  if (!wsLiveBuilt) {
    wsHideDemoObjects();
    if (useInstanced) {
      wsBuildInstancedNeurons(state);
    } else {
      wsBuildLiveNeurons(state);
    }
    wsLiveBuilt = true;
    dbg(`LIVE: ${state.neurons.length} neurons (${useInstanced ? 'instanced' : 'mesh'}) | ${(state.synapses||[]).length} synapses`);
  }

  if (useInstanced) {
    wsUpdateInstancedNeurons(state);
    wsUpdateInstancedSynapses(state);
  } else {
    // --- Legacy per-mesh update for small networks ---
    state.neurons.forEach(n => {
      const mesh = wsLiveNeuronMeshes[n.id];
      if (!mesh) return;

      const target = new THREE.Vector3(n.x, n.y, n.z);
      if (target.length() > BRAIN_RADIUS) {
        target.normalize().multiplyScalar(BRAIN_RADIUS);
      }
      mesh.position.lerp(target, 0.2);

      const hue = REGION_HUES[n.region] || { h: 0.6, s: 0.8, l: 0.5 };
      const heat = n.heat || 0;
      mesh.material.color.setHSL(hue.h, hue.s, 0.25 + heat * 0.55);
      mesh.material.opacity = 0.4 + heat * 0.6;
      const scale = 0.6 + heat * 0.8;
      mesh.scale.setScalar(scale);

      const glow = mesh.userData.glowSprite;
      if (glow) {
        glow.material.opacity = heat * 0.45;
        const gs = 0.06 + heat * 0.15;
        glow.scale.set(gs, gs, 1);
      }
      neuronHeat[`live_${n.id}`] = heat * 15;
    });

    if (state.synapses && Array.isArray(state.synapses)) {
      wsUpdateLiveSynapses(state);
    }
  }

  // --- Update metrics HUD ---
  if (state.metrics) {
    wsUpdateMetricsHUD(state.metrics);
  }
}

/**
 * Build instanced neuron mesh for 1K+ neurons (1 draw call).
 */
function wsBuildInstancedNeurons(state) {
  const count = state.neurons.length;
  wsInstancedCount = count;
  wsInstancedNeuronData = state.neurons;

  // Sphere geometry shared by all instances (low-poly for performance)
  const geo = new THREE.SphereGeometry(0.015, 6, 4);
  const mat = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.85,
    depthWrite: false,
  });

  wsInstancedMesh = new THREE.InstancedMesh(geo, mat, count);
  wsInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

  // Per-instance color attribute
  wsInstancedColors = new Float32Array(count * 3);
  wsInstancedScales = new Float32Array(count);

  // Initialize positions and colors
  for (let i = 0; i < count; i++) {
    const n = state.neurons[i];
    let px = n.x, py = n.y, pz = n.z;
    const d = Math.sqrt(px*px + py*py + pz*pz);
    if (d > BRAIN_RADIUS) {
      const s = BRAIN_RADIUS / d;
      px *= s; py *= s; pz *= s;
    }

    const heat = n.heat || 0;
    const scale = 0.6 + heat * 0.8;
    wsInstancedScales[i] = scale;

    _tmpMatrix.makeScale(scale, scale, scale);
    _tmpMatrix.setPosition(px, py, pz);
    wsInstancedMesh.setMatrixAt(i, _tmpMatrix);

    const hue = REGION_HUES[n.region] || { h: 0.6, s: 0.8, l: 0.5 };
    _tmpColor.setHSL(hue.h, hue.s, 0.25 + heat * 0.55);
    wsInstancedMesh.setColorAt(i, _tmpColor);
    wsInstancedColors[i*3] = _tmpColor.r;
    wsInstancedColors[i*3+1] = _tmpColor.g;
    wsInstancedColors[i*3+2] = _tmpColor.b;
  }

  wsInstancedMesh.instanceMatrix.needsUpdate = true;
  wsInstancedMesh.instanceColor.needsUpdate = true;
  brainGroup.add(wsInstancedMesh);

  // Build initial synapses
  if (state.synapses && state.synapses.length > 0) {
    wsBuildInstancedSynapses(state);
  }
}

/**
 * Batch-update all instanced neuron positions and colors (every WS frame).
 */
function wsUpdateInstancedNeurons(state) {
  if (!wsInstancedMesh) return;

  // Build position lookup from previous frame for lerping
  const neurons = state.neurons;
  const count = Math.min(neurons.length, wsInstancedCount);

  for (let i = 0; i < count; i++) {
    const n = neurons[i];
    let px = n.x, py = n.y, pz = n.z;
    const d = Math.sqrt(px*px + py*py + pz*pz);
    if (d > BRAIN_RADIUS) {
      const s = BRAIN_RADIUS / d;
      px *= s; py *= s; pz *= s;
    }

    // Get current position for lerp
    wsInstancedMesh.getMatrixAt(i, _tmpMatrix);
    _tmpVec.setFromMatrixPosition(_tmpMatrix);
    _tmpVec.lerp(new THREE.Vector3(px, py, pz), 0.15);

    const heat = n.heat || 0;
    const scale = 0.5 + heat * 1.0;

    _tmpMatrix.makeScale(scale, scale, scale);
    _tmpMatrix.setPosition(_tmpVec.x, _tmpVec.y, _tmpVec.z);
    wsInstancedMesh.setMatrixAt(i, _tmpMatrix);

    // Color: shift from region hue toward white as heat increases
    const hue = REGION_HUES[n.region] || { h: 0.6, s: 0.8, l: 0.5 };
    const h = hue.h + heat * (0.12 - hue.h) * 0.3;
    const s = hue.s * (1 - heat * 0.5);
    const l = 0.2 + heat * 0.65;
    _tmpColor.setHSL(h, s, l);
    wsInstancedMesh.setColorAt(i, _tmpColor);
  }

  wsInstancedMesh.instanceMatrix.needsUpdate = true;
  if (wsInstancedMesh.instanceColor) wsInstancedMesh.instanceColor.needsUpdate = true;

  // Cache neuron data for synapse updates
  wsInstancedNeuronData = neurons;
}

/**
 * Build LineSegments geometry for all synapses (one draw call).
 */
function wsBuildInstancedSynapses(state) {
  const synapses = state.synapses || [];
  const maxSyn = Math.min(synapses.length, 2000);

  // 2 vertices per line segment, 3 floats per vertex
  wsSynapsePositionBuffer = new Float32Array(maxSyn * 2 * 3);
  wsSynapseColorBuffer = new Float32Array(maxSyn * 2 * 3);

  // Build neuron position lookup
  const posMap = {};
  state.neurons.forEach(n => {
    let px = n.x, py = n.y, pz = n.z;
    const d = Math.sqrt(px*px + py*py + pz*pz);
    if (d > BRAIN_RADIUS) {
      const s = BRAIN_RADIUS / d;
      px *= s; py *= s; pz *= s;
    }
    posMap[n.id] = { x: px, y: py, z: pz, region: n.region };
  });

  let idx = 0;
  for (let s = 0; s < maxSyn; s++) {
    const syn = synapses[s];
    const from = posMap[syn.from];
    const to = posMap[syn.to];
    if (!from || !to) continue;

    const w = Math.abs(syn.weight);
    const hue = REGION_HUES[from.region] || { h: 0.6, s: 0.8, l: 0.5 };
    const opacity = syn.active ? 0.15 + w * 0.6 : 0.02 + w * 0.12;
    _tmpColor.setHSL(hue.h, syn.active ? 1.0 : hue.s * 0.5, syn.active ? 0.5 : 0.15 + w * 0.2);

    const i6 = idx * 6;
    wsSynapsePositionBuffer[i6] = from.x;
    wsSynapsePositionBuffer[i6+1] = from.y;
    wsSynapsePositionBuffer[i6+2] = from.z;
    wsSynapsePositionBuffer[i6+3] = to.x;
    wsSynapsePositionBuffer[i6+4] = to.y;
    wsSynapsePositionBuffer[i6+5] = to.z;

    wsSynapseColorBuffer[i6] = _tmpColor.r * opacity;
    wsSynapseColorBuffer[i6+1] = _tmpColor.g * opacity;
    wsSynapseColorBuffer[i6+2] = _tmpColor.b * opacity;
    wsSynapseColorBuffer[i6+3] = _tmpColor.r * opacity;
    wsSynapseColorBuffer[i6+4] = _tmpColor.g * opacity;
    wsSynapseColorBuffer[i6+5] = _tmpColor.b * opacity;
    idx++;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(wsSynapsePositionBuffer, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(wsSynapseColorBuffer, 3));
  geo.setDrawRange(0, idx * 2);

  const mat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 1.0,
    depthWrite: false, blending: THREE.AdditiveBlending
  });

  wsSynapseLineSegments = new THREE.LineSegments(geo, mat);
  brainGroup.add(wsSynapseLineSegments);
}

/**
 * Update synapse positions and colors from live state.
 */
function wsUpdateInstancedSynapses(state) {
  if (!wsSynapseLineSegments || !state.synapses) return;

  const synapses = state.synapses;
  const maxSyn = Math.min(synapses.length, 2000);

  // Build position lookup from current neuron data
  const posMap = {};
  (state.neurons || wsInstancedNeuronData).forEach(n => {
    let px = n.x, py = n.y, pz = n.z;
    const d = Math.sqrt(px*px + py*py + pz*pz);
    if (d > BRAIN_RADIUS) {
      const s = BRAIN_RADIUS / d;
      px *= s; py *= s; pz *= s;
    }
    posMap[n.id] = { x: px, y: py, z: pz, region: n.region };
  });

  const posArr = wsSynapseLineSegments.geometry.attributes.position.array;
  const colArr = wsSynapseLineSegments.geometry.attributes.color.array;

  let idx = 0;
  for (let s = 0; s < maxSyn; s++) {
    const syn = synapses[s];
    const from = posMap[syn.from];
    const to = posMap[syn.to];
    if (!from || !to) continue;

    const w = Math.abs(syn.weight);
    const hue = REGION_HUES[from.region] || { h: 0.6, s: 0.8, l: 0.5 };
    const opacity = syn.active ? 0.15 + w * 0.6 : 0.02 + w * 0.12;
    _tmpColor.setHSL(hue.h, syn.active ? 1.0 : hue.s * 0.5, syn.active ? 0.5 : 0.15 + w * 0.2);

    const i6 = idx * 6;
    posArr[i6] = from.x; posArr[i6+1] = from.y; posArr[i6+2] = from.z;
    posArr[i6+3] = to.x;  posArr[i6+4] = to.y;  posArr[i6+5] = to.z;

    colArr[i6] = _tmpColor.r * opacity;   colArr[i6+1] = _tmpColor.g * opacity; colArr[i6+2] = _tmpColor.b * opacity;
    colArr[i6+3] = _tmpColor.r * opacity;  colArr[i6+4] = _tmpColor.g * opacity; colArr[i6+5] = _tmpColor.b * opacity;
    idx++;
  }

  wsSynapseLineSegments.geometry.setDrawRange(0, idx * 2);
  wsSynapseLineSegments.geometry.attributes.position.needsUpdate = true;
  wsSynapseLineSegments.geometry.attributes.color.needsUpdate = true;
}

/**
 * Hide demo/Supabase objects when entering live mode.
 */
function wsHideDemoObjects() {
  if (wsDemoHidden) return;
  // Hide demo neuron meshes
  Object.values(neuronMeshes).forEach(m => { m.visible = false; });
  // Hide demo synapse meshes
  Object.values(synapseMeshes).forEach(m => { m.visible = false; });
  Object.values(synapseGlows).forEach(m => { m.visible = false; });
  // Hide energy particles
  energyParticles.forEach(p => { p.visible = false; });
  // Hide labels
  labelSprites.forEach(s => { s.visible = false; });
  wsDemoHidden = true;
}

/**
 * Show demo objects again when leaving live mode.
 */
function wsShowDemoObjects() {
  if (!wsDemoHidden) return;
  Object.values(neuronMeshes).forEach(m => { m.visible = true; });
  Object.values(synapseMeshes).forEach(m => { m.visible = true; });
  Object.values(synapseGlows).forEach(m => { m.visible = true; });
  energyParticles.forEach(p => { p.visible = true; });
  labelSprites.forEach(s => { if (showLabels) s.visible = true; });
  wsDemoHidden = false;
}

/**
 * Build 256 live neuron meshes from PyTorch state.
 */
function wsBuildLiveNeurons(state) {
  const geo = new THREE.SphereGeometry(0.02, 8, 6);

  state.neurons.forEach(n => {
    const hue = REGION_HUES[n.region] || { h: 0.6, s: 0.8, l: 0.5 };
    const heat = n.heat || 0;
    const color = new THREE.Color().setHSL(hue.h, hue.s * 0.9, 0.25 + heat * 0.5);

    // Neuron core
    const core = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.5 + heat * 0.5
    }));

    let px = n.x, py = n.y, pz = n.z;
    const dist = Math.sqrt(px*px + py*py + pz*pz);
    if (dist > BRAIN_RADIUS) {
      const s = BRAIN_RADIUS / dist;
      px *= s; py *= s; pz *= s;
    }
    core.position.set(px, py, pz);
    core.userData = { type: 'live-neuron', neuronId: n.id, region: n.region };

    // Glow sprite per neuron (small, elegant)
    const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture,
      color: new THREE.Color().setHSL(hue.h, 1.0, 0.6),
      transparent: true,
      opacity: heat * 0.35,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    }));
    const gs = 0.05 + heat * 0.12;
    glowSprite.scale.set(gs, gs, 1);
    core.add(glowSprite);
    core.userData.glowSprite = glowSprite;

    brainGroup.add(core);
    wsLiveNeuronMeshes[n.id] = core;
  });
}

/**
 * Update live synapse lines from PyTorch state.
 * Rebuilds each frame for simplicity (synapses change as weights change).
 */
function wsUpdateLiveSynapses(state) {
  // Remove old live synapses
  wsLiveSynapseMeshes.forEach(m => { brainGroup.remove(m); m.geometry.dispose(); m.material.dispose(); });
  wsLiveSynapseGlows.forEach(m => { brainGroup.remove(m); m.material.dispose(); });
  wsLiveParticles.forEach(p => { brainGroup.remove(p); p.material.dispose(); });
  wsLiveSynapseMeshes = [];
  wsLiveSynapseGlows = [];
  wsLiveParticles = [];

  // Build neuron position lookup
  const posMap = {};
  state.neurons.forEach(n => {
    const mesh = wsLiveNeuronMeshes[n.id];
    if (mesh) posMap[n.id] = mesh.position.clone();
  });

  // Only show top synapses for performance (max 500)
  const sortedSynapses = (state.synapses || [])
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 500);

  sortedSynapses.forEach(syn => {
    const startPos = posMap[syn.from];
    const endPos = posMap[syn.to];
    if (!startPos || !endPos) return;

    const w = Math.abs(syn.weight);
    const srcNeuron = state.neurons.find(n => n.id === syn.from);
    const regionName = srcNeuron ? srcNeuron.region : 'frontal';
    const rHue = REGION_HUES[regionName] || { h: 0.6, s: 0.8, l: 0.5 };

    // Color: active synapses glow brighter
    const lineColor = syn.active
      ? new THREE.Color().setHSL(rHue.h, 1.0, 0.5 + w * 0.3)
      : new THREE.Color().setHSL(rHue.h, rHue.s * 0.6, 0.15 + w * 0.25);

    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([startPos, endPos]),
      new THREE.LineBasicMaterial({
        color: lineColor, transparent: true,
        opacity: syn.active ? 0.3 + w * 0.5 : 0.03 + w * 0.15,
        depthWrite: false
      })
    );
    brainGroup.add(line);
    wsLiveSynapseMeshes.push(line);

    // Glow on active strong synapses
    if (syn.active && w > 0.15) {
      const mid = new THREE.Vector3().addVectors(startPos, endPos).multiplyScalar(0.5);
      const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture,
        color: lineColor,
        transparent: true,
        opacity: w * 0.4,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      }));
      glowSprite.position.copy(mid);
      const d = startPos.distanceTo(endPos);
      glowSprite.scale.set(d * 0.3, d * 0.1, 1);
      brainGroup.add(glowSprite);
      wsLiveSynapseGlows.push(glowSprite);
    }

    // Energy particles on strong active synapses
    if (syn.active && w > 0.25 && Math.random() < 0.4) {
      const pColor = new THREE.Color().setHSL(rHue.h, 1.0, 0.7);
      const p = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture, color: pColor, transparent: true, opacity: 0.6,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      p.scale.set(0.08, 0.08, 1);
      const t = Math.random();
      p.position.lerpVectors(startPos, endPos, t);
      brainGroup.add(p);
      wsLiveParticles.push(p);
    }
  });
}

/**
 * Update the metrics HUD with live training data.
 */
function wsUpdateMetricsHUD(m) {
  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };

  if (m.train_accuracy) setEl('stat-firerate', `${(m.train_accuracy * 100).toFixed(1)}%`);
  if (m.epoch) setEl('stat-neurons', `E${m.epoch} | ${wsInstancedCount > 0 ? wsInstancedCount.toLocaleString() : '?'}N`);
  if (m.train_loss) setEl('stat-synapses', `L:${m.train_loss.toFixed(4)}`);
  if (m.positions_drift) setEl('stat-regions', `D:${m.positions_drift.toFixed(3)}`);
}

/**
 * Update the Live/Demo toggle button in the UI.
 */
function wsUpdateToggleUI() {
  const btn = document.getElementById('ws-toggle-btn');
  if (!btn) return;
  btn.textContent = wsLiveMode ? 'LIVE' : 'DEMO';
  btn.style.background = wsLiveMode ? 'rgba(0,255,100,0.3)' : 'rgba(255,100,0,0.3)';
  btn.style.borderColor = wsLiveMode ? '#0f6' : '#f60';
}

/**
 * Toggle between Live and Demo mode.
 */
function wsToggleMode() {
  if (wsLiveMode) {
    // Switch to demo: disconnect WebSocket, remove live meshes
    wsLiveMode = false;
    if (wsConnection) {
      wsConnection.close();
      wsConnection = null;
    }
    if (wsReconnectTimer) {
      clearTimeout(wsReconnectTimer);
      wsReconnectTimer = null;
    }
    // Remove instanced mesh if present
    if (wsInstancedMesh) {
      brainGroup.remove(wsInstancedMesh);
      wsInstancedMesh.geometry.dispose();
      wsInstancedMesh.material.dispose();
      wsInstancedMesh = null;
    }
    // Remove instanced synapses
    if (wsSynapseLineSegments) {
      brainGroup.remove(wsSynapseLineSegments);
      wsSynapseLineSegments.geometry.dispose();
      wsSynapseLineSegments.material.dispose();
      wsSynapseLineSegments = null;
    }
    // Remove legacy live neuron meshes
    Object.values(wsLiveNeuronMeshes).forEach(m => { brainGroup.remove(m); });
    wsLiveNeuronMeshes = {};
    // Remove legacy live synapse meshes
    wsLiveSynapseMeshes.forEach(m => { brainGroup.remove(m); m.geometry.dispose(); m.material.dispose(); });
    wsLiveSynapseGlows.forEach(m => { brainGroup.remove(m); m.material.dispose(); });
    wsLiveParticles.forEach(p => { brainGroup.remove(p); p.material.dispose(); });
    wsLiveSynapseMeshes = [];
    wsLiveSynapseGlows = [];
    wsLiveParticles = [];
    wsLiveBuilt = false;
    wsInstancedCount = 0;
    wsInstancedNeuronData = [];
    // Show demo objects again
    wsShowDemoObjects();
    dbg('DEMO MODE: Simulation active');
  } else {
    // Switch to live: connect WebSocket
    wsConnect();
  }
  wsUpdateToggleUI();
}

// Expose toggle for HTML button onclick
window.wsToggleMode = wsToggleMode;

function dbg(msg) {
  console.log('[Bill]', msg);
  const el = document.getElementById('debug');
  if (el) el.textContent = msg;
}

// ============================================================
// SUPABASE HELPERS
// ============================================================
const sbHeaders = {
  'apikey': SUPABASE_ANON_KEY,
  'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
};

async function sbQuery(table, params = '', timeoutMs = 3000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${params}`, {
      headers: sbHeaders, signal: controller.signal
    });
    clearTimeout(timer);
    return res.json();
  } catch (e) {
    clearTimeout(timer);
    console.warn(`[Bill] sbQuery(${table}) timeout/error:`, e.message);
    return [];
  }
}

async function edgeFn(action, body = {}, timeoutMs = 3000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${EDGE_FN_URL}?action=${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal
    });
    clearTimeout(timer);
    return res.json();
  } catch (e) {
    clearTimeout(timer);
    console.warn(`[Bill] edgeFn(${action}) timeout/error:`, e.message);
    return {};
  }
}

// ============================================================
// GLOW TEXTURE
// ============================================================
function createGlowTexture() {
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  g.addColorStop(0, 'rgba(100,180,255,1.0)');
  g.addColorStop(0.15, 'rgba(60,140,255,0.8)');
  g.addColorStop(0.4, 'rgba(30,100,255,0.3)');
  g.addColorStop(0.7, 'rgba(10,60,200,0.08)');
  g.addColorStop(1, 'rgba(0,30,120,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}

// ============================================================
// INIT
// ============================================================
async function init() {
  try {
    dbg('Initializing...');
    glowTexture = createGlowTexture();

    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000510, 0.04);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 200);
    camera.position.set(5, 4, 6);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.5;
    renderer.localClippingEnabled = true;
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Bloom Post-Processing (Scientific Art aesthetic)
    composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.7,   // strength - slightly stronger for 10K neuron density
      0.6,   // radius - wider soft spread for organic neural glow
      0.65   // threshold - catch more subtle glows from sparse neurons
    );
    composer.addPass(bloomPass);

    // Vignette + Film Grain + Chromatic Aberration (cinematic feel)
    const vignetteShader = {
      uniforms: {
        tDiffuse: { value: null },
        time: { value: 0 },
        vignetteStrength: { value: 0.45 },
        grainStrength: { value: 0.04 },
        chromaticStrength: { value: 0.0012 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform float time;
        uniform float vignetteStrength;
        uniform float grainStrength;
        uniform float chromaticStrength;
        varying vec2 vUv;

        float rand(vec2 co) {
          return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
        }

        void main() {
          vec2 uv = vUv;

          // Chromatic aberration (subtle color fringing at edges)
          float dist = distance(uv, vec2(0.5));
          float ca = chromaticStrength * dist;
          float r = texture2D(tDiffuse, uv + vec2(ca, 0.0)).r;
          float g = texture2D(tDiffuse, uv).g;
          float b = texture2D(tDiffuse, uv - vec2(ca, 0.0)).b;
          vec3 color = vec3(r, g, b);

          // Vignette (dark edges)
          float vignette = 1.0 - vignetteStrength * dist * dist * 2.0;
          color *= vignette;

          // Film grain (animated noise)
          float grain = (rand(uv + fract(time * 0.01)) - 0.5) * grainStrength;
          color += grain;

          gl_FragColor = vec4(color, 1.0);
        }
      `
    };
    const vignettePass = new ShaderPass(vignetteShader);
    vignettePass.renderToScreen = true;
    composer.addPass(vignettePass);
    // Store reference for animation update
    window._vignettePass = vignettePass;

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.3;
    controls.minDistance = 2;
    controls.maxDistance = 25;

    raycaster = new THREE.Raycaster();
    raycaster.params.Line = { threshold: 0.15 };
    mouse = new THREE.Vector2();

    setupLighting();
    buildStarfield();

    brainGroup = new THREE.Group();
    scene.add(brainGroup);

    dbg('Loading data...');
    await loadData();
    dbg(`Data: ${regions.length} regions, ${neurons.length} neurons, ${synapses.length} synapses`);

    dbg('Loading brain model...');
    try { await loadBrainOBJ(); dbg('OBJ loaded'); }
    catch (e) { dbg('OBJ failed, using procedural'); buildProceduralBrain(); }

    buildRegions();
    buildNeurons();
    buildSynapses();
    buildLegend();
    buildInternalStructures();
    updateStats();

    window.addEventListener('resize', onResize);
    renderer.domElement.addEventListener('click', onMouseClick);
    renderer.domElement.addEventListener('mousemove', onMouseMove);

    setTimeout(() => document.getElementById('loading').classList.add('hidden'), 800);

    animate();
    setInterval(ambientFire, 150);
    // Periodic decay + attraction (every 30s)
    setInterval(async () => {
      try {
        await edgeFn('decay');
        await refreshPositions();
      } catch(e) { console.warn('Periodic update failed:', e); }
    }, 30000);

    // SOM attraction: neurons drift toward co-firing partners (every 3s)
    setInterval(somAttractionStep, 3000);

    // Live agent activity polling (every 5s)
    pollAgentActivity();
    setInterval(pollAgentActivity, 5000);

    // WebSocket: Try to connect to PyTorch backend (non-blocking)
    // If server is not running, stays in Demo Mode silently
    try { wsConnect(); } catch(e) { console.log('[Bill WS] Server not available, Demo Mode'); }

    dbg('Bill v5.0 - Bloom + Particle System + Live Neural Network');

  } catch (err) {
    console.error('Init failed:', err);
    dbg('ERROR: ' + err.message);
    document.getElementById('loading-text').textContent = 'FEHLER';
    setTimeout(() => document.getElementById('loading').classList.add('hidden'), 2000);
  }
}

// ============================================================
// REFRESH POSITIONS (after attract_neurons)
// ============================================================
async function refreshPositions() {
  try {
    const fresh = await sbQuery('neurons', 'select=id,pos_x,pos_y,pos_z');
    if (!Array.isArray(fresh)) return;
    fresh.forEach(n => {
      const mesh = neuronMeshes[n.id];
      if (mesh) {
        mesh.position.set(n.pos_x, n.pos_y, n.pos_z);
      }
      // Update local data
      const local = neurons.find(ln => ln.id === n.id);
      if (local) { local.pos_x = n.pos_x; local.pos_y = n.pos_y; local.pos_z = n.pos_z; }
    });
    // Update synapse line positions
    rebuildSynapseGeometry();
  } catch(e) { console.warn('Position refresh failed:', e); }
}

function rebuildSynapseGeometry() {
  const nMap = {};
  neurons.forEach(n => { nMap[n.id] = n; });
  synapses.forEach(syn => {
    const line = synapseMeshes[syn.id];
    if (!line) return;
    const src = nMap[syn.source_neuron_id];
    const tgt = nMap[syn.target_neuron_id];
    if (!src || !tgt) return;
    const positions = line.geometry.attributes.position;
    positions.setXYZ(0, src.pos_x, src.pos_y, src.pos_z);
    positions.setXYZ(1, tgt.pos_x, tgt.pos_y, tgt.pos_z);
    positions.needsUpdate = true;
    line.userData.start.set(src.pos_x, src.pos_y, src.pos_z);
    line.userData.end.set(tgt.pos_x, tgt.pos_y, tgt.pos_z);
    // Update glow sprite position (midpoint)
    const glow = synapseGlows[syn.id];
    if (glow) {
      glow.position.set(
        (src.pos_x + tgt.pos_x) / 2,
        (src.pos_y + tgt.pos_y) / 2,
        (src.pos_z + tgt.pos_z) / 2
      );
    }
  });
}

// ============================================================
// STARFIELD
// ============================================================
function buildStarfield() {
  const n = 2000;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    const r = 50 + Math.random() * 100;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    pos[i3] = r * Math.sin(phi) * Math.cos(theta);
    pos[i3+1] = r * Math.sin(phi) * Math.sin(theta);
    pos[i3+2] = r * Math.cos(phi);
    const b = 0.3 + Math.random() * 0.7;
    col[i3] = 0.4 * b; col[i3+1] = 0.5 * b; col[i3+2] = b;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.4, vertexColors: true, transparent: true, opacity: 0.8, sizeAttenuation: true
  })));
}

// ============================================================
// LIGHTING
// ============================================================
function setupLighting() {
  scene.add(new THREE.AmbientLight(0x0a1530, 0.6));
  const main = new THREE.DirectionalLight(0x4488ff, 0.8);
  main.position.set(5, 8, 5); scene.add(main);
  const fill = new THREE.DirectionalLight(0x2244aa, 0.3);
  fill.position.set(-5, -3, -5); scene.add(fill);
  const point = new THREE.PointLight(0x0066ff, 2.5, 25);
  point.position.set(0, 0, 0); scene.add(point);
  scene.add(new THREE.HemisphereLight(0x2244aa, 0x001122, 0.5));
  const rim = new THREE.DirectionalLight(0x0088ff, 0.5);
  rim.position.set(-3, 2, -5); scene.add(rim);
}

// ============================================================
// DATA LOADING
// ============================================================
async function loadData() {
  try {
    const [regRes, neuRes, synRes] = await Promise.all([
      sbQuery('brain_regions', 'select=*'),
      sbQuery('neurons', 'select=*'),
      sbQuery('synapses', 'select=*&limit=2000')
    ]);
    regions = Array.isArray(regRes) ? regRes : [];
    neurons = Array.isArray(neuRes) ? neuRes : [];
    synapses = Array.isArray(synRes) ? synRes : [];
    if (regions.length === 0) throw new Error('No data');
  } catch (e) {
    console.warn('Supabase load failed, using fallback:', e);
    generateFallbackData();
  }
}

function generateFallbackData() {
  const defs = [
    { id:'prefrontal', name:'Prefrontal Cortex', name_de:'Praefrontaler Cortex', center_x:0, center_y:1.8, center_z:2.2, radius:1.2, description:'Planung, Strategie' },
    { id:'frontal', name:'Frontal Lobe', name_de:'Frontallappen', center_x:0, center_y:1.5, center_z:1.0, radius:1.4, description:'Ausfuehrung, Sprache' },
    { id:'temporal_left', name:'Left Temporal', name_de:'Linker Temporallappen', center_x:-2.0, center_y:-0.5, center_z:0, radius:1.1, description:'Sprache, Kommunikation' },
    { id:'temporal_right', name:'Right Temporal', name_de:'Rechter Temporallappen', center_x:2.0, center_y:-0.5, center_z:0, radius:1.1, description:'Muster, Analyse' },
    { id:'parietal', name:'Parietal Lobe', name_de:'Parietallappen', center_x:0, center_y:1.6, center_z:-0.8, radius:1.3, description:'Integration' },
    { id:'occipital', name:'Occipital Lobe', name_de:'Okzipitallappen', center_x:0, center_y:0.5, center_z:-2.2, radius:1.0, description:'Visualisierung' },
    { id:'thalamus', name:'Thalamus', name_de:'Thalamus', center_x:0, center_y:0, center_z:0, radius:0.6, description:'Signal-Router' },
    { id:'hippocampus', name:'Hippocampus', name_de:'Hippocampus', center_x:0, center_y:-0.3, center_z:-0.3, radius:0.5, description:'Gedaechtnis' },
    { id:'amygdala', name:'Amygdala', name_de:'Amygdala', center_x:0, center_y:-0.5, center_z:0.3, radius:0.4, description:'Risikobewertung' },
    { id:'cerebellum', name:'Cerebellum', name_de:'Kleinhirn', center_x:0, center_y:-1.5, center_z:-2.0, radius:1.2, description:'Optimierung' },
    { id:'brainstem', name:'Brainstem', name_de:'Hirnstamm', center_x:0, center_y:-2.0, center_z:-1.0, radius:0.5, description:'Infrastruktur' },
    { id:'motor_cortex', name:'Motor Cortex', name_de:'Motorischer Cortex', center_x:0, center_y:2.0, center_z:0, radius:1.0, description:'Aktionen' }
  ];
  const labels = {
    prefrontal: ['Quartals-Strategie','Ressourcen-Allokation','Risikobewertung','Produktpriorisierung'],
    frontal: ['Task-Delegation','Agent-Koordination','Workflow-Optimierung','Sprint-Planung'],
    temporal_left: ['Kunden-E-Mail-Muster','LinkedIn-Content','Technische-Doku','DE-Markt-Messaging'],
    temporal_right: ['Umsatzmuster-Q1','Lead-Quellen-Analyse','Churn-Signal','Upsell-Timing'],
    parietal: ['Cross-Agent-Sync','API-Integration-Map','Event-Pipeline','Daten-Konsistenz'],
    occipital: ['OrgSphere-Rendering','Dashboard-KPIs','Bill-Gehirn-3D','Report-Generator'],
    thalamus: ['Task-Prioritaets-Router','Agent-Capability-Matcher','Message-Bus','Load-Balancer'],
    hippocampus: ['Kundeninteraktion-History','Erfolgreiche-Deal-Muster','Agent-Performance-Log','Knowledge-Base-Index'],
    amygdala: ['Deadline-Warnsystem','Budget-Ueberlauf','Security-Bedrohung','Kunden-Churn-Risiko'],
    cerebellum: ['Query-Performance','Cache-Optimierung','Token-Kosten-Optimierung','Prompt-Effizienz'],
    brainstem: ['Supabase-Verbindung','GitHub-Actions-Runner','DNS-Health-Check','SSL-Zertifikat-Monitor'],
    motor_cortex: ['API-Executor','File-Deploy','DB-Migration-Runner','Webhook-Dispatcher']
  };
  regions = defs;
  neurons = [];
  synapses = [];
  defs.forEach(r => {
    const rLabels = labels[r.id] || [];
    rLabels.forEach((lbl, i) => {
      const a1 = (i / rLabels.length) * Math.PI * 2 + Math.random() * 0.5;
      const a2 = Math.random() * 0.8 - 0.4;
      const d = 0.3 + Math.random() * r.radius * 0.5;
      let px = r.center_x + d*Math.cos(a1)*Math.cos(a2);
      let py = r.center_y + d*Math.sin(a2);
      let pz = r.center_z + d*Math.sin(a1)*Math.cos(a2);
      // Clamp inside brain surface
      const dist = Math.sqrt(px*px + py*py + pz*pz);
      if (dist > BRAIN_RADIUS) {
        const s = BRAIN_RADIUS / dist;
        px *= s; py *= s; pz *= s;
      }
      neurons.push({
        id: `${r.id}_${i}`, region_id: r.id, label: lbl,
        neuron_type: 'concept', description: '',
        pos_x: px, pos_y: py, pos_z: pz,
        activation: 0, success_rate: 0.5, fire_count: 0
      });
    });
  });
  // Create intra-region synapses
  for (let i = 0; i < neurons.length; i++) {
    for (let j = i+1; j < neurons.length; j++) {
      if (neurons[i].region_id === neurons[j].region_id) {
        synapses.push({ id:`syn_${i}_${j}`, source_neuron_id:neurons[i].id, target_neuron_id:neurons[j].id, weight:0.2+Math.random()*0.3 });
      }
    }
  }
}

// ============================================================
// BRAIN MESH SEGMENTATION (Anatomical Vertex Coloring)
// ============================================================
function segmentBrainMesh(mesh) {
  const geometry = mesh.geometry;
  const positions = geometry.attributes.position;
  const count = positions.count;
  const colors = new Float32Array(count * 3);

  // Build region center map with wide sigma for full surface coverage
  const rCenters = {};
  regions.forEach(r => {
    rCenters[r.id] = {
      v: new THREE.Vector3(r.center_x, r.center_y, r.center_z),
      radius: r.radius || 1.0
    };
  });

  const tmpColor = new THREE.Color();
  for (let i = 0; i < count; i++) {
    const x = positions.getX(i), y = positions.getY(i), z = positions.getZ(i);
    const pos = new THREE.Vector3(x, y, z);

    // Gaussian blend with WIDE sigma → every vertex gets colored, organic boundaries
    let totalW = 0, cr = 0, cg = 0, cb = 0;
    regions.forEach(region => {
      const rc = rCenters[region.id];
      if (!rc) return;
      const dist = pos.distanceTo(rc.v);
      // Wide sigma ensures full brain surface coverage with smooth transitions
      const sigma = Math.max(rc.radius * 1.8, 1.5);
      const w = Math.exp(-(dist * dist) / (2 * sigma * sigma));
      if (w < 0.001) return;
      const hue = REGION_HUES[region.id] || { h: 0.6, s: 0.8, l: 0.5 };
      tmpColor.setHSL(hue.h, hue.s, hue.l);
      cr += tmpColor.r * w;
      cg += tmpColor.g * w;
      cb += tmpColor.b * w;
      totalW += w;
    });

    if (totalW > 0) {
      colors[i * 3] = cr / totalW;
      colors[i * 3 + 1] = cg / totalW;
      colors[i * 3 + 2] = cb / totalW;
    } else {
      // Default: deep blue for uncovered areas
      colors[i * 3] = 0.05; colors[i * 3 + 1] = 0.1; colors[i * 3 + 2] = 0.3;
    }
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  mesh.material = new THREE.MeshPhysicalMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.30,
    roughness: 0.15,
    metalness: 0.0,
    clearcoat: 0.4,
    clearcoatRoughness: 0.1,
    side: THREE.FrontSide,
    depthWrite: false
  });
}

// ============================================================
// LOAD OBJ BRAIN MODEL
// ============================================================
async function loadBrainOBJ() {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => { buildProceduralBrain(); resolve(); }, 10000);
    const loader = new OBJLoader();
    loader.load('Free_Brain.obj',
      (obj) => {
        clearTimeout(timeout);
        try {
          // Find the largest mesh (= brain), hide smaller studio junk
          let largestMesh = null;
          let largestCount = 0;
          obj.traverse(child => {
            if (child.isMesh) {
              const vCount = child.geometry.attributes.position?.count || 0;
              if (vCount > largestCount) {
                largestCount = vCount;
                largestMesh = child;
              }
            }
          });

          // Hide all non-brain meshes (studio watermarks)
          obj.traverse(child => {
            if (child.isMesh && child !== largestMesh) {
              child.visible = false;
            }
          });

          // Center and scale the brain mesh
          if (largestMesh) {
            largestMesh.geometry.computeBoundingBox();
            const bbox = largestMesh.geometry.boundingBox;
            const center = new THREE.Vector3(); bbox.getCenter(center);
            const size = new THREE.Vector3(); bbox.getSize(size);
            const s = 6.0 / Math.max(size.x, size.y, size.z);
            largestMesh.geometry.translate(-center.x, -center.y, -center.z);
            largestMesh.geometry.scale(s, s, s);
            segmentBrainMesh(largestMesh);
          }

          brainGroup.add(obj);
          const glow = new THREE.Sprite(new THREE.SpriteMaterial({
            map: glowTexture, color: 0x0066ff, transparent: true,
            opacity: 0.18, blending: THREE.AdditiveBlending, depthWrite: false
          }));
          glow.scale.set(13, 13, 1);
          brainGroup.add(glow);
          resolve();
        } catch (e) { buildProceduralBrain(); resolve(); }
      },
      (xhr) => {
        if (xhr.total > 0)
          document.getElementById('loading-text').textContent = `BILL ${(xhr.loaded/xhr.total*100).toFixed(0)}%`;
      },
      () => { clearTimeout(timeout); buildProceduralBrain(); resolve(); }
    );
  });
}

// ============================================================
// PROCEDURAL BRAIN
// ============================================================
function buildProceduralBrain() {
  const geo = new THREE.SphereGeometry(3.2, 64, 48);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    x *= 0.85; z *= 1.1;
    if (y > 0) y *= 1.1;
    if (y < -0.5) { const f = 1.0 - Math.abs(y+0.5)*0.15; x *= f; z *= f; }
    if (y > 1.0) y -= Math.exp(-x*x*8) * 0.15;
    const bulge = Math.exp(-(y+0.5)*(y+0.5)*2) * 0.2;
    x *= 1 + bulge * Math.sign(x);
    const n = (Math.sin(x*5+y*3)*Math.cos(z*4+x*2)) * 0.05;
    pos.setXYZ(i, x+n, y+n*0.5, z+n);
  }
  geo.computeVertexNormals();
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x0066cc, transparent: true, opacity: 0.12,
    roughness: 0.3, metalness: 0.1, side: THREE.DoubleSide, depthWrite: false
  });
  const mainMesh = new THREE.Mesh(geo, mat);
  segmentBrainMesh(mainMesh);
  brainGroup.add(mainMesh);
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTexture, color: 0x0066ff, transparent: true,
    opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false
  }));
  glow.scale.set(14, 14, 1); brainGroup.add(glow);
}

// ============================================================
// BRAIN REGIONS
// ============================================================
function buildRegions() {
  regions.forEach(region => {
    const hue = REGION_HUES[region.id] || { h: 0.6, s: 0.8, l: 0.5 };
    const color = new THREE.Color().setHSL(hue.h, hue.s, hue.l);
    // Invisible raycast-only sphere (no visual - regions are shown via vertex coloring)
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(region.radius, 12, 8),
      new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0, depthWrite: false, depthTest: false
      })
    );
    mesh.position.set(region.center_x, region.center_y, region.center_z);
    mesh.userData = { type: 'region', regionId: region.id, data: region };
    mesh.renderOrder = -1;
    brainGroup.add(mesh);
    regionMeshes[region.id] = mesh;
    // Region label - projected onto brain surface (outside)
    if (showLabels) {
      const sprite = makeTextSprite(region.name_de || region.name, color);
      // Push label outward from brain center to surface + offset
      const cx = region.center_x, cy = region.center_y, cz = region.center_z;
      const d = Math.sqrt(cx*cx + cy*cy + cz*cz) || 0.01;
      const labelDist = Math.max(3.6, d + region.radius * 0.5);
      sprite.position.set(cx/d * labelDist, cy/d * labelDist + 0.2, cz/d * labelDist);
      sprite.userData = { type: 'label', regionId: region.id };
      brainGroup.add(sprite);
      labelSprites.push(sprite);
    }
  });
}

// ============================================================
// NEURONS: Subtle dots, NO glow. Knowledge lives in labels.
// ============================================================
function buildNeurons() {
  const geo = new THREE.SphereGeometry(0.025, 8, 6);
  neurons.forEach(neuron => {
    const hue = REGION_HUES[neuron.region_id] || { h: 0.6, s: 0.8, l: 0.5 };
    const color = new THREE.Color().setHSL(hue.h, hue.s * 0.8, 0.30 + (neuron.activation || 0) * 0.35);
    const core = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.65
    }));
    // Clamp position inside brain surface
    let px = neuron.pos_x, py = neuron.pos_y, pz = neuron.pos_z;
    const dist = Math.sqrt(px*px + py*py + pz*pz);
    if (dist > BRAIN_RADIUS) {
      const s = BRAIN_RADIUS / dist;
      px *= s; py *= s; pz *= s;
    }
    core.position.set(px, py, pz);
    core.userData = { type: 'neuron', neuronId: neuron.id, data: neuron };
    brainGroup.add(core);
    neuronMeshes[neuron.id] = core;
    neuronHeat[neuron.id] = 0;
  });
}

// ============================================================
// SYNAPSES: The stars of the show. Glow based on weight.
// ============================================================
function buildSynapses() {
  const nMap = {};
  neurons.forEach(n => { nMap[n.id] = n; });

  synapses.forEach(syn => {
    const src = nMap[syn.source_neuron_id];
    const tgt = nMap[syn.target_neuron_id];
    if (!src || !tgt) return;

    const start = new THREE.Vector3(src.pos_x, src.pos_y, src.pos_z);
    const end = new THREE.Vector3(tgt.pos_x, tgt.pos_y, tgt.pos_z);
    const w = syn.weight || 0.1;

    // Determine color from source region
    const rHue = REGION_HUES[src.region_id] || { h: 0.6, s: 0.8, l: 0.5 };
    // Cross-region synapses get a brighter, whiter hue
    const crossRegion = src.region_id !== tgt.region_id;
    const lineColor = crossRegion
      ? new THREE.Color().setHSL(rHue.h, 0.4, 0.35 + w * 0.3)
      : new THREE.Color().setHSL(rHue.h, rHue.s * 0.8, 0.2 + w * 0.35);

    // Line - opacity scales with weight
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([start, end]),
      new THREE.LineBasicMaterial({
        color: lineColor, transparent: true,
        opacity: 0.05 + w * 0.6,
        depthWrite: false
      })
    );
    line.userData = { type: 'synapse', data: syn, start, end };
    brainGroup.add(line);
    synapseMeshes[syn.id] = line;

    // Glow sprite at midpoint for strong synapses
    if (w > 0.2) {
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const dist = start.distanceTo(end);
      const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture,
        color: lineColor,
        transparent: true,
        opacity: w * 0.3,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      }));
      glowSprite.position.copy(mid);
      glowSprite.scale.set(dist * 0.4, dist * 0.15, 1);
      brainGroup.add(glowSprite);
      synapseGlows[syn.id] = glowSprite;
    }

    // Energy particle on strong synapses
    if (w > 0.3 && Math.random() < 0.5) {
      const pColor = new THREE.Color().setHSL(rHue.h, 1.0, 0.7);
      const p = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture, color: pColor, transparent: true, opacity: 0.5,
        blending: THREE.AdditiveBlending, depthWrite: false
      }));
      p.scale.set(0.1, 0.1, 1);
      p.position.copy(start);
      p.userData = { start: start.clone(), end: end.clone(), progress: Math.random(), speed: 0.15 + w * 0.4, synapseId: syn.id };
      brainGroup.add(p);
      energyParticles.push(p);
    }
  });
}

// ============================================================
// TEXT SPRITES
// ============================================================
function makeTextSprite(text, color, fontSize = 26) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 512; canvas.height = 64;
  ctx.font = `600 ${fontSize}px Segoe UI, sans-serif`;
  ctx.textAlign = 'center';
  ctx.shadowColor = `rgba(${Math.round(color.r*255)},${Math.round(color.g*255)},${Math.round(color.b*255)},0.5)`;
  ctx.shadowBlur = 8;
  ctx.fillStyle = `rgba(${Math.round(color.r*255)},${Math.round(color.g*255)},${Math.round(color.b*255)},0.9)`;
  ctx.fillText(text, 256, 42);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false, depthTest: false
  }));
  sprite.scale.set(2.0, 0.25, 1);
  return sprite;
}

function makeNeuronLabel(neuron) {
  const hue = REGION_HUES[neuron.region_id] || { h: 0.6, s: 0.8, l: 0.5 };
  const color = new THREE.Color().setHSL(hue.h, hue.s, 0.7);
  const label = makeTextSprite(neuron.label || neuron.id, color, 22);
  label.position.set(neuron.pos_x, neuron.pos_y + 0.2, neuron.pos_z);
  label.userData = { type: 'neuronLabel', neuronId: neuron.id };
  return label;
}

// ============================================================
// LEGEND
// ============================================================
function buildLegend() {
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  regions.forEach(r => {
    const hue = REGION_HUES[r.id] || { h: 0.6, s: 0.8, l: 0.5 };
    const c = new THREE.Color().setHSL(hue.h, hue.s, hue.l);
    const hex = '#' + c.getHexString();
    const nCount = neurons.filter(n => n.region_id === r.id).length;
    const div = document.createElement('div');
    div.className = 'legend-item';
    div.innerHTML = `<div class="legend-dot" style="background:${hex};box-shadow:0 0 6px ${hex}"></div>${r.name_de || r.name} (${nCount})`;
    div.onclick = () => focusRegion(r.id);
    legend.appendChild(div);
  });
}

// ============================================================
// INTERNAL BRAIN STRUCTURES (Cross-Section Mode)
// ============================================================
function buildInternalStructures() {
  internalGroup = new THREE.Group();
  brainGroup.add(internalGroup);
  clipPlane = new THREE.Plane(clipAxis, clipOffset);

  // -- THALAMUS (Verteilzentrum) -- The central router
  const thGeo = new THREE.SphereGeometry(0.45, 24, 18);
  thGeo.scale(1.3, 0.8, 0.9);
  const thMat = new THREE.MeshPhysicalMaterial({
    color: 0x3388ee, emissive: 0x1144aa, emissiveIntensity: 0.6,
    transparent: true, opacity: 0.75, roughness: 0.2, metalness: 0.2, side: THREE.DoubleSide
  });
  const thL = new THREE.Mesh(thGeo, thMat.clone());
  thL.position.set(-0.28, 0, 0);
  internalGroup.add(thL);
  const thR = new THREE.Mesh(thGeo.clone(), thMat.clone());
  thR.position.set(0.28, 0, 0);
  internalGroup.add(thR);

  // Thalamus glow (bright pulsing center)
  thalamusGlowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTexture, color: 0x4488ff, transparent: true,
    opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false
  }));
  thalamusGlowSprite.scale.set(3, 2, 1);
  internalGroup.add(thalamusGlowSprite);

  // Thalamus label
  const thLabel = makeTextSprite('Thalamus (Verteilzentrum)', new THREE.Color(0x66bbff), 28);
  thLabel.position.set(0, 0.9, 0);
  internalGroup.add(thLabel);

  // -- CORPUS CALLOSUM -- Bridge between hemispheres
  const ccCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-1.8, 0.9, 0),
    new THREE.Vector3(-0.8, 1.2, 0),
    new THREE.Vector3(0, 1.3, 0),
    new THREE.Vector3(0.8, 1.2, 0),
    new THREE.Vector3(1.8, 0.9, 0)
  ]);
  const ccGeo = new THREE.TubeGeometry(ccCurve, 30, 0.12, 8, false);
  const ccMat = new THREE.MeshPhysicalMaterial({
    color: 0x8899bb, transparent: true, opacity: 0.5,
    roughness: 0.4, side: THREE.DoubleSide
  });
  const ccMesh = new THREE.Mesh(ccGeo, ccMat);
  internalGroup.add(ccMesh);
  const ccLabel = makeTextSprite('Corpus Callosum', new THREE.Color(0x8899cc), 20);
  ccLabel.position.set(0, 1.55, 0);
  internalGroup.add(ccLabel);

  // -- LATERAL VENTRICLES -- Fluid-filled cavities
  const vMat = new THREE.MeshPhysicalMaterial({
    color: 0x0a1540, transparent: true, opacity: 0.35,
    roughness: 0.1, side: THREE.DoubleSide
  });
  [[-1], [1]].forEach(([sx]) => {
    const vCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(sx * 0.3, 0.6, 0.7),
      new THREE.Vector3(sx * 0.5, 0.3, 0.3),
      new THREE.Vector3(sx * 0.55, 0, -0.2),
      new THREE.Vector3(sx * 0.4, -0.3, -0.5)
    ]);
    internalGroup.add(new THREE.Mesh(
      new THREE.TubeGeometry(vCurve, 16, 0.1, 8, false), vMat.clone()
    ));
  });

  // -- HIPPOCAMPUS -- Memory formation
  const hCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.7, -0.4, 0.2),
    new THREE.Vector3(-0.3, -0.3, -0.1),
    new THREE.Vector3(0, -0.25, -0.3),
    new THREE.Vector3(0.3, -0.3, -0.1),
    new THREE.Vector3(0.7, -0.4, 0.2)
  ]);
  const hMat = new THREE.MeshPhysicalMaterial({
    color: 0x3366aa, emissive: 0x112244, emissiveIntensity: 0.3,
    transparent: true, opacity: 0.55, roughness: 0.3, side: THREE.DoubleSide
  });
  internalGroup.add(new THREE.Mesh(new THREE.TubeGeometry(hCurve, 20, 0.13, 8, false), hMat));
  const hLabel = makeTextSprite('Hippocampus', new THREE.Color(0x5588bb), 20);
  hLabel.position.set(0, -0.05, -0.3);
  internalGroup.add(hLabel);

  // -- AMYGDALA -- Risk assessment (paired)
  const aMat = new THREE.MeshPhysicalMaterial({
    color: 0x5566aa, emissive: 0x223355, emissiveIntensity: 0.3,
    transparent: true, opacity: 0.5, roughness: 0.3, side: THREE.DoubleSide
  });
  [[-0.7, -0.5, 0.4], [0.7, -0.5, 0.4]].forEach(pos => {
    const aGeo = new THREE.SphereGeometry(0.2, 16, 12);
    aGeo.scale(1, 0.8, 0.9);
    const aMesh = new THREE.Mesh(aGeo, aMat.clone());
    aMesh.position.set(pos[0], pos[1], pos[2]);
    internalGroup.add(aMesh);
  });
  const aLabel = makeTextSprite('Amygdala', new THREE.Color(0x7788bb), 20);
  aLabel.position.set(-0.7, -0.25, 0.4);
  internalGroup.add(aLabel);

  // -- THALAMUS ROUTES -- Lines from center to each brain region
  const center = new THREE.Vector3(0, 0, 0);
  regions.forEach(r => {
    if (r.id === 'thalamus') return;
    const target = new THREE.Vector3(r.center_x, r.center_y, r.center_z);
    const mid = center.clone().lerp(target, 0.5);
    mid.y += 0.3;
    const curve = new THREE.QuadraticBezierCurve3(center, mid, target);
    const hue = REGION_HUES[r.id] || { h: 0.6, s: 0.8, l: 0.5 };
    const color = new THREE.Color().setHSL(hue.h, hue.s, hue.l);
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.25, depthWrite: false })
    );
    line.userData = { routeRegion: r.id };
    internalGroup.add(line);
    thalamusRoutes.push(line);
  });

  // -- CROSS-SECTION DISC -- Visible surface at the cut plane
  const discCanvas = document.createElement('canvas');
  discCanvas.width = 512; discCanvas.height = 512;
  const dCtx = discCanvas.getContext('2d');
  const dGrad = dCtx.createRadialGradient(256, 256, 0, 256, 256, 256);
  dGrad.addColorStop(0, 'rgba(30,60,120,0.7)');
  dGrad.addColorStop(0.12, 'rgba(50,90,160,0.5)');
  dGrad.addColorStop(0.3, 'rgba(40,70,130,0.35)');
  dGrad.addColorStop(0.6, 'rgba(30,55,100,0.25)');
  dGrad.addColorStop(0.85, 'rgba(25,45,85,0.3)');
  dGrad.addColorStop(1.0, 'rgba(15,30,60,0.1)');
  dCtx.fillStyle = dGrad;
  dCtx.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 150; i++) {
    const x = Math.random() * 512, y = Math.random() * 512;
    const d = Math.sqrt((x - 256) ** 2 + (y - 256) ** 2) / 256;
    if (d < 0.92) {
      dCtx.beginPath();
      dCtx.arc(x, y, 1 + Math.random() * 2, 0, Math.PI * 2);
      dCtx.fillStyle = `rgba(40,70,120,${0.08 + Math.random() * 0.15})`;
      dCtx.fill();
    }
  }
  crossSectionDisc = new THREE.Mesh(
    new THREE.CircleGeometry(3.5, 64),
    new THREE.MeshBasicMaterial({
      map: new THREE.CanvasTexture(discCanvas),
      transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false
    })
  );
  crossSectionDisc.visible = false;
  brainGroup.add(crossSectionDisc);

  // Initially hidden
  internalGroup.visible = false;
}

// ============================================================
// CROSS-SECTION MODE
// ============================================================
function applyClipping(enable) {
  const planes = enable ? [clipPlane] : [];
  brainGroup.traverse(child => {
    if (child === crossSectionDisc) return;
    if (internalGroup && isChildOf(child, internalGroup)) return;
    if (child.material && (child.isMesh || child.isLine || child.isSprite)) {
      child.material.clippingPlanes = planes;
      child.material.needsUpdate = true;
    }
  });
}

function isChildOf(obj, parent) {
  let p = obj.parent;
  while (p) {
    if (p === parent) return true;
    p = p.parent;
  }
  return false;
}

window.toggleCrossSection = function() {
  crossSectionActive = !crossSectionActive;
  document.getElementById('btn-cross').classList.toggle('active', crossSectionActive);
  document.getElementById('crosssection-controls').classList.toggle('visible', crossSectionActive);

  if (crossSectionActive) {
    clipOffset = 4.0;
    document.getElementById('clip-slider').value = 40;
    clipPlane.set(clipAxis, clipOffset);
    applyClipping(true);
    internalGroup.visible = true;
    crossSectionDisc.visible = true;
    updateDiscPosition();
    dbg('Querschnitt-Modus aktiviert');
  } else {
    applyClipping(false);
    internalGroup.visible = hemispheresSplit;
    crossSectionDisc.visible = false;
    dbg('Querschnitt-Modus deaktiviert');
  }
};

window.updateClipPosition = function(value) {
  clipOffset = parseFloat(value);
  clipPlane.set(clipAxis, clipOffset);
  updateDiscPosition();
  document.getElementById('clip-label').textContent = 'Tiefe: ' + clipOffset.toFixed(1);
};

window.setClipAxis = function(axis) {
  document.querySelectorAll('.axis-btns .btn').forEach(b => b.classList.remove('active'));
  if (axis === 'z') {
    clipAxis.set(0, 0, -1);
    document.getElementById('ax-z').classList.add('active');
  } else if (axis === 'x') {
    clipAxis.set(-1, 0, 0);
    document.getElementById('ax-x').classList.add('active');
  } else {
    clipAxis.set(0, -1, 0);
    document.getElementById('ax-y').classList.add('active');
  }
  clipPlane.set(clipAxis, clipOffset);
  updateDiscPosition();
};

function updateDiscPosition() {
  if (!crossSectionDisc) return;
  const ax = clipAxis;
  if (Math.abs(ax.z) > 0.5) {
    crossSectionDisc.position.set(0, 0, clipOffset);
    crossSectionDisc.rotation.set(0, 0, 0);
  } else if (Math.abs(ax.x) > 0.5) {
    crossSectionDisc.position.set(clipOffset, 0, 0);
    crossSectionDisc.rotation.set(0, Math.PI / 2, 0);
  } else {
    crossSectionDisc.position.set(0, clipOffset, 0);
    crossSectionDisc.rotation.set(-Math.PI / 2, 0, 0);
  }
}

// ============================================================
// HEMISPHERE SPLIT
// ============================================================
function applyHemisphereOffset(offset) {
  neurons.forEach(n => {
    const mesh = neuronMeshes[n.id];
    if (!mesh) return;
    mesh.position.x = n.pos_x + (n.pos_x <= 0 ? -offset : offset);
  });

  regions.forEach(r => {
    const rm = regionMeshes[r.id];
    if (!rm) return;
    rm.position.x = r.center_x + (r.center_x <= 0 ? -offset : offset);
  });

  labelSprites.forEach(sprite => {
    const rId = sprite.userData?.regionId;
    if (!rId) return;
    const r = regions.find(reg => reg.id === rId);
    if (!r) return;
    sprite.position.x = r.center_x + (r.center_x <= 0 ? -offset : offset);
  });

  // Update synapse lines from mesh positions
  synapses.forEach(syn => {
    const line = synapseMeshes[syn.id];
    if (!line) return;
    const srcMesh = neuronMeshes[syn.source_neuron_id];
    const tgtMesh = neuronMeshes[syn.target_neuron_id];
    if (!srcMesh || !tgtMesh) return;
    const pos = line.geometry.attributes.position;
    pos.setXYZ(0, srcMesh.position.x, srcMesh.position.y, srcMesh.position.z);
    pos.setXYZ(1, tgtMesh.position.x, tgtMesh.position.y, tgtMesh.position.z);
    pos.needsUpdate = true;
    line.userData.start.set(srcMesh.position.x, srcMesh.position.y, srcMesh.position.z);
    line.userData.end.set(tgtMesh.position.x, tgtMesh.position.y, tgtMesh.position.z);
    const glow = synapseGlows[syn.id];
    if (glow) {
      glow.position.set(
        (srcMesh.position.x + tgtMesh.position.x) / 2,
        (srcMesh.position.y + tgtMesh.position.y) / 2,
        (srcMesh.position.z + tgtMesh.position.z) / 2
      );
    }
  });
}

window.splitHemispheres = function() {
  if (hemiAnimating) return;
  hemispheresSplit = !hemispheresSplit;
  document.getElementById('btn-hemi').classList.toggle('active', hemispheresSplit);
  hemiAnimating = true;

  const splitDist = 1.8;
  const duration = 1500;
  const t0 = Date.now();

  if (hemispheresSplit) internalGroup.visible = true;

  function animSplit() {
    const t = Math.min((Date.now() - t0) / duration, 1);
    const e = t * t * (3 - 2 * t);
    const currentSplit = hemispheresSplit ? splitDist * e : splitDist * (1 - e);
    applyHemisphereOffset(currentSplit);

    if (t < 1) {
      requestAnimationFrame(animSplit);
    } else {
      hemiAnimating = false;
      if (!hemispheresSplit) internalGroup.visible = crossSectionActive;
    }
  }
  animSplit();
  dbg(hemispheresSplit ? 'Hemisph\u00e4ren getrennt' : 'Hemisph\u00e4ren zusammengef\u00fchrt');
};

// ============================================================
// NEURON FIRING: Real Hebbian Learning
// ============================================================
async function fireNeuron(neuronId, propagate = true) {
  const mesh = neuronMeshes[neuronId];
  if (!mesh) return;

  const neuron = neurons.find(n => n.id === neuronId);
  if (!neuron) return;

  // Visual: activation flash (bloom-compatible - bright colors trigger glow)
  const origOpacity = mesh.material.opacity;
  const origColor = mesh.material.color.clone();
  mesh.material.opacity = 1.0;
  mesh.material.color.setHSL(
    origColor.getHSL({}).h, 1.0, 0.85  // very bright = bloom glow
  );
  mesh.scale.set(1.6, 1.6, 1.6);

  // Visual: light up ALL connected synapses (the main visual effect)
  const connectedSynapses = synapses.filter(s =>
    s.source_neuron_id === neuronId || s.target_neuron_id === neuronId
  );

  connectedSynapses.forEach(syn => {
    const line = synapseMeshes[syn.id];
    const glow = synapseGlows[syn.id];
    if (line) {
      const origOp = line.material.opacity;
      line.material.opacity = 0.95;
      line.material.color.set(0x66ccff);
      // Glow pulse
      if (glow) {
        glow.material.opacity = 0.8;
        glow.material.color.set(0x66ccff);
      }
      setTimeout(() => {
        const rHue = REGION_HUES[neuron.region_id] || { h: 0.6, s: 0.8, l: 0.5 };
        const w = syn.weight || 0.1;
        line.material.color.setHSL(rHue.h, rHue.s * 0.8, 0.2 + w * 0.35);
        line.material.opacity = origOp;
        if (glow) {
          glow.material.color.setHSL(rHue.h, rHue.s, 0.5);
          glow.material.opacity = w * 0.3;
        }
      }, 600);
    }
  });

  fireCount++;
  // Track heat (for color temperature visualization)
  neuronHeat[neuronId] = (neuronHeat[neuronId] || 0) + 1;

  // Reset neuron visual (smooth fade back)
  setTimeout(() => {
    mesh.material.opacity = origOpacity;
    mesh.material.color.copy(origColor);
    mesh.scale.set(1, 1, 1);
  }, 250);

  // Thought ripple (subtle expanding wave from firing neuron)
  if (propagate && ripples.length < 5) {
    createRipple(mesh.position.clone(), mesh.material.color.clone());
  }

  // Propagate with DISTANCE-BASED speed (nearby = instant, far = delayed)
  if (propagate) {
    connectedSynapses.forEach(syn => {
      const targetId = syn.source_neuron_id === neuronId ? syn.target_neuron_id : syn.source_neuron_id;
      const targetMesh = neuronMeshes[targetId];
      const w = syn.weight || 0.1;
      if (Math.random() < w * 0.35) {
        // Distance-based delay: nearby neurons fire fast, far ones slow
        const dist = targetMesh ? mesh.position.distanceTo(targetMesh.position) : 2;
        const delay = 40 + dist * 80; // 40ms nearby → 280ms far
        setTimeout(() => fireNeuron(targetId, false), delay);
      }
    });
  }

  // Server-side Hebbian learning via Edge Function - async, don't block
  try {
    edgeFn('fire', {
      neuron_id: neuronId,
      source: 'visualization',
      detail: neuron.label,
      strength: 1.0
    });
  } catch(e) { /* non-blocking */ }
}

function ambientFire() {
  if (neurons.length === 0) return;
  // Fire 2-4 random neurons per tick for a more lively brain
  const count = 2 + Math.floor(Math.random() * 3);
  for (let i = 0; i < count; i++) {
    const thalamusNeurons = neurons.filter(n => n.region_id === 'thalamus');
    const pool = Math.random() < 0.2 && thalamusNeurons.length > 0 ? thalamusNeurons : neurons;
    fireNeuron(pool[Math.floor(Math.random() * pool.length)].id);
  }
}

window.simulateFiring = function() {
  // Fire thalamus cascade (signal routing demo)
  neurons.filter(n => n.region_id === 'thalamus').forEach((n, i) => {
    setTimeout(() => fireNeuron(n.id), i * 100);
  });
};

// ============================================================
// CREATE NEURON (from UI)
// ============================================================
async function createNeuronFromUI() {
  const label = document.getElementById('nn-label').value.trim();
  const region = document.getElementById('nn-region').value;
  const type = document.getElementById('nn-type').value;
  const desc = document.getElementById('nn-desc').value.trim();
  if (!label || !region) { alert('Label und Region sind Pflichtfelder'); return; }

  try {
    const result = await edgeFn('create', {
      label, region_id: region, type,
      description: desc || null, created_by: 'user'
    });
    if (result.error) throw new Error(result.error);
    // Reload data and rebuild
    await loadData();
    rebuildScene();
    closeNewNeuronPanel();
    dbg(`Neuron "${label}" erstellt`);
  } catch(e) {
    console.error('Create neuron failed:', e);
    alert('Fehler beim Erstellen: ' + e.message);
  }
}

function rebuildScene() {
  // Remove old neurons and synapses from scene
  Object.values(neuronMeshes).forEach(m => brainGroup.remove(m));
  Object.values(synapseMeshes).forEach(m => brainGroup.remove(m));
  Object.values(synapseGlows).forEach(m => brainGroup.remove(m));
  energyParticles.forEach(p => brainGroup.remove(p));
  neuronMeshes = {};
  synapseMeshes = {};
  synapseGlows = {};
  energyParticles = [];
  buildNeurons();
  buildSynapses();
  updateStats();
  buildLegend();
}

window.createNeuronFromUI = createNeuronFromUI;

window.openNewNeuronPanel = function() {
  const panel = document.getElementById('new-neuron-panel');
  if (!panel) return;
  // Populate region dropdown
  const sel = document.getElementById('nn-region');
  sel.innerHTML = '';
  regions.forEach(r => {
    const opt = document.createElement('option');
    opt.value = r.id;
    opt.textContent = r.name_de || r.name;
    sel.appendChild(opt);
  });
  panel.classList.add('visible');
};

function closeNewNeuronPanel() {
  document.getElementById('new-neuron-panel')?.classList.remove('visible');
  document.getElementById('nn-label').value = '';
  document.getElementById('nn-desc').value = '';
}
window.closeNewNeuronPanel = closeNewNeuronPanel;

// ============================================================
// REGION FOCUS
// ============================================================
function focusRegion(regionId) {
  const region = regions.find(r => r.id === regionId);
  if (!region) return;
  const target = new THREE.Vector3(region.center_x, region.center_y, region.center_z);
  const newPos = target.clone().add(new THREE.Vector3(region.radius*2, region.radius*1.5, region.radius*2));
  const sp = camera.position.clone(), st = controls.target.clone();
  const t0 = Date.now();
  function anim() {
    const t = Math.min((Date.now()-t0)/1000, 1);
    const e = t*t*(3-2*t);
    camera.position.lerpVectors(sp, newPos, e);
    controls.target.lerpVectors(st, target, e);
    controls.update();
    if (t < 1) requestAnimationFrame(anim);
  }
  anim();
  // Fire all neurons in region
  neurons.filter(n => n.region_id===regionId).forEach((n,i) => {
    setTimeout(() => fireNeuron(n.id, false), i*80);
  });
  showRegionPanel(regionId);
  document.querySelectorAll('.legend-item').forEach((item, idx) => {
    item.classList.toggle('active', regions[idx]?.id === regionId);
  });
}

function showRegionPanel(regionId) {
  const region = regions.find(r => r.id === regionId);
  if (!region) return;
  const rNeurons = neurons.filter(n => n.region_id === regionId);
  const rSynapses = synapses.filter(s => {
    const sn = neurons.find(n => n.id===s.source_neuron_id);
    return sn && sn.region_id===regionId;
  });
  const avgSuccess = rNeurons.length > 0 ? rNeurons.reduce((s,n) => s+(n.success_rate||0.5), 0)/rNeurons.length : 0;
  const avgWeight = rSynapses.length > 0 ? rSynapses.reduce((s,syn) => s+(syn.weight||0.1), 0)/rSynapses.length : 0;
  document.getElementById('rp-name').textContent = region.name_de || region.name;
  document.getElementById('rp-desc').textContent = region.description || '';
  document.getElementById('rp-neurons').textContent = rNeurons.length;
  document.getElementById('rp-synapses').textContent = rSynapses.length;
  document.getElementById('rp-success').textContent = (avgSuccess*100).toFixed(1)+'%';
  document.getElementById('rp-model').textContent = region.agent_model || 'sonnet';
  document.getElementById('rp-status').textContent = `${(avgWeight*100).toFixed(0)}% Synapse-Staerke`;
  // List neuron labels
  const list = document.getElementById('rp-neuron-list');
  if (list) {
    list.innerHTML = rNeurons.map(n =>
      `<div class="neuron-list-item" onclick="window.focusNeuron('${n.id}')">${n.label} <span style="opacity:0.5">(${n.fire_count||0}x)</span></div>`
    ).join('');
  }
  document.getElementById('region-panel').classList.add('visible');
}

window.focusNeuron = function(neuronId) {
  const neuron = neurons.find(n => n.id === neuronId);
  if (!neuron) return;
  fireNeuron(neuronId);
  const target = new THREE.Vector3(neuron.pos_x, neuron.pos_y, neuron.pos_z);
  const newPos = target.clone().add(new THREE.Vector3(1, 0.8, 1));
  const sp = camera.position.clone(), st = controls.target.clone();
  const t0 = Date.now();
  function anim() {
    const t = Math.min((Date.now()-t0)/800, 1);
    const e = t*t*(3-2*t);
    camera.position.lerpVectors(sp, newPos, e);
    controls.target.lerpVectors(st, target, e);
    controls.update();
    if (t < 1) requestAnimationFrame(anim);
  }
  anim();
};

// ============================================================
// INTERACTION
// ============================================================
function onMouseClick(event) {
  mouse.x = (event.clientX/window.innerWidth)*2-1;
  mouse.y = -(event.clientY/window.innerHeight)*2+1;
  raycaster.setFromCamera(mouse, camera);
  for (const hit of raycaster.intersectObjects(brainGroup.children)) {
    if (hit.object.userData.type==='neuron') {
      fireNeuron(hit.object.userData.neuronId);
      showNeuronDetail(hit.object.userData.neuronId);
      return;
    }
    if (hit.object.userData.type==='region') { focusRegion(hit.object.userData.regionId); return; }
  }
  document.getElementById('region-panel').classList.remove('visible');
  document.querySelectorAll('.legend-item').forEach(i => i.classList.remove('active'));
}

function showNeuronDetail(neuronId) {
  const neuron = neurons.find(n => n.id === neuronId);
  if (!neuron) return;
  const connCount = synapses.filter(s => s.source_neuron_id === neuronId || s.target_neuron_id === neuronId).length;
  document.getElementById('rp-name').textContent = neuron.label || neuron.id;
  document.getElementById('rp-desc').textContent = neuron.description || neuron.neuron_type || '';
  document.getElementById('rp-neurons').textContent = neuron.neuron_type || 'concept';
  document.getElementById('rp-synapses').textContent = connCount;
  document.getElementById('rp-success').textContent = ((neuron.success_rate||0.5)*100).toFixed(1)+'%';
  document.getElementById('rp-model').textContent = neuron.created_by || '-';
  document.getElementById('rp-status').textContent = `${neuron.fire_count||0}x gefeuert`;
  const list = document.getElementById('rp-neuron-list');
  if (list) {
    // Show connected neurons
    const connected = synapses
      .filter(s => s.source_neuron_id === neuronId || s.target_neuron_id === neuronId)
      .map(s => {
        const otherId = s.source_neuron_id === neuronId ? s.target_neuron_id : s.source_neuron_id;
        const other = neurons.find(n => n.id === otherId);
        return other ? `<div class="neuron-list-item" onclick="window.focusNeuron('${otherId}')">${other.label} <span style="opacity:0.5">(w:${(s.weight||0).toFixed(2)})</span></div>` : '';
      }).join('');
    list.innerHTML = connected;
  }
  document.getElementById('region-panel').classList.add('visible');
}

function onMouseMove(event) {
  mouse.x = (event.clientX/window.innerWidth)*2-1;
  mouse.y = -(event.clientY/window.innerHeight)*2+1;
  raycaster.setFromCamera(mouse, camera);

  // Remove previous hover label
  if (hoverLabel) { brainGroup.remove(hoverLabel); hoverLabel = null; }

  let foundHover = false;
  for (const hit of raycaster.intersectObjects(brainGroup.children)) {
    if (hit.object.userData.type === 'neuron') {
      const nId = hit.object.userData.neuronId;
      if (hoveredNeuron !== nId) {
        hoveredNeuron = nId;
        const neuron = neurons.find(n => n.id === nId);
        if (neuron) {
          hoverLabel = makeNeuronLabel(neuron);
          brainGroup.add(hoverLabel);
        }
      } else if (hoveredNeuron === nId) {
        const neuron = neurons.find(n => n.id === nId);
        if (neuron) {
          hoverLabel = makeNeuronLabel(neuron);
          brainGroup.add(hoverLabel);
        }
      }
      // Highlight connected synapses on hover
      highlightConnections(nId);
      renderer.domElement.style.cursor = 'pointer';
      foundHover = true;
      break;
    }
    if (hit.object.userData.type === 'region') {
      renderer.domElement.style.cursor = 'pointer';
      foundHover = true;
      break;
    }
  }

  if (!foundHover) {
    if (hoveredNeuron) {
      unhighlightConnections();
      hoveredNeuron = null;
    }
    renderer.domElement.style.cursor = 'default';
  }
}

function highlightConnections(neuronId) {
  // Dim all synapses first, then brighten connected ones
  Object.entries(synapseMeshes).forEach(([id, line]) => {
    const syn = line.userData.data;
    if (syn.source_neuron_id === neuronId || syn.target_neuron_id === neuronId) {
      line.material.opacity = 0.8;
      const glow = synapseGlows[id];
      if (glow) glow.material.opacity = 0.6;
    } else {
      line.material.opacity = 0.02;
      const glow = synapseGlows[id];
      if (glow) glow.material.opacity = 0.02;
    }
  });
}

function unhighlightConnections() {
  Object.entries(synapseMeshes).forEach(([id, line]) => {
    const syn = line.userData.data;
    const w = syn.weight || 0.1;
    line.material.opacity = 0.05 + w * 0.6;
    const glow = synapseGlows[id];
    if (glow) glow.material.opacity = w * 0.3;
  });
}

// ============================================================
// CONTROLS
// ============================================================
window.toggleAutoRotate = function() {
  autoRotate = !autoRotate;
  controls.autoRotate = autoRotate;
  document.getElementById('btn-rotate').classList.toggle('active', autoRotate);
};
window.toggleLabels = function() {
  showLabels = !showLabels;
  labelSprites.forEach(s => { s.visible = showLabels; });
  document.getElementById('btn-labels').classList.toggle('active', showLabels);
};

// ============================================================
// STATS
// ============================================================
function updateStats() {
  document.getElementById('stat-neurons').textContent = neurons.length;
  document.getElementById('stat-synapses').textContent = synapses.length;
  document.getElementById('stat-regions').textContent = regions.length;
  // Average synapse weight as "brain connectivity"
  const avgW = synapses.length > 0
    ? synapses.reduce((s, syn) => s + (syn.weight||0.1), 0) / synapses.length : 0;
  const connEl = document.getElementById('stat-connectivity');
  if (connEl) connEl.textContent = (avgW * 100).toFixed(0) + '%';
}

// ============================================================
// ANIMATION LOOP
// ============================================================
function animate() {
  requestAnimationFrame(animate);
  const time = clock.getElapsedTime();

  if (brainGroup) brainGroup.scale.setScalar(1 + Math.sin(time*0.5)*0.003);

  // Cross-section: Thalamus pulse + route animation
  if ((crossSectionActive || hemispheresSplit) && thalamusGlowSprite) {
    thalamusGlowSprite.material.opacity = 0.3 + Math.sin(time * 2) * 0.2;
    thalamusRoutes.forEach((line, i) => {
      line.material.opacity = 0.15 + Math.sin(time * 1.5 + i * 0.5) * 0.15;
    });
  }

  // Synapse glow pulse (the key visual)
  Object.entries(synapseGlows).forEach(([id, sprite]) => {
    const syn = synapses.find(s => s.id === id);
    if (!syn) return;
    const w = syn.weight || 0.1;
    const pulse = 0.02 + Math.sin(time * 1.2 + w * 10) * 0.08;
    sprite.material.opacity = w * 0.25 + pulse;
  });

  // Energy particles travel along synapses
  energyParticles.forEach(particle => {
    const ud = particle.userData;
    ud.progress += ud.speed * 0.006;
    if (ud.progress > 1) {
      ud.progress = 0;
      if (Math.random() < 0.3) {
        const tmp = ud.start.clone(); ud.start.copy(ud.end); ud.end.copy(tmp);
      }
    }
    particle.position.lerpVectors(ud.start, ud.end, ud.progress);
    particle.material.opacity = 0.2 + Math.sin(ud.progress * Math.PI) * 0.5;
    const ps = 0.06 + Math.sin(ud.progress * Math.PI) * 0.05;
    particle.scale.set(ps, ps, 1);
  });

  // Fire rate counter
  if (time - lastFireCheck > 1.0) {
    fireRate = fireCount; fireCount = 0; lastFireCheck = time;
    document.getElementById('stat-firerate').textContent = fireRate;
  }

  // Neuron heat color update (cool blue → warm cyan → hot white)
  Object.entries(neuronMeshes).forEach(([id, mesh]) => {
    const heat = neuronHeat[id] || 0;
    if (heat <= 0.01) return;
    const neuron = neurons.find(n => n.id === id);
    if (!neuron) return;
    const hue = REGION_HUES[neuron.region_id] || { h: 0.6, s: 0.8, l: 0.5 };
    const hf = Math.min(heat / 12, 1); // normalize to 0-1
    // Shift hue toward yellow, desaturate, brighten as heat rises
    const h = hue.h + hf * (0.15 - hue.h) * 0.4;
    const s = hue.s * (1 - hf * 0.6);
    const l = 0.3 + hf * 0.55;
    mesh.material.color.setHSL(h, s, l);
    mesh.material.opacity = 0.55 + hf * 0.4;
    // Scale: hot neurons are slightly larger
    const sc = 1 + hf * 0.5;
    mesh.scale.set(sc, sc, sc);
    // Decay heat
    neuronHeat[id] = Math.max(0, heat - 0.02);
  });

  // Thought ripple animation
  ripples = ripples.filter(ripple => {
    const age = time - ripple.userData.startTime;
    const progress = age / 1.2;
    if (progress > 1) {
      brainGroup.remove(ripple);
      ripple.geometry.dispose();
      ripple.material.dispose();
      return false;
    }
    const s = 2.0 * progress;
    ripple.scale.set(s, s, s);
    ripple.material.opacity = 0.15 * (1 - progress);
    return true;
  });

  // Update vignette time uniform for animated film grain
  if (window._vignettePass) {
    window._vignettePass.uniforms.time.value = time;
  }

  // Instanced mesh pulse animation (subtle breathing for 10K neurons)
  if (wsInstancedMesh && wsLiveMode) {
    // Very subtle global pulse — makes the brain feel alive
    const pulse = 1.0 + Math.sin(time * 0.8) * 0.002;
    wsInstancedMesh.scale.setScalar(pulse);
  }

  controls.update();
  // Bloom + cinematic post-processing render
  if (composer) {
    composer.render();
  } else {
    renderer.render(scene, camera);
  }
}

// ============================================================
// RESIZE
// ============================================================
function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (composer) composer.setSize(window.innerWidth, window.innerHeight);
}

// ============================================================
// THOUGHT RIPPLE (visual wave on neuron firing)
// ============================================================
function createRipple(position, color) {
  const rippleGeo = new THREE.RingGeometry(0.05, 0.12, 24);
  const rippleMat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.15,
    side: THREE.DoubleSide, depthWrite: false
  });
  const ripple = new THREE.Mesh(rippleGeo, rippleMat);
  ripple.position.copy(position);
  // Random orientation for organic look
  ripple.rotation.set(Math.random()*Math.PI, Math.random()*Math.PI, 0);
  ripple.userData = { startTime: clock.getElapsedTime() };
  brainGroup.add(ripple);
  ripples.push(ripple);
}

// ============================================================
// SOM ATTRACTION (neurons drift toward co-firing partners)
// ============================================================
function somAttractionStep() {
  const ATTRACT_STRENGTH = 0.008;   // weaker attraction
  const ANCHOR_STRENGTH = 0.025;    // spring back to area center
  const REPULSION_STRENGTH = 0.003; // push away when too close
  const REPULSION_RADIUS = 0.25;    // minimum distance between neurons
  let totalDrift = 0;

  // Build region center lookup from region definitions
  const regionCenters = {};
  regions.forEach(r => {
    regionCenters[r.id] = new THREE.Vector3(r.center_x || 0, r.center_y || 0, r.center_z || 0);
  });

  neurons.forEach(n => {
    const mesh = neuronMeshes[n.id];
    if (!mesh) return;

    let dx = 0, dy = 0, dz = 0;

    // 1. Synapse attraction (weaker than before)
    const connected = synapses.filter(s =>
      s.source_neuron_id === n.id || s.target_neuron_id === n.id
    );
    if (connected.length > 0) {
      let px = 0, py = 0, pz = 0, tw = 0;
      connected.forEach(s => {
        const otherId = s.source_neuron_id === n.id ? s.target_neuron_id : s.source_neuron_id;
        const om = neuronMeshes[otherId];
        if (!om) return;
        const w = Math.pow(s.weight || 0.1, 2);
        px += om.position.x * w;
        py += om.position.y * w;
        pz += om.position.z * w;
        tw += w;
      });
      if (tw > 0) {
        dx += (px/tw - mesh.position.x) * ATTRACT_STRENGTH;
        dy += (py/tw - mesh.position.y) * ATTRACT_STRENGTH;
        dz += (pz/tw - mesh.position.z) * ATTRACT_STRENGTH;
      }
    }

    // 2. Area anchoring (spring toward region center)
    const center = regionCenters[n.region_id];
    if (center) {
      dx += (center.x - mesh.position.x) * ANCHOR_STRENGTH;
      dy += (center.y - mesh.position.y) * ANCHOR_STRENGTH;
      dz += (center.z - mesh.position.z) * ANCHOR_STRENGTH;
    }

    // 3. Short-range repulsion (prevent collapse)
    neurons.forEach(other => {
      if (other.id === n.id) return;
      const om = neuronMeshes[other.id];
      if (!om) return;
      const ddx = mesh.position.x - om.position.x;
      const ddy = mesh.position.y - om.position.y;
      const ddz = mesh.position.z - om.position.z;
      const dist = Math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
      if (dist < REPULSION_RADIUS && dist > 0.001) {
        const force = REPULSION_STRENGTH / (dist * dist);
        dx += (ddx / dist) * force;
        dy += (ddy / dist) * force;
        dz += (ddz / dist) * force;
      }
    });

    mesh.position.x += dx;
    mesh.position.y += dy;
    mesh.position.z += dz;
    totalDrift += Math.abs(dx) + Math.abs(dy) + Math.abs(dz);

    // Clamp inside brain
    const d = mesh.position.length();
    if (d > BRAIN_RADIUS) mesh.position.multiplyScalar(BRAIN_RADIUS / d);
  });

  // Rebuild synapse geometry after neuron movement
  rebuildSynapseGeometry();

  if (totalDrift > 0.01) {
    dbg(`SOM: ${totalDrift.toFixed(3)} drift (area-anchored)`);
  }
}

// ============================================================
// LIVE AGENT ACTIVITY (Supabase Realtime Polling)
// ============================================================
async function pollAgentActivity() {
  try {
    const activities = await sbQuery('agent_activity_log',
      `select=*&created_at=gt.${encodeURIComponent(lastActivityTime)}&order=created_at.desc&limit=10`);
    if (!Array.isArray(activities) || activities.length === 0) return;

    if (!liveConnected) {
      liveConnected = true;
      updateLiveIndicator();
    }

    lastActivityTime = activities[0].created_at;
    liveActivityCount += activities.length;

    activities.forEach(activity => {
      const regionId = AGENT_REGION_MAP[activity.agent_name];
      if (!regionId) return;

      // Fire neurons in the agent's brain region
      const regionNeurons = neurons.filter(n => n.region_id === regionId);
      if (regionNeurons.length === 0) return;

      // Staggered firing for visual cascade effect
      regionNeurons.forEach((n, i) => {
        setTimeout(() => fireNeuron(n.id, true), i * 60);
      });

      dbg(`LIVE: ${activity.agent_name} → ${regionId} (${activity.event})`);
    });
  } catch(e) {
    // Silently fail - polling is best-effort
  }
}

function updateLiveIndicator() {
  const el = document.getElementById('live-indicator');
  if (el) {
    el.style.display = liveConnected ? 'flex' : 'none';
  }
}

// ============================================================
// START
// ============================================================
init().catch(err => {
  console.error('Fatal:', err);
  document.getElementById('loading')?.classList.add('hidden');
});
