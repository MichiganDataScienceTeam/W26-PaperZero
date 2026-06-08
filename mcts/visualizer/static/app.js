const PAPER_MARGIN = 0.10;

const state = {
  tree: null,
  selectedNodeId: "root",
  selectedNode: null,
  foldPoints: [],
  checkpoints: [],
  loadedCheckpoint: null,
};

const el = {
  treeSvg: document.getElementById("treeSvg"),
  treeTooltip: document.getElementById("treeTooltip"),
  statusText: document.getElementById("statusText"),
  paperCanvas: document.getElementById("paperCanvas"),
  startCanvas: document.getElementById("startCanvas"),
  endCanvas: document.getElementById("endCanvas"),
  checkpointSelect: document.getElementById("checkpointSelect"),
  refreshCheckpoints: document.getElementById("refreshCheckpoints"),
  loadCheckpoint: document.getElementById("loadCheckpoint"),
  refreshTree: document.getElementById("refreshTree"),
  clearPoints: document.getElementById("clearPoints"),
  applyFold: document.getElementById("applyFold"),
  runExpand: document.getElementById("runExpand"),
  runSelect: document.getElementById("runSelect"),
};

const paperCtx = el.paperCanvas.getContext("2d");
const startCtx = el.startCanvas.getContext("2d");
const endCtx = el.endCanvas.getContext("2d");

function setStatus(text, isError = false) {
  el.statusText.textContent = text;
  el.statusText.style.color = isError ? "#a32020" : "#5a5a5a";
}

async function apiJson(url, options = undefined, label = "Request") {
  const res = await fetch(url, options);
  let body = null;
  try {
    body = await res.json();
  } catch {
    throw new Error(`${label} failed (${res.status}): non-JSON response`);
  }

  if (!res.ok) {
    throw new Error(body.error || `${label} failed (${res.status})`);
  }
  return body;
}

function drawMaskIntoRect(ctx, mask, rect) {
  const h = mask.length;
  const w = h > 0 ? mask[0].length : 0;
  if (h === 0 || w === 0) {
    return;
  }

  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const v = Math.max(0, Math.min(1, mask[y][x]));
      const c = Math.round(v * 255);
      const idx = (y * w + x) * 4;
      img.data[idx] = c;
      img.data[idx + 1] = c;
      img.data[idx + 2] = c;
      img.data[idx + 3] = 255;
    }
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = w;
  offscreen.height = h;
  offscreen.getContext("2d").putImageData(img, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, rect.x, rect.y, rect.w, rect.h);
}

function getPaperRect(canvas) {
  const pad = Math.round(canvas.width * PAPER_MARGIN);
  return {
    x: pad,
    y: pad,
    w: canvas.width - 2 * pad,
    h: canvas.height - 2 * pad,
  };
}

function drawFoldOverlay() {
  if (state.foldPoints.length === 0) {
    return;
  }

  paperCtx.save();
  paperCtx.strokeStyle = "#0057d8";
  paperCtx.fillStyle = "#0057d8";
  paperCtx.lineWidth = 2;

  for (const p of state.foldPoints) {
    paperCtx.beginPath();
    paperCtx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    paperCtx.fill();
  }

  if (state.foldPoints.length === 2) {
    paperCtx.beginPath();
    paperCtx.moveTo(state.foldPoints[0].x, state.foldPoints[0].y);
    paperCtx.lineTo(state.foldPoints[1].x, state.foldPoints[1].y);
    paperCtx.stroke();
  }

  paperCtx.restore();
}

function drawFoldSelector(mask) {
  paperCtx.clearRect(0, 0, el.paperCanvas.width, el.paperCanvas.height);
  paperCtx.fillStyle = "#000000";
  paperCtx.fillRect(0, 0, el.paperCanvas.width, el.paperCanvas.height);

  const paperRect = getPaperRect(el.paperCanvas);
  paperCtx.fillStyle = "#ffffff";
  paperCtx.fillRect(paperRect.x, paperRect.y, paperRect.w, paperRect.h);

  drawMaskIntoRect(paperCtx, mask, paperRect);
  paperCtx.strokeStyle = "#8d8d8d";
  paperCtx.lineWidth = 1;
  paperCtx.strokeRect(paperRect.x + 0.5, paperRect.y + 0.5, paperRect.w - 1, paperRect.h - 1);

  drawFoldOverlay();
}

function heatColor(v) {
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(255 * Math.min(1, 2 * t));
  const g = Math.round(255 * (1 - Math.abs(2 * t - 1)));
  const b = Math.round(255 * Math.max(0, 1 - 2 * t));
  return [r, g, b];
}

function drawHeatmap(ctx, map) {
  const h = map.length;
  const w = h > 0 ? map[0].length : 0;
  if (h === 0 || w === 0) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    return;
  }

  let maxVal = 0;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (map[y][x] > maxVal) {
        maxVal = map[y][x];
      }
    }
  }
  const denom = maxVal > 0 ? maxVal : 1;

  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const [r, g, b] = heatColor(map[y][x] / denom);
      const idx = (y * w + x) * 4;
      img.data[idx] = r;
      img.data[idx + 1] = g;
      img.data[idx + 2] = b;
      img.data[idx + 3] = 255;
    }
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = w;
  offscreen.height = h;
  offscreen.getContext("2d").putImageData(img, 0, 0);

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, ctx.canvas.width, ctx.canvas.height);
}

function maskToDataUrl(mask) {
  const h = mask.length;
  const w = h > 0 ? mask[0].length : 0;
  if (h === 0 || w === 0) {
    return "";
  }

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const v = Math.max(0, Math.min(1, mask[y][x]));
      const c = Math.round(v * 255);
      const idx = (y * w + x) * 4;
      img.data[idx] = c;
      img.data[idx + 1] = c;
      img.data[idx + 2] = c;
      img.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png");
}

function flattenTree(root) {
  const nodes = [];
  const links = [];
  let nextX = 0;

  function walk(node, depth) {
    const children = node.children || [];
    let x;

    if (children.length === 0) {
      x = nextX;
      nextX += 1;
    } else {
      const childXs = children.map((child) => walk(child, depth + 1));
      x = childXs.reduce((a, b) => a + b, 0) / childXs.length;
    }

    nodes.push({ ...node, depth, x });
    for (const child of children) {
      links.push({ parent: node.id, child: child.id });
    }

    return x;
  }

  walk(root, 0);

  return {
    nodes,
    links,
    maxDepth: nodes.reduce((m, n) => Math.max(m, n.depth), 0),
    maxX: nodes.reduce((m, n) => Math.max(m, n.x), 0),
  };
}

function makeSvg(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, String(v)));
  return node;
}

function tooltipForNode(node) {
  return [
    `N: ${node.visits}`,
    `W: ${node.value_sum.toFixed(4)}`,
    `Q: ${node.q.toFixed(4)}`,
    `P: ${node.prior.toFixed(4)}`,
    `Unvisited children: ${node.unvisited_children}`,
  ].join("\n");
}

function showTooltip(text, clientX, clientY) {
  el.treeTooltip.textContent = text;
  el.treeTooltip.hidden = false;
  el.treeTooltip.style.left = `${clientX + 10}px`;
  el.treeTooltip.style.top = `${clientY + 10}px`;
}

function hideTooltip() {
  el.treeTooltip.hidden = true;
}

function renderTree() {
  if (!state.tree) {
    return;
  }

  const { nodes, links, maxDepth, maxX } = flattenTree(state.tree);

  const dx = 130;
  const dy = 105;
  const margin = 72;
  const nodeW = 72;
  const nodeH = 72;
  const thumb = 56;

  const width = Math.max(420, (maxX + 1) * dx + margin * 2);
  const height = Math.max(260, (maxDepth + 1) * dy + margin * 2);

  el.treeSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  el.treeSvg.setAttribute("width", width);
  el.treeSvg.setAttribute("height", height);
  el.treeSvg.innerHTML = "";

  const positions = new Map();
  for (const node of nodes) {
    positions.set(node.id, {
      ...node,
      px: margin + node.x * dx,
      py: margin + node.depth * dy,
    });
  }

  for (const link of links) {
    const parent = positions.get(link.parent);
    const child = positions.get(link.child);
    if (!parent || !child) {
      continue;
    }

    const x1 = parent.px;
    const y1 = parent.py + nodeH / 2;
    const x2 = child.px;
    const y2 = child.py - nodeH / 2;
    const my = (y1 + y2) / 2;

    el.treeSvg.append(
      makeSvg("path", {
        class: "tree-link",
        d: `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`,
      })
    );
  }

  for (const node of nodes) {
    const pos = positions.get(node.id);
    const group = makeSvg("g", { class: "tree-node", transform: `translate(${pos.px}, ${pos.py})` });

    if (node.id === state.selectedNodeId) {
      group.classList.add("active");
    }

    group.append(
      makeSvg("rect", {
        x: -nodeW / 2,
        y: -nodeH / 2,
        width: nodeW,
        height: nodeH,
      })
    );

    const imageHref = maskToDataUrl(node.preview_mask || []);
    if (imageHref) {
      const image = makeSvg("image", {
        x: -thumb / 2,
        y: -thumb / 2,
        width: thumb,
        height: thumb,
        preserveAspectRatio: "none",
      });
      image.setAttribute("href", imageHref);
      image.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", imageHref);
      group.append(image);
    }

    if ((node.unvisited_children || 0) > 0) {
      const badgeText = `+${node.unvisited_children}`;
      group.append(
        makeSvg("rect", {
          x: nodeW / 2 - 24,
          y: -nodeH / 2 + 4,
          width: 20,
          height: 12,
          fill: "#f0f0f0",
          stroke: "#7f7f7f",
          "stroke-width": 1,
        })
      );
      const t = makeSvg("text", {
        class: "tree-badge",
        x: nodeW / 2 - 14,
        y: -nodeH / 2 + 13,
        "text-anchor": "middle",
      });
      t.textContent = badgeText;
      group.append(t);
    }

    group.style.cursor = "pointer";
    group.addEventListener("mouseenter", (event) => showTooltip(tooltipForNode(node), event.clientX, event.clientY));
    group.addEventListener("mousemove", (event) => showTooltip(tooltipForNode(node), event.clientX, event.clientY));
    group.addEventListener("mouseleave", hideTooltip);
    group.addEventListener("click", async () => {
      try {
        await fetchNode(node.id);
        renderTree();
      } catch (err) {
        setStatus(err.message, true);
      }
    });

    el.treeSvg.append(group);
  }
}

function canvasToPaperCoords(px, py, canvas, bounds) {
  const [minX, maxX, minY, maxY] = bounds || [0, 1, 0, 1];
  const rect = getPaperRect(canvas);
  const rows = state.selectedNode?.paper_mask?.length || 128;
  const cols = rows > 0 ? (state.selectedNode.paper_mask[0]?.length || 128) : 128;

  // Canvas point -> raster pixel coordinates.
  const xr = ((px - rect.x) / rect.w) * (cols - 1);
  const yr = ((py - rect.y) / rect.h) * (rows - 1);

  // Invert C++ Rasterizer::to_raster (theta=0):
  // x_r = (x - min_x) * scale
  // y_r = (rows - 1) - (y - min_y) * scale
  const spanX = Math.max(maxX - minX, 1e-12);
  const spanY = Math.max(maxY - minY, 1e-12);
  const scale = Math.min((cols - 1) / spanX, (rows - 1) / spanY);

  return {
    x: minX + xr / scale,
    y: minY + ((rows - 1) - yr) / scale,
  };
}

async function refreshTree() {
  state.tree = await apiJson("/api/tree", undefined, "Tree fetch");
  renderTree();
}

async function fetchNode(nodeId) {
  const node = await apiJson(`/api/node/${encodeURIComponent(nodeId)}`, undefined, "Node fetch");
  state.selectedNode = node;
  state.selectedNodeId = node.id;
  state.foldPoints = [];

  drawFoldSelector(node.paper_mask);
  drawHeatmap(startCtx, node.nn.start_map);
  drawHeatmap(endCtx, node.nn.end_map);

  const ckpt = node.checkpoint ? node.checkpoint.split("/").pop() : "random init";
  setStatus(`Selected node. Model: ${ckpt}`);
}

async function runNodeAction(endpoint, label) {
  if (!state.selectedNodeId) {
    throw new Error("Select a node first.");
  }

  const data = await apiJson(
    endpoint,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: state.selectedNodeId }),
    },
    label
  );

  if (!data.ok) {
    throw new Error(data.error || `${label} failed`);
  }

  state.tree = data.tree;
  renderTree();

  if (data.selected_node_id) {
    await fetchNode(data.selected_node_id);
    renderTree();
  } else if (data.selected_visible === false) {
    setStatus(`${label} ran. Selected child is hidden (unvisited).`);
    return;
  }

  setStatus(`${label} ran.`);
}

async function applyFold() {
  if (!state.selectedNode) {
    throw new Error("Select a node first.");
  }
  if (state.foldPoints.length !== 2) {
    throw new Error("Click exactly 2 points in fold selector.");
  }

  const p1 = canvasToPaperCoords(
    state.foldPoints[0].x,
    state.foldPoints[0].y,
    el.paperCanvas,
    state.selectedNode.bounds
  );
  const p2 = canvasToPaperCoords(
    state.foldPoints[1].x,
    state.foldPoints[1].y,
    el.paperCanvas,
    state.selectedNode.bounds
  );

  const data = await apiJson(
    "/api/fold",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: state.selectedNodeId, x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }),
    },
    "Fold"
  );

  if (!data.ok) {
    throw new Error(data.error || "Fold failed");
  }

  state.tree = data.tree;
  renderTree();

  if (data.new_node_id) {
    await fetchNode(data.new_node_id);
    renderTree();
  }

  setStatus("Fold applied.");
}

function renderCheckpointSelect() {
  el.checkpointSelect.innerHTML = "";

  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = state.checkpoints.length === 0 ? "No checkpoints found" : "Select checkpoint...";
  el.checkpointSelect.append(blank);

  for (const ckpt of state.checkpoints) {
    const opt = document.createElement("option");
    opt.value = ckpt.name;
    opt.textContent = ckpt.loaded ? `${ckpt.name} (loaded)` : ckpt.name;
    if (ckpt.loaded) {
      opt.selected = true;
    }
    el.checkpointSelect.append(opt);
  }

  el.loadCheckpoint.disabled = state.checkpoints.length === 0;
}

async function refreshCheckpoints() {
  const payload = await apiJson("/api/checkpoints", undefined, "Checkpoint list");
  state.checkpoints = payload.checkpoints || [];
  state.loadedCheckpoint = payload.loaded_checkpoint || null;
  renderCheckpointSelect();
}

async function loadSelectedCheckpoint() {
  const name = el.checkpointSelect.value;
  if (!name) {
    throw new Error("Select a checkpoint first.");
  }

  const payload = await apiJson(
    "/api/checkpoints/load",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    },
    "Checkpoint load"
  );

  if (!payload.ok) {
    throw new Error(payload.error || "Checkpoint load failed");
  }

  state.checkpoints = payload.checkpoints || [];
  state.loadedCheckpoint = payload.loaded_checkpoint || null;
  renderCheckpointSelect();

  if (state.selectedNodeId) {
    await fetchNode(state.selectedNodeId);
    renderTree();
  }

  const loadedName = state.loadedCheckpoint ? state.loadedCheckpoint.split("/").pop() : name;
  setStatus(`Loaded checkpoint: ${loadedName}`);
}

function bindEvents() {
  el.paperCanvas.addEventListener("click", (event) => {
    if (!state.selectedNode) {
      return;
    }

    const rect = el.paperCanvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * el.paperCanvas.width;
    const y = ((event.clientY - rect.top) / rect.height) * el.paperCanvas.height;

    if (state.foldPoints.length >= 2) {
      state.foldPoints = [];
    }
    state.foldPoints.push({ x, y });
    drawFoldSelector(state.selectedNode.paper_mask);
  });

  el.refreshTree.addEventListener("click", async () => {
    try {
      await refreshTree();
      await fetchNode(state.selectedNodeId || "root");
    } catch {
      await fetchNode("root");
      await refreshTree();
    }
  });

  el.clearPoints.addEventListener("click", () => {
    state.foldPoints = [];
    if (state.selectedNode) {
      drawFoldSelector(state.selectedNode.paper_mask);
    }
    setStatus("Points cleared.");
  });

  el.applyFold.addEventListener("click", async () => {
    try {
      await applyFold();
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  el.runExpand.addEventListener("click", async () => {
    try {
      await runNodeAction("/api/expand", "Expand");
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  el.runSelect.addEventListener("click", async () => {
    try {
      await runNodeAction("/api/select", "Select");
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  el.refreshCheckpoints.addEventListener("click", async () => {
    try {
      await refreshCheckpoints();
      setStatus("Checkpoint list refreshed.");
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  el.loadCheckpoint.addEventListener("click", async () => {
    try {
      await loadSelectedCheckpoint();
    } catch (err) {
      setStatus(err.message, true);
    }
  });
}

async function init() {
  try {
    bindEvents();
    await refreshCheckpoints();
    await refreshTree();
    await fetchNode(state.selectedNodeId);
    renderTree();
    setStatus("Ready.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

init();
