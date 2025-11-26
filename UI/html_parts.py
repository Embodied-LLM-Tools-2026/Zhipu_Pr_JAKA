
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Robot Monitor Demo</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-color: #050608;
      --card-bg: rgba(13, 17, 23, 0.85);
      --border-color: rgba(0, 243, 255, 0.15);
      --accent-color: #00f3ff;
      --accent-dim: rgba(0, 243, 255, 0.1);
      --text-primary: #e6f1f5;
      --text-secondary: #94a3b8;
      --success-color: #00ff9d;
      --warning-color: #ffb86c;
      --error-color: #ff5555;
      --grid-color: rgba(0, 243, 255, 0.03);
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg-color);
      color: var(--text-primary);
      min-height: 100vh;
      overflow-x: hidden;
      overflow-y: auto;
      background-image: 
        linear-gradient(var(--grid-color) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-color) 1px, transparent 1px),
        radial-gradient(circle at 50% 0%, rgba(0, 243, 255, 0.1) 0%, transparent 60%);
      background-size: 40px 40px, 40px 40px, 100% 100%;
    }

    /* Animations */
    @keyframes fade-in {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse-glow {
      0% { box-shadow: 0 0 5px var(--success-color); }
      50% { box-shadow: 0 0 15px var(--success-color); }
      100% { box-shadow: 0 0 5px var(--success-color); }
    }

    @keyframes scanline {
      0% { transform: translateY(-100%); }
      100% { transform: translateY(100%); }
    }

    @keyframes border-flash {
      0% { border-color: var(--border-color); }
      50% { border-color: var(--accent-color); }
      100% { border-color: var(--border-color); }
    }

    /* Layout */
    .app {
      display: grid;
      grid-template-columns: 3fr 2fr;
      grid-template-rows: 64px 1fr;
      min-height: 100vh;
      gap: 20px;
      padding: 20px;
    }

    /* Common Card Style */
    .hud-panel {
      position: relative;
      background: var(--card-bg);
      border: 1px solid var(--border-color);
      backdrop-filter: blur(10px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Tech Corners */
    .hud-panel::before, .hud-panel::after {
      content: '';
      position: absolute;
      width: 20px;
      height: 20px;
      border: 1px solid var(--accent-color);
      transition: all 0.3s ease;
      opacity: 0.5;
      pointer-events: none; /* 防止阻挡点击 */
    }
    .hud-panel::before { top: -1px; left: -1px; border-right: none; border-bottom: none; }
    .hud-panel::after { bottom: -1px; right: -1px; border-left: none; border-top: none; }
    
    .hud-panel:hover::before, .hud-panel:hover::after {
      width: 100%;
      height: 100%;
      opacity: 0.1;
      background: var(--accent-dim);
    }

    /* Header */
    header {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 24px;
      background: var(--card-bg);
      border-bottom: 1px solid var(--border-color);
      /* clip-path: polygon(0 0, 100% 0, 100% 70%, 98% 100%, 2% 100%, 0 70%); */
      animation: fade-in 0.5s ease-out;
    }

    h1 {
      font-family: 'JetBrains Mono', monospace;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 2px;
      color: #fff;
      text-shadow: 0 0 10px var(--accent-color);
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    h1::before {
      content: "■";
      color: var(--accent-color);
      font-size: 12px;
      animation: pulse-glow 2s infinite;
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      color: var(--success-color);
      padding: 6px 12px;
      background: rgba(0, 255, 157, 0.05);
      border: 1px solid rgba(0, 255, 157, 0.3);
      box-shadow: 0 0 10px rgba(0, 255, 157, 0.1);
    }
    
    .status-dot {
      width: 8px;
      height: 8px;
      background: var(--success-color);
      box-shadow: 0 0 8px var(--success-color);
    }

    /* Main Content - 左侧摄像头+视觉分析区域 */
    .main-view {
      display: flex;
      flex-direction: column;
      gap: 6px;  /* 紧凑间距 */
      min-height: 0;
      overflow-y: auto;
      animation: fade-in 0.6s ease-out;
    }
    
    /* 主摄像头容器 */
    .main-view .primary-cam-container {
      flex-shrink: 0;
    }
    
    /* 副摄像头横向排列 */
    .main-view .secondary-cams-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      flex-shrink: 0;
    }
    
    .main-view .secondary-cams-row .sub-cam {
      min-height: 120px;
      max-height: 160px;
    }
    
    /* VLM和SAM横向排列 */
    .main-view .vision-analysis-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      flex-shrink: 0;
    }
    
    .main-view .vision-analysis-row .card {
      padding: 10px;
    }
    
    .main-view .vision-analysis-row .vlm-preview {
      min-height: 120px;
    }
    
    /* 底部：Telemetry */
    .main-view .telemetry-bar {
      flex-shrink: 0;
    }

    .camera-stage, .primary-cam-container {
      display: flex;
      flex-direction: column;
      border: 1px solid var(--border-color);
      background: #000;
      position: relative;
      height: fit-content;
    }
    
    /* Scanline Effect */
    .camera-stage::after {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: linear-gradient(to bottom, transparent 50%, rgba(0, 243, 255, 0.03) 51%);
      background-size: 100% 3px;
      pointer-events: none;
      z-index: 10;
    }

    .primary-cam {
      position: relative;
      background: #020202;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      aspect-ratio: 4 / 3;  /* 摄像头比例 */
      max-height: 480px;
    }
    
    .primary-cam img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    .overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 5;
    }

    .secondary-cams {
      height: 160px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 2px;
      background: var(--border-color);
      border-top: 2px solid var(--border-color);
    }
    
    .secondary-cams-row .sub-cam {
      border: 1px solid var(--border-color);
      background: #050505;
      height: 100%;
    }
    
    .secondary-cams-row .sub-cam {
      border: 1px solid var(--border-color);
      background: #050505;
      height: 100%;
    }

    .sub-cam {
      position: relative;
      background: #050505;
      overflow: hidden;
      cursor: pointer;
    }
    
    .sub-cam img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      opacity: 0.7;
      transition: all 0.3s ease;
      filter: grayscale(30%);
    }
    .sub-cam:hover img { 
      opacity: 1; 
      transform: scale(1.02);
      filter: grayscale(0%);
    }

    .cam-label {
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(0,0,0,0.7);
      padding: 4px 8px;
      font-size: 10px;
      color: var(--accent-color);
      font-family: 'JetBrains Mono', monospace;
      border-left: 2px solid var(--accent-color);
      z-index: 6;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .cam-actions {
      position: absolute;
      bottom: 12px;
      right: 12px;
      display: flex;
      gap: 8px;
      z-index: 6;
      opacity: 0;
      transition: opacity 0.3s;
    }
    .camera-stage:hover .cam-actions, .sub-cam:hover .cam-actions {
      opacity: 1;
    }

    /* Telemetry Bar */
    .telemetry-bar {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1px;
      background: var(--border-color); /* For grid lines */
      border: 1px solid var(--border-color);
      padding: 1px;
    }

    .stat-item {
      background: var(--card-bg);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      position: relative;
    }
    
    .stat-item:hover {
      background: rgba(0, 243, 255, 0.05);
    }

    .stat-label {
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .stat-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 18px;
      color: var(--accent-color);
      font-weight: 600;
      text-shadow: 0 0 5px rgba(0, 243, 255, 0.3);
    }

    /* Visual Gauges */
    .gauge-container {
      display: flex;
      gap: 16px;
    }
    .mini-gauge {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 4px;
    }
    .gauge-bar-bg {
      width: 6px;
      height: 28px;
      background: rgba(255,255,255,0.1);
      position: relative;
      overflow: hidden;
    }
    .gauge-bar-fill {
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      background: var(--accent-color);
      transition: height 0.3s ease;
      box-shadow: 0 0 5px var(--accent-color);
    }
    .gauge-label {
      font-family: 'JetBrains Mono', monospace;
      font-size: 9px;
      color: var(--text-secondary);
    }

    /* Sidebar (Right) */
    .sidebar {
      display: flex;
      flex-direction: column;
      gap: 20px;
      min-height: 0;
      overflow-y: auto;
      padding-right: 4px;
      animation: fade-in 0.8s ease-out;
    }

    .sidebar::-webkit-scrollbar { width: 4px; }
    .sidebar::-webkit-scrollbar-track { background: transparent; }
    .sidebar::-webkit-scrollbar-thumb { background: var(--border-color); }

    .card {
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .card h3 {
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 2px;
      display: flex;
      align-items: center;
      gap: 10px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      padding-bottom: 8px;
    }
    
    .card h3::before {
      content: '';
      display: block;
      width: 6px;
      height: 6px;
      background: var(--accent-color);
      box-shadow: 0 0 8px var(--accent-color);
      transform: rotate(45deg);
    }

    /* VLM & Mask */
    .vlm-preview {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--border-color);
      background: #000;
      min-height: 200px;
    }
    
    .vlm-preview img {
      width: 100%;
      height: auto;
      display: block;
      transition: opacity 0.3s;
    }
    
    .loading-overlay {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.8);
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s;
      z-index: 20;
    }
    .loading-overlay.active { opacity: 1; }
    
    .spinner {
      width: 40px;
      height: 40px;
      border: 2px solid rgba(0, 243, 255, 0.1);
      border-top-color: var(--accent-color);
      border-radius: 50%;
      animation: spin 1s linear infinite;
      box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .json-box {
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px;
      color: var(--text-primary);
      background: rgba(0,0,0,0.5);
      padding: 10px;
      border-left: 2px solid var(--border-color);
      max-height: 120px;
      overflow-y: auto;
      white-space: pre-wrap;
    }

    /* Confidence Bar */
    .conf-bar-container {
      height: 4px;
      background: rgba(255,255,255,0.05);
      margin-top: 6px;
      overflow: hidden;
    }
    .conf-bar-fill {
      height: 100%;
      background: var(--accent-color);
      width: 0%;
      transition: width 0.5s ease;
      box-shadow: 0 0 5px var(--accent-color);
    }

    /* Logs & Lists */
    .log-container {
      height: 100%;
      min-height: 300px;
      flex: 1;
      overflow-y: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    
    .log-entry {
      padding: 4px 8px;
      background: rgba(0, 243, 255, 0.02);
      display: flex;
      gap: 10px;
      border-left: 2px solid transparent;
      transition: all 0.2s;
    }
    .log-entry:hover {
      background: rgba(0, 243, 255, 0.05);
      border-left-color: var(--accent-color);
    }
    
    .log-time { color: var(--text-secondary); min-width: 60px; opacity: 0.7; }
    .log-level.info { color: var(--accent-color); }
    .log-level.warning { color: var(--warning-color); }
    .log-level.error { color: var(--error-color); }
    .log-level.success { color: var(--success-color); }

    /* Buttons */
    button {
      background: rgba(0, 243, 255, 0.05);
      border: 1px solid var(--accent-color);
      color: var(--accent-color);
      padding: 8px 16px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      text-transform: uppercase;
      letter-spacing: 1px;
      clip-path: polygon(10px 0, 100% 0, 100% calc(100% - 10px), calc(100% - 10px) 100%, 0 100%, 0 10px);
      position: relative;
      z-index: 10; /* 确保按钮在最上层 */
      pointer-events: auto; /* 明确允许点击 */
    }
    
    button:hover {
      background: var(--accent-color);
      color: #000;
      box-shadow: 0 0 15px var(--accent-color);
      transform: translateY(-1px);
    }
    button:active { 
      transform: scale(0.98) translateY(0); 
      box-shadow: 0 0 8px var(--accent-color);
    }
    button:disabled {
      opacity: 0.3;
      cursor: not-allowed;
      pointer-events: none;
    }

    /* Toast Notification */
    .toast-container {
      position: fixed;
      bottom: 30px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 100;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .toast {
      background: rgba(5, 10, 15, 0.95);
      border: 1px solid var(--accent-color);
      color: var(--accent-color);
      padding: 12px 24px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      box-shadow: 0 0 20px rgba(0, 243, 255, 0.2);
      animation: fade-in 0.3s ease-out;
      display: flex;
      align-items: center;
      gap: 12px;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border-color); }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-color); }

    /* Utility */
    .row { display: flex; gap: 12px; align-items: center; position: relative; z-index: 1; }
    .flex-1 { flex: 1; }
    .text-small { font-size: 10px; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace; }
    
    /* Checkbox 样式改进 */
    input[type="checkbox"] {
      appearance: none;
      width: 14px;
      height: 14px;
      border: 1px solid var(--accent-color);
      background: rgba(0, 243, 255, 0.05);
      cursor: pointer;
      position: relative;
      transition: all 0.2s;
      z-index: 10;
    }
    input[type="checkbox"]:hover {
      background: rgba(0, 243, 255, 0.15);
      box-shadow: 0 0 8px rgba(0, 243, 255, 0.3);
    }
    input[type="checkbox"]:checked {
      background: var(--accent-color);
    }
    input[type="checkbox"]:checked::after {
      content: '✓';
      position: absolute;
      top: -2px;
      left: 1px;
      font-size: 12px;
      color: #000;
      font-weight: bold;
    }
    label {
      cursor: pointer;
      user-select: none;
      position: relative;
      z-index: 10;
    }
    
    /* Plan Steps */
    .plan-node {
      padding: 10px;
      border-left: 1px solid var(--border-color);
      background: rgba(255,255,255,0.01);
      margin-bottom: 4px;
      font-size: 12px;
      position: relative;
    }
    .plan-node::before {
      content: '';
      position: absolute;
      left: -4px;
      top: 14px;
      width: 7px;
      height: 7px;
      background: var(--bg-color);
      border: 1px solid var(--text-secondary);
      transform: rotate(45deg);
    }
    .plan-node.active {
      border-left-color: var(--warning-color);
      background: rgba(255, 184, 108, 0.05);
    }
    .plan-node.active::before {
      border-color: var(--warning-color);
      background: var(--warning-color);
      box-shadow: 0 0 8px var(--warning-color);
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="hud-panel">
      <h1>ROBOT MONITOR <span style="font-weight:400; opacity:0.6; font-size:14px;">// DASHBOARD</span></h1>
      <div class="row">
        <div class="status-badge">
          <div class="status-dot"></div>
          SYSTEM ONLINE
        </div>
        <button onclick="sendControl('pause')">PAUSE</button>
        <button onclick="sendControl('resume')">RESUME</button>
      </div>
    </header>

    <!-- Main Content Area - 左侧：摄像头区域 -->
    <div class="main-view">
      <!-- 主摄像头 (最上方) -->
      <div class="primary-cam-container hud-panel">
        <div class="primary-cam">
          <div class="cam-label">FRONT_CAM_01</div>
          <img id="cam-front" src="/api/cam/front?ts=0" alt="front feed" />
          <canvas id="front-overlay" class="overlay"></canvas>
          <div class="cam-actions">
            <button onclick="capture('front')">CAPTURE FRAME</button>
          </div>
        </div>
      </div>

      <!-- 副摄像头 (紧贴主摄像头下方，左右并排) -->
      <div class="secondary-cams-row">
        <div class="sub-cam hud-panel">
          <div class="cam-label">LEFT_CAM</div>
          <img id="cam-left" src="/api/cam/left?ts=0" alt="left feed" />
          <div class="cam-actions">
            <button onclick="capture('left')">CAP</button>
          </div>
        </div>
        <div class="sub-cam hud-panel">
          <div class="cam-label">RIGHT_CAM</div>
          <img id="cam-right" src="/api/cam/right?ts=0" alt="right feed" />
          <div class="cam-actions">
            <button onclick="capture('right')">CAP</button>
          </div>
        </div>
      </div>

      <!-- VLM 和 SAM 横向排列 -->
      <div class="vision-analysis-row">
        <!-- VLM Analysis -->
        <div class="card hud-panel">
          <h3>VLM</h3>
          <div class="vlm-preview">
            <div id="vlm-loader" class="loading-overlay"><div class="spinner"></div></div>
            <img id="vlm-img" src="" alt="Waiting for capture..." onload="syncCanvasSize()" />
            <canvas id="overlay" class="overlay"></canvas>
          </div>
          <div id="bbox-json" class="json-box" style="max-height:60px;">Waiting...</div>
        </div>

        <!-- SAM Mask -->
        <div class="card hud-panel">
          <h3>SAM</h3>
          <div class="vlm-preview">
            <img id="sam-mask-img" src="" style="display:none;" />
            <div id="sam-mask-placeholder" style="padding:15px; text-align:center; color:#666; font-size:11px;">
              NO MASK
            </div>
          </div>
          <div class="row" style="justify-content: space-between; margin-top:2px;">
              <div id="sam-mask-meta" class="text-small"></div>
              <div class="conf-bar-container" style="width: 50px;">
                  <div id="mask-conf-bar" class="conf-bar-fill"></div>
              </div>
          </div>
        </div>
      </div>

      <!-- Bottom: Telemetry Data -->
      <div class="telemetry-bar hud-panel">
        <div class="stat-item">
          <span class="stat-label">AGV X</span>
          <span class="stat-value"><span id="agv-x">0.00</span> m</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">AGV Y</span>
          <span class="stat-value"><span id="agv-y">0.00</span> m</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">θ</span>
          <span class="stat-value"><span id="theta">0.00</span> rad</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">ORIENT</span>
          <div class="gauge-container">
            <div class="mini-gauge" title="Yaw">
              <div class="gauge-bar-bg"><div id="gauge-yaw" class="gauge-bar-fill" style="height:50%"></div></div>
              <span class="gauge-label">Y</span>
            </div>
            <div class="mini-gauge" title="Pitch">
              <div class="gauge-bar-bg"><div id="gauge-pitch" class="gauge-bar-fill" style="height:50%"></div></div>
              <span class="gauge-label">P</span>
            </div>
            <div class="mini-gauge" title="Roll">
              <div class="gauge-bar-bg"><div id="gauge-roll" class="gauge-bar-fill" style="height:50%"></div></div>
              <span class="gauge-label">R</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column: System Logs + Status -->
    <div class="sidebar">
      
      <!-- System Logs (最上方) -->
      <div class="card hud-panel system-log-panel" style="height: 500px; flex-shrink: 0;">
        <h3>System Logs</h3>
        <div class="row">
            <span id="log-count-main" class="text-small flex-1">0 entries</span>
            <label style="display:flex; align-items:center; gap:4px; font-size:10px; color:var(--text-secondary); cursor:pointer;">
              <input type="checkbox" id="auto-scroll-logs-main" checked style="cursor:pointer;">
              Auto
            </label>
            <button onclick="clearTaskLogs()" style="padding:2px 6px; font-size:10px;">CLEAR</button>
        </div>
        <div id="task-logs-main" class="log-container" style="height: 100%; overflow-y: auto;"></div>
      </div>

      <!-- World Model -->
      <div class="card hud-panel">
        <h3>World Model</h3>
        <div class="text-small" id="world-model-updated">Last update: --</div>
        <div id="world-model-entries" class="log-container" style="height: 120px;">
          <div style="padding:8px; color:#666;">No objects detected</div>
        </div>
      </div>

      <!-- Behavior Tree -->
      <div class="card hud-panel">
        <h3>Plan Execution</h3>
        <div class="text-small" id="plan-updated">Status: Idle</div>
        <div id="plan-steps" class="log-container" style="height: 150px;">
          <!-- Steps injected here -->
        </div>
        <div style="display:none;">
            <pre id="plan-tree-json"></pre>
            <div id="execution-timeline"></div>
        </div>
      </div>

      <!-- Suggestions -->
      <div class="card hud-panel">
        <h3>System Suggestions</h3>
        <div class="row">
            <span id="suggestion-count" class="text-small flex-1">0 items</span>
            <button onclick="clearSuggestions()" style="padding:2px 6px; font-size:10px;">CLEAR</button>
        </div>
        <div id="suggestion-list" class="log-container" style="height: 100px;"></div>
      </div>

    </div>
  </div>

  <!-- Toast Container -->
  <div id="toast-container" class="toast-container"></div>

  <!-- Scripts -->
  <script>
  let lastVlmTs = -1;
  let lastWorldTs = -1;
  let lastPlanTs = -1;
  let lastSuggestionSize = 0;

  // Toast Function
  function showToast(msg) {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = 'toast';
    el.innerHTML = `<span>ℹ️</span> ${msg}`;
    container.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
    }, 3000);
  }

  // 1. Camera Feeds
  setInterval(() => {
    ['front','left','right'].forEach(id => {
      const img = document.getElementById('cam-' + id);
      if (!img) return;
      img.src = '/api/cam/' + id + '?ts=' + Date.now();
      if (id === 'front') {
        syncCanvasSize(document.getElementById('front-overlay'), img);
      }
    });
  }, 100);

  // 2. Capture & VLM
  async function capture(which) {
    showToast(`Capturing from ${which.toUpperCase()}...`);
    const loader = document.getElementById('vlm-loader');
    if(loader) loader.classList.add('active');
    
    await fetch('/api/capture?cam=' + which);
    clearBoxes();
    clearBoxes('front-overlay');
    const box = document.getElementById('bbox-json');
    if(box) box.textContent = 'Analyzing...';
  }

  let lastCapturePath = '';
  async function pollLatestCapture() {
    try {
      const r = await fetch('/api/capture/latest');
      const j = await r.json();
      if (j.url && j.url !== lastCapturePath) {
        lastCapturePath = j.url;
        const imgEl = document.getElementById('vlm-img');
        if(imgEl) {
            imgEl.src = j.url + '?ts=' + Date.now();
            // Remove loader when image loads
            imgEl.onload = () => {
                const loader = document.getElementById('vlm-loader');
                if(loader) loader.classList.remove('active');
                syncCanvasSize();
            };
        }
        clearBoxes();
        clearBoxes('front-overlay');
      }
    } catch(e) {}
  }
  setInterval(pollLatestCapture, 500);

  async function pollVlmResult() {
    try {
      const resp = await fetch('/api/vlm/latest');
      if (!resp.ok) return;
      const data = await resp.json();
      if (!data || typeof data.ts !== 'number') return;
      if (data.ts === lastVlmTs) return;
      
      lastVlmTs = data.ts;
      const boxes = Array.isArray(data.boxes) ? data.boxes : (data.bbox ? [data.bbox] : []);
      
      if (boxes.length) {
        drawBoxes(boxes);
        drawBoxes(boxes, 'front-overlay', 'cam-front');
      } else {
        clearBoxes();
        clearBoxes('front-overlay');
      }

      const bboxJson = document.getElementById('bbox-json');
      if (bboxJson) {
        const summary = {
          bbox: data.bbox,
          mapped_bbox: data.mapped_bbox,
          confidence: data.confidence,
          range: data.range_estimate,
          analysis: data.analysis,
          mask_score: data.mask_score,
        };
        bboxJson.textContent = JSON.stringify(summary, null, 2);
      }

      if (data.annotated_url) {
        const imgEl = document.getElementById('vlm-img');
        if (imgEl) imgEl.src = data.annotated_url + '?ts=' + Date.now();
      }
      updateMaskPreview(data);
    } catch(e) { console.error(e); }
  }
  setInterval(pollVlmResult, 500);

  function drawBoxes(boxes, canvasId = 'overlay', imgId = 'vlm-img') {
    const canvas = document.getElementById(canvasId);
    const img = document.getElementById(imgId);
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    syncCanvasSize(canvas, img);
    ctx.clearRect(0,0,canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00f3ff';
    ctx.shadowBlur = 4;
    ctx.shadowColor = '#00f3ff';
    boxes.forEach(b => {
      const [x1,y1,x2,y2] = b;
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    });
  }

  function clearBoxes(canvasId = 'overlay') {
    const canvas = document.getElementById(canvasId);
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0,0,canvas.width,canvas.height);
    }
  }

  function syncCanvasSize(canvas = document.getElementById('overlay'), img = document.getElementById('vlm-img')) {
    if (!canvas || !img) return;
    const r = img.getBoundingClientRect();
    canvas.width = r.width;
    canvas.height = r.height;
    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
  }

  // 3. Telemetry
  async function tick() {
    try {
      // Telemetry
      const t = await (await fetch('/api/telemetry')).json();
      
      // Update Gauges
      const updateGauge = (id, val) => {
        const el = document.getElementById(id);
        // Normalize -3.14 to 3.14 -> 0% to 100%
        const pct = ((val + 3.14) / 6.28) * 100;
        if(el) el.style.height = Math.max(0, Math.min(100, pct)) + '%';
      };
      
      updateGauge('gauge-yaw', t.orientation.yaw);
      updateGauge('gauge-pitch', t.orientation.pitch);
      updateGauge('gauge-roll', t.orientation.roll);

      // AGV Pose
      try {
        const agv = await (await fetch('/api/agv/pose')).json();
        if (agv.status === 'ok' && agv.pose) {
            document.getElementById('theta').textContent = agv.pose.theta.toFixed(2);
            document.getElementById('agv-x').textContent = agv.pose.x.toFixed(2);
            document.getElementById('agv-y').textContent = agv.pose.y.toFixed(2);
        }
      } catch(e) {}

    } catch(e) {}
  }
  setInterval(tick, 200);

  // 4. Logs
  let lastLogCount = 0;
  async function pollTaskLogs() {
    try {
      const r = await fetch('/api/task/logs');
      const j = await r.json();
      const logs = j.logs || [];
      renderTaskLogs(logs);
    } catch(e) {}
  }
  
  function renderTaskLogs(logs) {
    const container = document.getElementById('task-logs-main');
    const countEl = document.getElementById('log-count-main');
    const autoScrollCheckbox = document.getElementById('auto-scroll-logs-main');
    if(!container) return;
    
    if(countEl) countEl.textContent = logs.length + ' entries';
    
    if (logs.length > lastLogCount) {
      for (let i = lastLogCount; i < logs.length; i++) {
        const log = logs[i];
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `
          <span class="log-time">${log.time}</span>
          <span class="log-level ${log.level}">[${log.level}]</span>
          <span class="log-message">${log.message}</span>
        `;
        container.appendChild(entry);
      }
      // Auto-scroll to bottom if checkbox is checked
      if (autoScrollCheckbox && autoScrollCheckbox.checked) {
        container.scrollTop = container.scrollHeight;
      }
      lastLogCount = logs.length;
    } else if (logs.length < lastLogCount) {
      container.innerHTML = '';
      lastLogCount = 0;
    }
  }
  
  async function clearTaskLogs() {
    await fetch('/api/task/logs', { method: 'DELETE' });
    document.getElementById('task-logs-main').innerHTML = '';
    lastLogCount = 0;
    showToast('Logs cleared');
  }
  setInterval(pollTaskLogs, 500);

  // 5. Mask Preview
  function updateMaskPreview(data) {
    const maskImg = document.getElementById('sam-mask-img');
    const placeholder = document.getElementById('sam-mask-placeholder');
    const meta = document.getElementById('sam-mask-meta');
    const bar = document.getElementById('mask-conf-bar');
    
    if (!maskImg) return;

    if (data && data.mask_url) {
      maskImg.src = data.mask_url + '?ts=' + Date.now();
      maskImg.style.display = 'block';
      if(placeholder) placeholder.style.display = 'none';
      
      const score = data.mask_score || 0;
      if(meta) meta.textContent = `Score: ${score.toFixed(3)}`;
      if(bar) bar.style.width = (score * 100) + '%';
      
    } else {
      maskImg.style.display = 'none';
      if(placeholder) placeholder.style.display = 'block';
      if(meta) meta.textContent = '';
      if(bar) bar.style.width = '0%';
    }
  }

  // 6. World Model
  async function pollWorldModel() {
    try {
      const resp = await fetch('/api/world_model');
      if (!resp.ok) return;
      const data = await resp.json();
      if (!data || (typeof data.ts === 'number' && data.ts === lastWorldTs)) return;
      
      lastWorldTs = data.ts || Date.now();
      renderWorldModel(data);
    } catch (e) {}
  }

  function renderWorldModel(state) {
    const statusEl = document.getElementById('world-model-updated');
    const listEl = document.getElementById('world-model-entries');
    if (!listEl) return;
    
    const snapshot = state.snapshot || {};
    if(statusEl) statusEl.textContent = `Goal: ${snapshot.goal || 'None'}`;
    
    const objects = snapshot.objects || {};
    const entries = Object.entries(objects);
    
    if (!entries.length) {
      listEl.innerHTML = '<div style="padding:8px; color:#666;">No objects detected</div>';
      return;
    }
    
    listEl.innerHTML = entries.map(([id, obj]) => {
      const dist = obj.attrs?.range_estimate?.toFixed(2) || '?';
      const conf = obj.confidence || 0;
      return `
        <div class="log-entry" style="flex-direction:column; gap:2px;">
          <div style="display:flex; justify-content:space-between; color:var(--accent-color);">
            <span>${id}</span>
            <span>${dist}m</span>
          </div>
          <div class="row" style="width:100%">
            <div class="conf-bar-container flex-1"><div class="conf-bar-fill" style="width:${conf*100}%"></div></div>
            <span class="text-small">${(conf*100).toFixed(0)}%</span>
          </div>
        </div>
      `;
    }).join('');
  }
  setInterval(pollWorldModel, 1000);

  // 7. Plan & Suggestions
  async function pollPlanState() {
    try {
      const resp = await fetch('/api/plan');
      if (!resp.ok) return;
      const data = await resp.json();
      if (!data || (typeof data.ts === 'number' && data.ts === lastPlanTs)) return;
      lastPlanTs = data.ts || Date.now();
      renderPlanState(data);
    } catch (e) {}
  }

  function renderPlanState(state) {
    const statusEl = document.getElementById('plan-updated');
    const stepsEl = document.getElementById('plan-steps');
    if (!stepsEl) return;

    const steps = state.steps || [];
    if(statusEl) statusEl.textContent = `Source: ${state.metadata?.source || 'Unknown'}`;
    
    if (!steps.length) {
      stepsEl.innerHTML = '<div style="padding:8px; color:#666;">No active plan</div>';
      return;
    }

    const currentIndex = state.current_index ?? -1;
    stepsEl.innerHTML = steps.map((step, idx) => {
      const label = step.name || step.type || `Node ${idx}`;
      return `
        <div class="plan-node ${idx === currentIndex ? 'active' : ''}">
          ${idx + 1}. ${label}
        </div>
      `;
    }).join('');
  }
  setInterval(pollPlanState, 1000);

  async function pollSuggestions() {
    try {
      const resp = await fetch('/api/suggestions');
      if (!resp.ok) return;
      const data = await resp.json();
      renderSuggestions(data.suggestions || []);
    } catch (e) {}
  }

  function renderSuggestions(items) {
    const listEl = document.getElementById('suggestion-list');
    const countEl = document.getElementById('suggestion-count');
    if (!listEl) return;
    
    if(countEl) countEl.textContent = items.length + ' items';
    
    if (!items.length) {
      listEl.innerHTML = '<div style="padding:8px; color:#666;">No suggestions</div>';
      lastSuggestionSize = 0;
      return;
    }
    
    if (items.length === lastSuggestionSize) return;
    lastSuggestionSize = items.length;
    
    listEl.innerHTML = items.slice(-20).map(entry => `
      <div class="log-entry">
        <span class="log-level ${entry.level}">${entry.level}</span>
        <span>${entry.message}</span>
      </div>
    `).join('');
    listEl.scrollTop = listEl.scrollHeight;
  }
  
  async function clearSuggestions() {
    await fetch('/api/suggestions', { method: 'DELETE' });
    renderSuggestions([]);
    showToast('Suggestions cleared');
  }
  setInterval(pollSuggestions, 2000);

  async function sendControl(action) {
    showToast(`Sending command: ${action.toUpperCase()}`);
    await fetch('/api/task/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
    });
  }
  </script>
</body>
</html>
"""
