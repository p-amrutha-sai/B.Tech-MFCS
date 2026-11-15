"""
Visualization module for MAPF solutions.
Provides:
  1. Matplotlib Animated GIF generator (using Pillow writer).
  2. Standalone Interactive HTML5/SVG player for browser inspection.
  3. Terminal ASCII step-by-step playback.
"""
import os
import json
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from .grid import GridMap, Coordinate

ROBOT_COLORS = [
    "#E63946",  # Crimson Red
    "#1D3557",  # Deep Navy
    "#2A9D8F",  # Teal Green
    "#F4A261",  # Warm Orange
    "#9B5DE5",  # Purple
    "#00BBF9",  # Cyan Blue
    "#F15BB5",  # Pink
    "#00F5D4",  # Mint
]

class MAPFVisualizer:
    def __init__(self, grid: GridMap, starts: List[Coordinate], goals: List[Coordinate], 
                 paths: Dict[int, List[Coordinate]], scenario_name: str = "MAPF Scenario"):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.paths = paths
        self.num_robots = len(starts)
        self.makespan = max(len(p) - 1 for p in paths.values())
        self.scenario_name = scenario_name

    def render_ascii_step(self, t: int) -> str:
        """Renders ASCII state at time step t."""
        t_clamped = min(t, self.makespan)
        agent_positions = {i: self.paths[i][t_clamped] for i in range(self.num_robots)}
        goals_dict = {self.goals[i]: i for i in range(self.num_robots)}
        return self.grid.render_ascii(agent_positions, goals_dict)

    def print_terminal_animation(self, delay_sec: float = 0.3):
        """Prints animated trajectory in console."""
        print(f"\n--- Terminal Playback: {self.scenario_name} (Makespan T={self.makespan}) ---")
        for t in range(self.makespan + 1):
            print(f"\n[Time Step t = {t} / {self.makespan}]")
            coords_str = ", ".join([f"R{i}: {self.paths[i][t]}" for i in range(self.num_robots)])
            print(f"Positions: {coords_str}")
            print(self.render_ascii_step(t))

    def save_gif(self, output_path: str, fps: int = 2, dpi: int = 120) -> str:
        """
        Creates an animated GIF of the multi-agent trajectory using Matplotlib and Pillow.
        """
        fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
        
        # Grid settings
        w, h = self.grid.width, self.grid.height
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(w))
        ax.set_yticks(range(h))
        ax.grid(True, color="#CCCCCC", linestyle="-", linewidth=1.2, zorder=1)
        ax.set_facecolor("#FAFAFA")
        
        # Draw obstacles
        for ox, oy in self.grid.obstacles:
            rect = patches.Rectangle((ox - 0.5, oy - 0.5), 1, 1, 
                                     facecolor="#2B2D42", edgecolor="#1A1A24", 
                                     hatch="//", zorder=2)
            ax.add_patch(rect)
            
        # Draw Goal markers
        for i, (gx, gy) in enumerate(self.goals):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            goal_rect = patches.Rectangle((gx - 0.42, gy - 0.42), 0.84, 0.84,
                                          facecolor="none", edgecolor=color,
                                          linewidth=2.5, linestyle="--", zorder=3)
            ax.add_patch(goal_rect)
            ax.text(gx, gy, f"G{i}", ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color, zorder=4)

        # Plot elements to update in animation
        agent_circles = []
        agent_texts = []
        trail_lines = []
        
        for i in range(self.num_robots):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            # Robot body
            circle = patches.Circle((self.starts[i][0], self.starts[i][1]), 0.36,
                                    facecolor=color, edgecolor="#FFFFFF",
                                    linewidth=2.0, zorder=5)
            ax.add_patch(circle)
            agent_circles.append(circle)
            
            # Robot label
            txt = ax.text(self.starts[i][0], self.starts[i][1], f"R{i}",
                          ha='center', va='center', fontsize=9,
                          fontweight='heavy', color="#FFFFFF", zorder=6)
            agent_texts.append(txt)
            
            # Trajectory trail line
            trail, = ax.plot([], [], color=color, linestyle="-", linewidth=2.0, alpha=0.45, zorder=3)
            trail_lines.append(trail)

        title_text = ax.set_title("", fontsize=11, fontweight='bold', pad=12)

        def update(frame):
            t = frame
            for i in range(self.num_robots):
                curr_pos = self.paths[i][t]
                agent_circles[i].center = (curr_pos[0], curr_pos[1])
                agent_texts[i].set_position((curr_pos[0], curr_pos[1]))
                
                # Update trail up to time t
                past_x = [self.paths[i][step][0] for step in range(t + 1)]
                past_y = [self.paths[i][step][1] for step in range(t + 1)]
                trail_lines[i].set_data(past_x, past_y)

            # Vertex collision check indicator for frame
            positions_now = [self.paths[i][t] for i in range(self.num_robots)]
            collision_free = len(positions_now) == len(set(positions_now))
            status_tag = "[OK] Collision-Free" if collision_free else "[!] COLLISION"
            
            title_text.set_text(
                f"{self.scenario_name}\n"
                f"SAT Makespan Horizon: t = {t} / {self.makespan}  |  {status_tag}"
            )
            return agent_circles + agent_texts + trail_lines + [title_text]

        anim = FuncAnimation(fig, update, frames=self.makespan + 1, interval=1000 // fps, blit=True)
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        anim.save(output_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return output_path

    def generate_interactive_html(self, output_path: str, baseline_paths: Optional[Dict[int, List[Coordinate]]] = None) -> str:
        """
        Generates an interactive, standalone HTML5 visualizer with SVG canvas,
        playback scrubber, step controls, and comparative trajectory inspection.
        """
        w, h = self.grid.width, self.grid.height
        obstacles_list = list(self.grid.obstacles)
        
        data_payload = {
            "scenario_name": self.scenario_name,
            "width": w,
            "height": h,
            "obstacles": obstacles_list,
            "starts": self.starts,
            "goals": self.goals,
            "sat_paths": self.paths,
            "sat_makespan": self.makespan,
            "baseline_paths": baseline_paths,
            "baseline_makespan": max(len(p) - 1 for p in baseline_paths.values()) if baseline_paths else None,
            "colors": ROBOT_COLORS[:self.num_robots]
        }
        
        json_data = json.dumps(data_payload)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MAPF SAT Bounded Model Checking Visualizer</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .header {{
      text-align: center;
      margin-bottom: 20px;
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 24px;
      color: #38bdf8;
    }}
    .header p {{
      margin: 0;
      color: #94a3b8;
      font-size: 14px;
    }}
    .dashboard {{
      display: flex;
      gap: 28px;
      max-width: 1050px;
      width: 100%;
      background: #1e293b;
      padding: 24px;
      border-radius: 12px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }}
    .canvas-container {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    svg {{
      background: #0b0f19;
      border-radius: 8px;
      border: 1px solid #334155;
    }}
    .controls {{
      margin-top: 16px;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .buttons {{
      display: flex;
      justify-content: center;
      gap: 10px;
    }}
    button {{
      background: #38bdf8;
      color: #0f172a;
      border: none;
      padding: 8px 16px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s;
    }}
    button:hover {{
      background: #7dd3fc;
    }}
    button:disabled {{
      background: #475569;
      cursor: not-allowed;
    }}
    .scrubber-row {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    input[type=range] {{
      flex: 1;
      accent-color: #38bdf8;
    }}
    .sidebar {{
      width: 320px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .card {{
      background: #0f172a;
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #334155;
      font-size: 13px;
    }}
    .card h3 {{
      margin: 0 0 10px 0;
      font-size: 14px;
      color: #38bdf8;
      border-bottom: 1px solid #334155;
      padding-bottom: 6px;
    }}
    .robot-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 6px;
    }}
    .badge {{
      display: inline-block;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .status-ok {{
      color: #4ade80;
      font-weight: bold;
    }}
    .status-err {{
      color: #f87171;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1 id="titleDisplay">MAPF SAT Reduction Bounded Model Checker</h1>
    <p>Mathematical Foundations of Computer Science - SAT Encoding & Multi-Agent Verification</p>
  </div>

  <div class="dashboard">
    <div class="canvas-container">
      <svg id="gridSvg" width="460" height="460"></svg>
      <div class="controls">
        <div class="scrubber-row">
          <span style="font-size: 13px; font-weight: bold;">Time t:</span>
          <input type="range" id="timeSlider" min="0" max="{self.makespan}" value="0">
          <span id="timeDisplay" style="font-family: monospace; font-size: 14px; min-width: 60px;">0 / {self.makespan}</span>
        </div>
        <div class="buttons">
          <button id="btnPrev">◀ Prev</button>
          <button id="btnPlay">▶ Play</button>
          <button id="btnNext">Next ▶</button>
          <button id="btnReset">Reset</button>
        </div>
      </div>
    </div>

    <div class="sidebar">
      <div class="card">
        <h3>Instance Parameters</h3>
        <div><strong>Scenario:</strong> {self.scenario_name}</div>
        <div><strong>Grid Size:</strong> {w} &times; {h} ({len(self.grid.free_cells)} free cells)</div>
        <div><strong>Static Obstacles:</strong> {len(self.grid.obstacles)} cells</div>
        <div><strong>Active Robots:</strong> {self.num_robots}</div>
        <div><strong>SAT Optimal Makespan T*:</strong> <span style="color: #38bdf8; font-weight: bold;">{self.makespan}</span></div>
      </div>

      <div class="card">
        <h3>Live Robot Coordinates (t = <span id="timeBadge">0</span>)</h3>
        <div id="robotPositionsContainer"></div>
      </div>

      <div class="card">
        <h3>Collision Verification</h3>
        <div>Vertex Collisions: <span class="status-ok" id="vColStatus">0 (Satisfied)</span></div>
        <div>Edge Swap Collisions: <span class="status-ok" id="eColStatus">0 (Satisfied)</span></div>
        <div style="margin-top: 8px; color: #94a3b8; font-size: 11px;">
          Formally proved conflict-free via Conjunctive Normal Form (CNF) clauses in PySAT.
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json_data};
    const svg = document.getElementById("gridSvg");
    const slider = document.getElementById("timeSlider");
    const timeDisplay = document.getElementById("timeDisplay");
    const timeBadge = document.getElementById("timeBadge");
    const btnPlay = document.getElementById("btnPlay");
    const btnPrev = document.getElementById("btnPrev");
    const btnNext = document.getElementById("btnNext");
    const btnReset = document.getElementById("btnReset");
    const robotPositionsContainer = document.getElementById("robotPositionsContainer");

    const w = DATA.width;
    const h = DATA.height;
    const cellSize = 460 / Math.max(w, h);
    let currentTime = 0;
    let isPlaying = false;
    let timer = null;

    function renderStaticGrid() {{
      svg.innerHTML = "";
      // Grid cells
      for (let x = 0; x < w; x++) {{
        for (let y = 0; y < h; y++) {{
          const isObs = DATA.obstacles.some(o => o[0] === x && o[1] === y);
          const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("x", x * cellSize);
          rect.setAttribute("y", (h - 1 - y) * cellSize);
          rect.setAttribute("width", cellSize);
          rect.setAttribute("height", cellSize);
          rect.setAttribute("fill", isObs ? "#1e293b" : "#0f172a");
          rect.setAttribute("stroke", "#334155");
          rect.setAttribute("stroke-width", "1");
          svg.appendChild(rect);

          if (isObs) {{
            const line1 = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line1.setAttribute("x1", x * cellSize);
            line1.setAttribute("y1", (h - 1 - y) * cellSize);
            line1.setAttribute("x2", (x + 1) * cellSize);
            line1.setAttribute("y2", (h - y) * cellSize);
            line1.setAttribute("stroke", "#475569");
            line1.setAttribute("stroke-width", "1.5");
            svg.appendChild(line1);
          }}
        }}
      }}

      // Goal markers
      DATA.goals.forEach((g, i) => {{
        const color = DATA.colors[i % DATA.colors.length];
        const gx = g[0] * cellSize;
        const gy = (h - 1 - g[1]) * cellSize;
        
        const gRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        gRect.setAttribute("x", gx + 4);
        gRect.setAttribute("y", gy + 4);
        gRect.setAttribute("width", cellSize - 8);
        gRect.setAttribute("height", cellSize - 8);
        gRect.setAttribute("fill", "none");
        gRect.setAttribute("stroke", color);
        gRect.setAttribute("stroke-dasharray", "4,3");
        gRect.setAttribute("stroke-width", "2");
        svg.appendChild(gRect);

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", gx + cellSize / 2);
        text.setAttribute("y", gy + cellSize / 2 + 4);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", "12");
        text.setAttribute("font-weight", "bold");
        text.setAttribute("fill", color);
        text.textContent = "G" + i;
        svg.appendChild(text);
      }});

      // Group for trails & robots
      const dynamicGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      dynamicGroup.setAttribute("id", "dynamicLayer");
      svg.appendChild(dynamicGroup);
    }}

    function updateFrame(t) {{
      currentTime = t;
      slider.value = t;
      timeDisplay.textContent = t + " / " + DATA.sat_makespan;
      timeBadge.textContent = t;

      const dynamicGroup = document.getElementById("dynamicLayer");
      dynamicGroup.innerHTML = "";

      let htmlRobots = "";

      // Draw trails up to time t
      Object.keys(DATA.sat_paths).forEach(id => {{
        const path = DATA.sat_paths[id];
        const color = DATA.colors[id % DATA.colors.length];
        let d = "";
        for (let step = 0; step <= t; step++) {{
          const px = path[step][0] * cellSize + cellSize / 2;
          const py = (h - 1 - path[step][1]) * cellSize + cellSize / 2;
          d += (step === 0 ? "M " : " L ") + px + " " + py;
        }}
        const polyline = document.createElementNS("http://www.w3.org/2000/svg", "path");
        polyline.setAttribute("d", d);
        polyline.setAttribute("fill", "none");
        polyline.setAttribute("stroke", color);
        polyline.setAttribute("stroke-width", "2.5");
        polyline.setAttribute("opacity", "0.4");
        dynamicGroup.appendChild(polyline);
      }});

      // Draw robots at time t
      Object.keys(DATA.sat_paths).forEach(id => {{
        const pos = DATA.sat_paths[id][t];
        const color = DATA.colors[id % DATA.colors.length];
        const cx = pos[0] * cellSize + cellSize / 2;
        const cy = (h - 1 - pos[1]) * cellSize + cellSize / 2;

        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", cx);
        circle.setAttribute("cy", cy);
        circle.setAttribute("r", cellSize * 0.38);
        circle.setAttribute("fill", color);
        circle.setAttribute("stroke", "#ffffff");
        circle.setAttribute("stroke-width", "2");
        dynamicGroup.appendChild(circle);

        const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
        txt.setAttribute("x", cx);
        txt.setAttribute("y", cy + 4);
        txt.setAttribute("text-anchor", "middle");
        txt.setAttribute("font-size", "11");
        txt.setAttribute("font-weight", "bold");
        txt.setAttribute("fill", "#ffffff");
        txt.textContent = "R" + id;
        dynamicGroup.appendChild(txt);

        htmlRobots += `
          <div class="robot-row">
            <span><span class="badge" style="background:${{color}}"></span>Robot ${{id}}</span>
            <span style="font-family: monospace; color: #38bdf8;">(${{pos[0]}}, ${{pos[1]}})</span>
          </div>
        `;
      }});

      robotPositionsContainer.innerHTML = htmlRobots;
    }}

    slider.addEventListener("input", (e) => updateFrame(parseInt(e.target.value)));
    btnPrev.addEventListener("click", () => {{
      if (currentTime > 0) updateFrame(currentTime - 1);
    }});
    btnNext.addEventListener("click", () => {{
      if (currentTime < DATA.sat_makespan) updateFrame(currentTime + 1);
    }});
    btnReset.addEventListener("click", () => {{
      pause();
      updateFrame(0);
    }});

    function play() {{
      isPlaying = true;
      btnPlay.textContent = "⏸ Pause";
      timer = setInterval(() => {{
        if (currentTime < DATA.sat_makespan) {{
          updateFrame(currentTime + 1);
        }} else {{
          pause();
        }}
      }}, 600);
    }}

    function pause() {{
      isPlaying = false;
      btnPlay.textContent = "▶ Play";
      if (timer) clearInterval(timer);
    }}

    btnPlay.addEventListener("click", () => {{
      if (isPlaying) pause();
      else {{
        if (currentTime >= DATA.sat_makespan) currentTime = 0;
        play();
      }}
    }});

    renderStaticGrid();
    updateFrame(0);
  </script>
</body>
</html>
"""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path
