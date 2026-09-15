import type { ArrayConfig } from "../optics/geometry";
import type { ChannelActual } from "../optics/types";
import { phaseColor } from "./color";

const CELL_W = 44;
const CELL_H = 30;
const COLS = 4;

/** Interactive 2D schematic sharing state with the 3D console. */
export class Schematic {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly config: ArrayConfig;
  onSelect: (index: number) => void = () => {};

  constructor(canvas: HTMLCanvasElement, config: ArrayConfig) {
    this.canvas = canvas;
    this.config = config;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Schematic: no 2d context");
    this.ctx = ctx;
    canvas.width = 260;
    canvas.height = Math.ceil(config.channelIds.length / COLS) * (CELL_H + 6) + 70;
    canvas.addEventListener("click", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
      const index = this.hitTest(x, y);
      if (index >= 0) {
        this.onSelect(index);
      }
    });
  }

  private cellRect(index: number): { x: number; y: number } {
    const col = index % COLS;
    const row = Math.floor(index / COLS);
    return { x: 10 + col * (CELL_W + 6), y: 60 + row * (CELL_H + 6) };
  }

  private hitTest(x: number, y: number): number {
    for (let i = 0; i < this.config.channelIds.length; i++) {
      const r = this.cellRect(i);
      if (x >= r.x && x <= r.x + CELL_W && y >= r.y && y <= r.y + CELL_H) return i;
    }
    return -1;
  }

  update(actual: readonly ChannelActual[], selected: number, mode: string): void {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    ctx.fillStyle = "#e6edf3";
    ctx.font = "11px monospace";
    ctx.fillText(`schematic / ${mode}`, 10, 16);
    ctx.fillStyle = "#8b949e";
    ctx.fillText(`selected: ${selected >= 0 ? this.config.channelIds[selected] : "-"}`, 10, 30);

    // Splitter trunk.
    ctx.strokeStyle = "#1b6b73";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(130, 40);
    ctx.lineTo(130, 52);
    ctx.stroke();

    for (let i = 0; i < this.config.channelIds.length; i++) {
      const r = this.cellRect(i);
      const s = actual[i];
      const [rr, gg, bb] = phaseColor(s.piston_rad);
      const on = selected < 0 || selected === i;

      // Branch line from trunk to cell.
      ctx.strokeStyle = on ? "#39c5cf" : "#16323a";
      ctx.beginPath();
      ctx.moveTo(130, 52);
      ctx.lineTo(r.x + CELL_W / 2, r.y);
      ctx.stroke();

      ctx.globalAlpha = s.enabled ? 1 : 0.3;
      ctx.fillStyle = `rgb(${Math.round(rr * 255)},${Math.round(gg * 255)},${Math.round(bb * 255)})`;
      ctx.fillRect(r.x, r.y, CELL_W, CELL_H);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = selected === i ? "#ffffff" : "#0d1117";
      ctx.lineWidth = selected === i ? 2 : 1;
      ctx.strokeRect(r.x, r.y, CELL_W, CELL_H);

      ctx.fillStyle = "#05070c";
      ctx.font = "bold 10px monospace";
      ctx.fillText(this.config.channelIds[i], r.x + 4, r.y + 12);
      ctx.font = "9px monospace";
      ctx.fillText(`${((s.piston_rad * 180) / Math.PI).toFixed(0)}deg`, r.x + 4, r.y + 24);
    }
  }
}
