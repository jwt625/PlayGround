export class ZoomControls {
  constructor(viewer) {
    this.viewer = viewer;
    this.zoomSpeed = 0.02;
    this.zoomIntervals = new Map();
    this.setupZoomControls();
  }

  setupZoomControls() {
    const zoomContainer = document.createElement('div');
    zoomContainer.className = 'zoom-controls';
    zoomContainer.style.cssText = 'display: inline-flex; align-items: center; margin-right: 10px; gap: 8px;';

    // Create axis-specific zoom controls
    const axes = [
      { label: 'X', scale: (k) => ({ kx: k, ky: 1 }) },
      { label: 'Y', scale: (k) => ({ kx: 1, ky: k }) }
    ];

    axes.forEach(({ label, scale }) => {
      const axisContainer = document.createElement('div');
      axisContainer.className = 'axis-zoom-controls';
      axisContainer.style.cssText = 'display: flex; align-items: center; gap: 4px;';

      const axisLabel = document.createElement('span');
      axisLabel.textContent = label;
      axisLabel.style.cssText = 'margin: 0 4px; font-weight: bold;';

      const zoomInBtn = this.createZoomButton(`${label}+`, () => this.startZoom(scale, 1.0));
      const zoomOutBtn = this.createZoomButton(`${label}-`, () => this.startZoom(scale, -1.0));

      axisContainer.appendChild(zoomOutBtn);
      axisContainer.appendChild(axisLabel);
      axisContainer.appendChild(zoomInBtn);
      
      zoomContainer.appendChild(axisContainer);
    });

    // Reset button
    const resetBtn = this.createZoomButton('Reset', () => this.resetZoom());
    resetBtn.style.marginLeft = '8px';
    zoomContainer.appendChild(resetBtn);

    // Add to controls
    const controls = document.getElementById('controls');
    if (controls) {
      const fileLoader = controls.querySelector('.load-container');
      if (fileLoader) {
        fileLoader.after(zoomContainer);
      } else {
        controls.insertBefore(zoomContainer, controls.firstChild);
      }
    }

    // Global event listeners
    document.addEventListener('mouseup', () => this.stopAllZoom());
    document.addEventListener('mouseleave', () => this.stopAllZoom());
  }

  createZoomButton(text, onPress) {
    const button = document.createElement('button');
    button.textContent = text;
    button.className = 'control-button zoom-button';
    button.style.cssText = `
      min-width: 32px;
      height: 32px;
      padding: 4px 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      user-select: none;
    `;

    button.addEventListener('mousedown', (e) => {
      e.preventDefault();
      onPress();
    });
    button.addEventListener('mouseup', () => this.stopAllZoom());
    button.addEventListener('mouseleave', () => this.stopAllZoom());

    return button;
  }

  startZoom(scaleFn, direction) {
    this.stopAllZoom();
    
    const svg = d3.select('#tree-container svg');
    const g = svg.select('g.main-group');
    let currentTransform = d3.zoomTransform(g.node());

    const intervalId = setInterval(() => {
      const scale = Math.exp(this.zoomSpeed * direction);
      const { kx, ky } = scaleFn(scale);
      
      // Calculate new transform
      currentTransform = currentTransform.scale(kx, ky);
      
      // Apply the new transform
      g.attr('transform', currentTransform);
      
      // Update the viewer's zoom level if needed
      if (this.viewer.treeVisualizer) {
        this.viewer.treeVisualizer.zoomLevel = currentTransform.k;
      }
    }, 16);

    this.zoomIntervals.set(scaleFn.toString(), intervalId);
  }

  stopAllZoom() {
    this.zoomIntervals.forEach(intervalId => clearInterval(intervalId));
    this.zoomIntervals.clear();
  }

  resetZoom() {
    const svg = d3.select('#tree-container svg');
    const g = svg.select('g.main-group');
    
    // Create a smooth transition to identity transform
    g.transition()
      .duration(750)
      .attr('transform', d3.zoomIdentity);
      
    // Reset the viewer's zoom level
    if (this.viewer.treeVisualizer) {
      this.viewer.treeVisualizer.zoomLevel = 1;
    }
  }
}