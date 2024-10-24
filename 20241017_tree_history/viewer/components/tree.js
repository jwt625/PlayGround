export class TreeVisualizer {
  constructor(container, data, options = {}) {
    this.container = container;
    this.data = data;
    this.options = {
      layout: 'vertical',
      nodeSize: 20,
      ...options
    };
    
    this.width = container.clientWidth;
    this.height = container.clientHeight;
    this.zoomLevel = 1;
    
    this.init();
  }

  init() {
    // Clear any existing content
    this.container.innerHTML = '';

    // Create SVG container
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .call(d3.zoom().on('zoom', (event) => this.handleZoom(event)))
      .append('g');

    this.render();
  }

  render() {
    const isVertical = this.options.layout === 'vertical';
    
    // Create tree layout
    this.treeLayout = d3.tree()
      .size(isVertical ? 
        [this.width - 100, this.height - 100] : 
        [this.height - 100, this.width - 100]);

    // Create root hierarchy
    const root = d3.hierarchy(this.data);
    
    // Calculate tree layout
    const treeData = this.treeLayout(root);
    
    // Draw links
    const links = this.svg.selectAll('.link')
      .data(treeData.links())
      .join('path')
      .attr('class', 'link')
      .attr('d', isVertical ? 
        d3.linkVertical()
          .x(d => d.x)
          .y(d => d.y) :
        d3.linkHorizontal()
          .x(d => d.y)
          .y(d => d.x)
      );

    // Create node groups
    const nodes = this.svg.selectAll('.node')
      .data(treeData.descendants())
      .join('g')
      .attr('class', 'node')
      .attr('transform', d => isVertical ?
        `translate(${d.x},${d.y})` :
        `translate(${d.y},${d.x})`
      )
      .on('click', (event, d) => {
        if (this.options.onNodeClick) {
          this.options.onNodeClick(d.data);
        }
      });

    // Add circles to nodes
    nodes.selectAll('circle')
      .data(d => [d])
      .join('circle')
      .attr('r', this.options.nodeSize / 2);

    // Add text labels
    nodes.selectAll('text')
      .data(d => [d])
      .join('text')
      .attr('dy', '0.31em')
      .attr('x', d => d.children ? -6 : 6)
      .attr('text-anchor', d => d.children ? 'end' : 'start')
      .text(d => d.data.name)
      .clone(true).lower()
      .attr('stroke', 'white')
      .attr('stroke-width', 3);
  }

  updateData(newData) {
    this.data = newData;
    this.render();
  }

  setLayout(layout) {
    this.options.layout = layout;
    this.render();
  }

  handleZoom(event) {
    this.svg.attr('transform', event.transform);
    this.zoomLevel = event.transform.k;
  }
}