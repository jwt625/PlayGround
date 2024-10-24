export class TreeVisualizer {
  constructor(container, data, options = {}) {
    this.container = container;
    this.data = data;
    this.options = {
      layout: 'vertical',
      nodeSize: 20,
      ...options
    };
    
    // Set initial dimensions
    this.width = container.clientWidth;
    this.height = container.clientHeight;
    this.zoomLevel = 1;
    
    // Initialize visualization
    this.init();

    // Add resize handler
    window.addEventListener('resize', () => {
      this.width = container.clientWidth;
      this.height = container.clientHeight;
      this.render();
    });
  }

  init() {
    // Clear any existing content
    this.container.innerHTML = '';

    // Create SVG container with zoom behavior
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', '100%')
      .attr('height', '100%')
      .call(d3.zoom()
        .scaleExtent([0.1, 3])
        .on('zoom', (event) => this.handleZoom(event)))
      .append('g')
      .attr('class', 'main-group');

    // Create groups for links and nodes
    this.linksGroup = this.svg.append('g').attr('class', 'links');
    this.nodesGroup = this.svg.append('g').attr('class', 'nodes');

    // Initial render
    this.render();
  }

  render() {
    const isVertical = this.options.layout === 'vertical';
    const margin = { top: 50, right: 50, bottom: 50, left: 50 };
    
    // Calculate available space
    const width = this.width - margin.left - margin.right;
    const height = this.height - margin.top - margin.bottom;
    
    // Create tree layout
    this.treeLayout = d3.tree()
      .size(isVertical ? [width, height] : [height, width])
      .separation((a, b) => {
        return (a.parent === b.parent ? 1 : 1.2);
      });

    // Create root hierarchy
    const root = d3.hierarchy(this.data);
    
    // Calculate tree layout
    const treeData = this.treeLayout(root);

    // Create links
    const linkGenerator = isVertical ? 
      d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y) :
      d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x);

    // Update links
    const links = this.linksGroup
      .selectAll('.link')
      .data(treeData.links(), d => d.target.data.name + d.source.data.name);

    links.exit().remove();

    const linksEnter = links
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('stroke', '#ccc')
      .attr('stroke-width', 2)
      .attr('fill', 'none');

    links.merge(linksEnter)
      .transition()
      .duration(750)
      .attr('d', linkGenerator);

    // Update nodes
    const nodes = this.nodesGroup
      .selectAll('.node')
      .data(treeData.descendants(), d => d.data.name);

    nodes.exit().remove();

    const nodesEnter = nodes
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('cursor', 'pointer')
      .on('click', (event, d) => {
        if (this.options.onNodeClick) {
          this.options.onNodeClick(d.data);
        }
      });

    // Add circles to new nodes
    nodesEnter
      .append('circle')
      .attr('r', this.options.nodeSize / 2)
      .attr('fill', '#fff')
      .attr('stroke', '#1a73e8')
      .attr('stroke-width', 2);

    // Add text background to new nodes
    nodesEnter
      .append('text')
      .attr('class', 'text-bg')
      .attr('dy', '0.31em')
      .attr('stroke', 'white')
      .attr('stroke-width', 3);

    // Add text to new nodes
    nodesEnter
      .append('text')
      .attr('class', 'text-main')
      .attr('dy', '0.31em');

    // Update all nodes
    const allNodes = nodes.merge(nodesEnter);

    // Transition nodes to their new position
    allNodes.transition()
      .duration(750)
      .attr('transform', d => isVertical ?
        `translate(${d.x},${d.y})` :
        `translate(${d.y},${d.x})`
      );

    // Update text position and content
    allNodes.selectAll('text')
      .attr('x', d => {
        const offset = 6;
        return isVertical ? 
          (d.children ? -offset : offset) :
          (d.children ? -offset : offset);
      })
      .attr('text-anchor', d => {
        return isVertical ?
          (d.children ? 'end' : 'start') :
          (d.children ? 'end' : 'start');
      })
      .text(d => d.data.name);
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