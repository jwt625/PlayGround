export class TreeVisualizer {
  constructor(container, data, options = {}) {
    this.container = container;
    this.data = data;
    this.options = {
      layout: 'vertical',
      nodeSize: 20,
      maxLineLength: 20,  // Maximum characters per line
      maxLines: 2,        // Maximum number of lines
      ...options
    };
    
    this.width = container.clientWidth;
    this.height = container.clientHeight;
    this.zoomLevel = 1;
    
    this.init();

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

  wrapText(text) {
    if (!text) return [''];
    
    const words = text.split(/\s+/);
    const lines = [];
    let currentLine = words[0];

    for (let i = 1; i < words.length; i++) {
      if (currentLine.length + words[i].length + 1 <= this.options.maxLineLength) {
        currentLine += ' ' + words[i];
      } else {
        lines.push(currentLine);
        currentLine = words[i];
        
        if (lines.length >= this.options.maxLines - 1) {
          if (i < words.length - 1) {
            currentLine = currentLine + '...';
          }
          lines.push(currentLine);
          break;
        }
      }
    }
    
    if (lines.length < this.options.maxLines && currentLine.length > 0) {
      lines.push(currentLine);
    }

    return lines;
  }

  render() {
    const isVertical = this.options.layout === 'vertical';
    const margin = { top: 50, right: 50, bottom: 50, left: 50 };
    
    const width = this.width - margin.left - margin.right;
    const height = this.height - margin.top - margin.bottom;
    
    this.treeLayout = d3.tree()
      .size(isVertical ? [width, height] : [height, width])
      .separation((a, b) => {
        return (a.parent === b.parent ? 1.5 : 2);
      });

    const root = d3.hierarchy(this.data);
    const treeData = this.treeLayout(root);

    // Define transition
    const transition = d3.transition()
      .duration(750)
      .ease(d3.easeQuadInOut);

    // Links
    const linkGenerator = isVertical ? 
      d3.linkVertical()
        .x(d => d.x)
        .y(d => d.y) :
      d3.linkHorizontal()
        .x(d => d.y)
        .y(d => d.x);

    const links = this.linksGroup
      .selectAll('.link')
      .data(treeData.links(), d => {
        return `${d.source.data.name}-${d.source.depth}-${d.target.data.name}-${d.target.depth}`;
      });

    // Remove old links with transition
    links.exit()
      .transition(transition)
      .style('opacity', 0)
      .remove();

    // Add new links with starting position
    const linksEnter = links
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('stroke', '#ccc')
      .attr('stroke-width', 2)
      .attr('fill', 'none')
      .style('opacity', 0)
      .attr('d', d => {
        const o = {
          x: d.source.x0 || d.source.x || 0,
          y: d.source.y0 || d.source.y || 0
        };
        return linkGenerator({source: o, target: o});
      });

    // Update all links with transition
    links.merge(linksEnter)
      .transition(transition)
      .style('opacity', 1)
      .attr('d', linkGenerator);

    // Nodes
    const nodes = this.nodesGroup
      .selectAll('.node')
      .data(treeData.descendants(), d => {
        return `${d.data.name}-${d.depth}-${d.parent?.data.name || 'root'}`;
      });

    // Remove old nodes with transition
    nodes.exit()
      .transition(transition)
      .style('opacity', 0)
      .remove();

    // Create new nodes with starting position
    const nodesEnter = nodes
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('opacity', 0)
      .attr('transform', d => {
        const x = d.parent?.x0 || d.parent?.x || d.x || 0;
        const y = d.parent?.y0 || d.parent?.y || d.y || 0;
        return isVertical ? 
          `translate(${x},${y})` : 
          `translate(${y},${x})`;
      });

    // Add circles to new nodes
    nodesEnter
      .append('circle')
      .attr('r', this.options.nodeSize / 2)
      .attr('fill', '#fff')
      .attr('stroke', d => d.data.url ? '#1a73e8' : '#666')
      .attr('stroke-width', 2)
      .style('cursor', d => d.data.url ? 'pointer' : 'default')
      .on('click', (event, d) => {
        event.stopPropagation();
        if (d.data.url) {
          window.open(d.data.url, '_blank');
        }
      })
      .on('mouseover', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('fill', '#f0f7ff');
      })
      .on('mouseout', function() {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('fill', '#fff');
      });

    // Add text group
    const textGroup = nodesEnter
      .append('g')
      .attr('class', 'text-group');

    // Add text background and foreground groups
    textGroup.append('g').attr('class', 'text-background');
    textGroup.append('g').attr('class', 'text-foreground');

    // Update all nodes with transition
    const allNodes = nodes.merge(nodesEnter)
      .transition(transition)
      .style('opacity', 1)
      .attr('transform', d => isVertical ?
        `translate(${d.x},${d.y})` :
        `translate(${d.y},${d.x})`
      );

    // Update text for all nodes
    this.nodesGroup.selectAll('.node').each((d, i, nodes) => {
      const node = d3.select(nodes[i]);
      const textGroup = node.select('.text-group');
      const background = textGroup.select('.text-background');
      const foreground = textGroup.select('.text-foreground');
      
      background.selectAll('*').remove();
      foreground.selectAll('*').remove();

      const lines = this.wrapText(d.data.name);
      const lineHeight = 1.2;
      
      lines.forEach((line, i) => {
        // Background
        background
          .append('text')
          .attr('dy', `${i * lineHeight}em`)
          .attr('x', d.children ? -8 : 8)
          .attr('text-anchor', d.children ? 'end' : 'start')
          .attr('stroke', 'white')
          .attr('stroke-width', 3)
          .text(line);

        // Foreground
        foreground
          .append('text')
          .attr('dy', `${i * lineHeight}em`)
          .attr('x', d.children ? -8 : 8)
          .attr('text-anchor', d.children ? 'end' : 'start')
          .attr('fill', '#000')
          .text(line);
      });

      // Add tooltip
      node.select('title').remove();
      node.append('title')
        .text(d.data.name + (d.data.url ? '\nClick node to open URL' : ''));
    });

    // Store positions for next transition
    nodes.each(d => {
      d.x0 = d.x;
      d.y0 = d.y;
    });
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