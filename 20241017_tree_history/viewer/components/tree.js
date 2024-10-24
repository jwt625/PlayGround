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

  
  // Add text wrapping helper function
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
        
        // Check if we've reached max lines
        if (lines.length >= this.options.maxLines - 1) {
          // Add last line with ellipsis if there are more words
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
        return (a.parent === b.parent ? 1.5 : 2); // Increased separation for wrapped text
      });

    const root = d3.hierarchy(this.data);
    const treeData = this.treeLayout(root);

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

    // Nodes
    const nodes = this.nodesGroup
      .selectAll('.node')
      .data(treeData.descendants(), d => d.data.name);

    nodes.exit().remove();

    // Create new nodes
    const nodesEnter = nodes
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('cursor', 'pointer');

    // Add circles to new nodes
    nodesEnter
      .append('circle')
      .attr('r', this.options.nodeSize / 2)
      .attr('fill', '#fff')
      .attr('stroke', '#1a73e8')
      .attr('stroke-width', 2)
      .on('click', (event, d) => {
        if (this.options.onNodeClick) {
          this.options.onNodeClick(d.data);
        }
      });

    // Create clickable link group
    const linkGroup = nodesEnter
      .append('g')
      .attr('class', 'link-group')
      .on('click', (event, d) => {
        event.stopPropagation(); // Prevent node click handler
        if (d.data.url) {
          window.open(d.data.url, '_blank');
        }
      });

    // Add text background and foreground to link group
    linkGroup
      .append('g')
      .attr('class', 'text-background');

    linkGroup
      .append('g')
      .attr('class', 'text-foreground');

    // Update all nodes
    const allNodes = nodes.merge(nodesEnter);

    // Transition nodes to their new position
    allNodes.transition()
      .duration(750)
      .attr('transform', d => isVertical ?
        `translate(${d.x},${d.y})` :
        `translate(${d.y},${d.x})`
      );

    // Update text for all nodes
    allNodes.each((d, i, nodes) => {
      const node = d3.select(nodes[i]);
      const linkGroup = node.select('.link-group');
      const background = linkGroup.select('.text-background');
      const foreground = linkGroup.select('.text-foreground');
      
      // Clear existing text
      background.selectAll('*').remove();
      foreground.selectAll('*').remove();

      // Get wrapped text
      const lines = this.wrapText(d.data.name);
      const lineHeight = 1.2; // em units
      
      // Add background and foreground text
      lines.forEach((line, i) => {
        // Background (white outline)
        background
          .append('text')
          .attr('dy', `${i * lineHeight}em`)
          .attr('x', d.children ? -8 : 8)
          .attr('text-anchor', d.children ? 'end' : 'start')
          .attr('stroke', 'white')
          .attr('stroke-width', 3)
          .text(line);

        // Foreground (actual text)
        foreground
          .append('text')
          .attr('dy', `${i * lineHeight}em`)
          .attr('x', d.children ? -8 : 8)
          .attr('text-anchor', d.children ? 'end' : 'start')
          .attr('fill', d.data.url ? '#1a73e8' : '#000')  // Blue for clickable links
          .attr('text-decoration', d.data.url ? 'underline' : 'none')
          .text(line);
      });

      // Add title attribute for hover tooltip showing full text
      linkGroup.select('title').remove();
      linkGroup.append('title')
        .text(d.data.name + (d.data.url ? '\nClick to open URL' : ''));
    });

    // Add hover effect for clickable nodes
    allNodes
      .select('.link-group')
      .style('cursor', d => d.data.url ? 'pointer' : 'default')
      .on('mouseover', function(event, d) {
        if (d.data.url) {
          d3.select(this).select('.text-foreground')
            .selectAll('text')
            .attr('fill', '#1557b0');
        }
      })
      .on('mouseout', function(event, d) {
        if (d.data.url) {
          d3.select(this).select('.text-foreground')
            .selectAll('text')
            .attr('fill', '#1a73e8');
        }
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