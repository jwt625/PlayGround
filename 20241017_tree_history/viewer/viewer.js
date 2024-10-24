import { TreeVisualizer } from './components/tree.js';
import { ViewerControls } from './components/controls.js';

class TabTreeViewer {
  constructor() {
    this.treeVisualizer = null;
    this.controls = null;
    this.currentLayout = 'vertical';
    this.isViewerTab = true;  // Flag to indicate this is a viewer tab
    this.init();
  }

  async init() {
    try {
      // Show loading state
      this.showLoading(true);

      // Mark this tab as a viewer tab
      if (window.chrome && chrome.runtime) {
        await chrome.runtime.sendMessage({ 
          action: "registerViewer",
          tabId: await this.getTabId()
        });
      }

      // Initialize controls
      this.controls = new ViewerControls(this);
      
      // Get initial tree data
      const { tabTree } = await this.requestData();
      
      // Initialize tree visualization
      this.treeVisualizer = new TreeVisualizer(
        document.getElementById('tree-container'),
        this.processTreeData(tabTree),
        {
          layout: this.currentLayout,
          onNodeClick: this.handleNodeClick.bind(this)
        }
      );

      // Set up message listener for updates
      this.setupMessageListener();
      
      // Hide loading state
      this.showLoading(false);

      // Clean up when viewer is closed
      window.addEventListener('unload', () => {
        if (window.chrome && chrome.runtime) {
          chrome.runtime.sendMessage({ 
            action: "unregisterViewer",
            tabId: this.getTabId()
          });
        }
      });

    } catch (error) {
      console.error('Initialization error:', error);
      this.showError('Failed to initialize viewer');
    }
  }

  async getTabId() {
    return new Promise((resolve) => {
      if (window.chrome && chrome.tabs) {
        chrome.tabs.getCurrent(tab => resolve(tab.id));
      } else {
        resolve(null);
      }
    });
  }
  
  async requestData() {
    return new Promise((resolve) => {
      chrome.runtime.sendMessage({ action: "getTabTree" }, response => {
        resolve(response || { tabTree: {} });
      });
    });
  }

  processTreeData(rawTree) {
    // Convert the raw tree data into D3-friendly format
    const root = {
      name: 'Root',
      children: []
    };

    Object.values(rawTree).forEach(node => {
      root.children.push(this.processNode(node));
    });

    return root;
  }

  processNode(node) {
    return {
      name: node.title || node.url,
      url: node.url,
      data: node,
      children: node.children ? node.children.map(child => this.processNode(child)) : []
    };
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      if (message.action === 'treeUpdated') {
        this.handleTreeUpdate(message.data);
      }
      sendResponse({ received: true });
      return true;
    });
  }

  async handleTreeUpdate(newTree) {
    if (this.treeVisualizer) {
      this.treeVisualizer.updateData(this.processTreeData(newTree));
    }
  }

  toggleLayout() {
    this.currentLayout = this.currentLayout === 'vertical' ? 'horizontal' : 'vertical';
    if (this.treeVisualizer) {
      this.treeVisualizer.setLayout(this.currentLayout);
    }
  }

  handleNodeClick(node) {
    console.log('Node clicked:', node);
    // We can expand this later with more functionality
  }

  showLoading(show) {
    const loader = document.getElementById('loading');
    if (loader) {
      loader.style.display = show ? 'block' : 'none';
    }
  }

  showError(message) {
    // Basic error handling for now
    alert(message);
  }
}

// Initialize the viewer when the page loads
window.addEventListener('DOMContentLoaded', () => {
  window.viewer = new TabTreeViewer();
});