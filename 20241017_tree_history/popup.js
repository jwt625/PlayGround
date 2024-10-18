document.addEventListener('DOMContentLoaded', function() {
  chrome.storage.local.get(['tabTree'], function(result) {
    const tabTree = result.tabTree;
    const tabTreeElement = document.getElementById('tabTree');
    
    function createTreeView(tree, element) {
      const ul = document.createElement('ul');
      for (let parentId in tree) {
        const li = document.createElement('li');
        li.textContent = `Tab ${parentId}`;
        if (tree[parentId].length > 0) {
          const childUl = document.createElement('ul');
          tree[parentId].forEach(child => {
            const childLi = document.createElement('li');
            childLi.textContent = `${child.title} (${child.url})`;
            childUl.appendChild(childLi);
          });
          li.appendChild(childUl);
        }
        ul.appendChild(li);
      }
      element.appendChild(ul);
    }

    createTreeView(tabTree, tabTreeElement);
  });
});