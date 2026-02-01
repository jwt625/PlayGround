<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		title = '',
		direction = 'horizontal',
		initialSize = 300,
		minSize = 100,
		collapsed = $bindable(false),
		children
	}: {
		title?: string;
		direction?: 'horizontal' | 'vertical';
		initialSize?: number;
		minSize?: number;
		collapsed?: boolean;
		children: Snippet;
	} = $props();

	let size = $state(initialSize);
	let isResizing = $state(false);
	let startPos = 0;
	let startSize = 0;

	function handleMouseDown(e: MouseEvent) {
		e.preventDefault();
		isResizing = true;
		startPos = direction === 'horizontal' ? e.clientX : e.clientY;
		startSize = size;
		window.addEventListener('mousemove', handleMouseMove);
		window.addEventListener('mouseup', handleMouseUp);
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isResizing) return;
		const currentPos = direction === 'horizontal' ? e.clientX : e.clientY;
		const delta = currentPos - startPos;
		size = Math.max(minSize, startSize + delta);
	}

	function handleMouseUp() {
		isResizing = false;
		window.removeEventListener('mousemove', handleMouseMove);
		window.removeEventListener('mouseup', handleMouseUp);
	}

	function toggleCollapse() {
		collapsed = !collapsed;
	}

	const sizeStyle = $derived(
		collapsed
			? direction === 'horizontal' ? 'width: 24px' : 'height: 24px'
			: direction === 'horizontal' ? `width: ${size}px` : `height: ${size}px`
	);
</script>

<div
	class="resizable-panel"
	class:horizontal={direction === 'horizontal'}
	class:vertical={direction === 'vertical'}
	class:collapsed
	style={sizeStyle}
>
	<div class="panel-header">
		<button class="collapse-btn" onclick={toggleCollapse} title={collapsed ? 'Expand' : 'Collapse'}>
			{#if direction === 'horizontal'}
				{collapsed ? '▶' : '◀'}
			{:else}
				{collapsed ? '▼' : '▲'}
			{/if}
		</button>
		{#if !collapsed && title}
			<span class="panel-title">{title}</span>
		{/if}
	</div>
	{#if !collapsed}
		<div class="panel-content">
			{@render children()}
		</div>
		<div
			class="resize-handle"
			class:horizontal={direction === 'horizontal'}
			class:vertical={direction === 'vertical'}
			onmousedown={handleMouseDown}
		></div>
	{/if}
</div>

<style>
	.resizable-panel {
		display: flex;
		flex-direction: column;
		position: relative;
		background: var(--bg-secondary);
		overflow: hidden;
	}

	.resizable-panel.horizontal {
		flex-direction: column;
		border-right: 1px solid var(--border-primary);
	}

	.resizable-panel.vertical {
		flex-direction: column;
		border-bottom: 1px solid var(--border-primary);
	}

	.resizable-panel.collapsed {
		flex: none !important;
	}

	.panel-header {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 4px 8px;
		background: var(--bg-tertiary);
		border-bottom: 1px solid var(--border-primary);
		min-height: 24px;
	}

	.collapse-btn {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		padding: 0 4px;
		font-size: 10px;
		line-height: 1;
	}

	.collapse-btn:hover {
		color: var(--text-primary);
	}

	.panel-title {
		font-size: 12px;
		color: var(--text-primary);
		font-weight: 500;
	}

	.panel-content {
		flex: 1;
		overflow: hidden;
	}

	.resize-handle {
		position: absolute;
		background: transparent;
		z-index: 10;
	}

	.resize-handle.horizontal {
		right: 0;
		top: 0;
		bottom: 0;
		width: 4px;
		cursor: ew-resize;
	}

	.resize-handle.vertical {
		bottom: 0;
		left: 0;
		right: 0;
		height: 4px;
		cursor: ns-resize;
	}

	.resize-handle:hover {
		background: var(--btn-primary-bg);
	}
</style>

