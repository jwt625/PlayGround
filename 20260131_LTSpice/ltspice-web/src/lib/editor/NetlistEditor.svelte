<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EditorState } from '@codemirror/state';
	import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter } from '@codemirror/view';
	import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
	import { spiceLanguage } from './spice-language';
	import { spiceEditorTheme, spiceSyntaxHighlighting } from './spice-theme';

	let { value = $bindable(''), onchange }: { value: string; onchange?: (value: string) => void } = $props();

	let editorContainer: HTMLDivElement;
	let view: EditorView | null = null;

	onMount(() => {
		const updateListener = EditorView.updateListener.of((update) => {
			if (update.docChanged) {
				const newValue = update.state.doc.toString();
				value = newValue;
				onchange?.(newValue);
			}
		});

		const state = EditorState.create({
			doc: value,
			extensions: [
				lineNumbers(),
				highlightActiveLine(),
				highlightActiveLineGutter(),
				history(),
				keymap.of([...defaultKeymap, ...historyKeymap]),
				spiceLanguage,
				spiceEditorTheme,
				spiceSyntaxHighlighting,
				updateListener,
				EditorView.lineWrapping
			]
		});

		view = new EditorView({
			state,
			parent: editorContainer
		});
	});

	onDestroy(() => {
		view?.destroy();
	});

	// Sync external value changes to editor
	$effect(() => {
		if (view && value !== view.state.doc.toString()) {
			view.dispatch({
				changes: { from: 0, to: view.state.doc.length, insert: value }
			});
		}
	});
</script>

<div class="editor-wrapper" bind:this={editorContainer}></div>

<style>
	.editor-wrapper {
		width: 100%;
		height: 100%;
		overflow: hidden;
	}

	.editor-wrapper :global(.cm-editor) {
		height: 100%;
	}

	.editor-wrapper :global(.cm-scroller) {
		overflow: auto;
	}
</style>

