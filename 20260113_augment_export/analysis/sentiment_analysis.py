#!/usr/bin/env python3
"""
Sentiment analysis on all user messages using Transformer model.
Saves results with full metadata for traceability back to original conversations.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys


def load_user_messages(data_dir: str) -> List[Dict[str, Any]]:
    """Load all user messages with metadata."""
    messages = []
    data_dir = Path(data_dir)
    
    print(f"Loading user messages from {data_dir}...")
    
    for json_file in sorted(data_dir.glob("*.json")):
        if json_file.name == "extraction_summary.json":
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            workspace_id = json_file.stem
            folder_path = data.get('folder_path', '')
            
            for conv_idx, conv in enumerate(data.get('conversations', [])):
                conv_id = conv.get('conversationId', f'conv_{conv_idx}')
                
                for exch_idx, exchange in enumerate(conv.get('exchanges', [])):
                    request = exchange.get('request_message', '')
                    
                    if request and request.strip():
                        messages.append({
                            'text': request,
                            'workspace_id': workspace_id,
                            'workspace_path': folder_path,
                            'conversation_id': conv_id,
                            'exchange_index': exch_idx,
                            'timestamp': exchange.get('timestamp', ''),
                            'source_file': json_file.name,
                        })
        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(messages):,} user messages")
    return messages


def run_sentiment_analysis_batched(messages: List[Dict[str, Any]], batch_size: int = 32):
    """Run sentiment analysis using transformer model with batching."""
    from transformers import pipeline
    import torch

    print("\nLoading sentiment analysis model...")
    print("(This will download ~268MB on first run)")

    # Determine device
    if torch.backends.mps.is_available():
        device = 0  # MPS device
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = 0  # CUDA device
        print("Using CUDA GPU")
    else:
        device = -1  # CPU
        print("Using CPU")

    start_load = time.time()
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s\n")
    
    results = []
    total_batches = (len(messages) + batch_size - 1) // batch_size
    
    print(f"Processing {len(messages):,} messages in {total_batches} batches...")
    start_time = time.time()
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batch_texts = [msg['text'][:512] for msg in batch]  # Truncate to model max length
        
        # Run inference
        batch_results = classifier(batch_texts)
        
        # Combine results with metadata
        for msg, sentiment in zip(batch, batch_results):
            results.append({
                **msg,  # Include all metadata
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score'],
            })
        
        # Progress update
        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(messages):
            elapsed = time.time() - start_time
            progress = (i + batch_size) / len(messages)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Batch {i//batch_size + 1}/{total_batches} "
                  f"({progress*100:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({total_time/len(messages)*1000:.1f}ms per message)")
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save sentiment analysis results to JSON."""
    output_path = Path(output_file)
    
    # Save full results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_messages': len(results),
                'model': 'distilbert-base-uncased-finetuned-sst-2-english',
            },
            'results': results,
        }, f, indent=2)
    
    print(f"\nFull results saved to: {output_path}")
    
    # Generate summary statistics
    positive = sum(1 for r in results if r['sentiment_label'] == 'POSITIVE')
    negative = sum(1 for r in results if r['sentiment_label'] == 'NEGATIVE')
    
    # High confidence negative (likely frustrated)
    frustrated = [r for r in results if r['sentiment_label'] == 'NEGATIVE' and r['sentiment_score'] > 0.95]
    
    # Group by workspace
    by_workspace = {}
    for r in results:
        ws = Path(r['workspace_path']).name if r['workspace_path'] else 'Unknown'
        if ws not in by_workspace:
            by_workspace[ws] = {'positive': 0, 'negative': 0, 'total': 0}
        by_workspace[ws]['total'] += 1
        if r['sentiment_label'] == 'POSITIVE':
            by_workspace[ws]['positive'] += 1
        else:
            by_workspace[ws]['negative'] += 1
    
    # Save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Sentiment Analysis Summary\n\n")
        f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total Messages**: {len(results):,}\n")
        f.write(f"- **Positive**: {positive:,} ({positive/len(results)*100:.1f}%)\n")
        f.write(f"- **Negative**: {negative:,} ({negative/len(results)*100:.1f}%)\n")
        f.write(f"- **High-Confidence Negative** (score > 0.95): {len(frustrated):,} ({len(frustrated)/len(results)*100:.1f}%)\n\n")
        
        f.write("---\n\n")
        f.write("## Sentiment by Workspace\n\n")
        f.write("| Workspace | Total | Positive | Negative | Negative % |\n")
        f.write("|-----------|-------|----------|----------|------------|\n")
        
        sorted_ws = sorted(by_workspace.items(), key=lambda x: x[1]['negative'], reverse=True)
        for ws, stats in sorted_ws:
            neg_pct = stats['negative'] / stats['total'] * 100 if stats['total'] > 0 else 0
            f.write(f"| {ws} | {stats['total']} | {stats['positive']} | {stats['negative']} | {neg_pct:.1f}% |\n")
        
        f.write("\n---\n\n")
        f.write("## Most Frustrated Messages (Top 20)\n\n")
        f.write("High-confidence negative sentiment (score > 0.95):\n\n")
        
        sorted_frustrated = sorted(frustrated, key=lambda x: x['sentiment_score'], reverse=True)[:20]
        for i, msg in enumerate(sorted_frustrated, 1):
            ws = Path(msg['workspace_path']).name if msg['workspace_path'] else 'Unknown'
            f.write(f"{i}. **{ws}** (confidence: {msg['sentiment_score']:.3f})\n")
            f.write(f"   ```\n   {msg['text'][:200]}...\n   ```\n")
            f.write(f"   *Source: {msg['source_file']}, Conv: {msg['conversation_id']}, Exchange: {msg['exchange_index']}*\n\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../augment_conversations_export_leveldb"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sentiment_results.json"
    
    # Load messages
    messages = load_user_messages(data_dir)
    
    if not messages:
        print("No messages found!")
        return
    
    # Run sentiment analysis
    results = run_sentiment_analysis_batched(messages, batch_size=32)
    
    # Save results
    save_results(results, output_file)
    
    print("\n✓ Sentiment analysis complete!")


if __name__ == "__main__":
    main()

