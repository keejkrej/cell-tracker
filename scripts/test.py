#!/usr/bin/env python3
"""
Test script for cell segmentation and adjacency graph analysis.

This script processes a timelapse TIFF file, performs cell segmentation using Cellpose,
and generates region adjacency graphs with boundary length calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

from src.cell_tracker.segmentation import CellposeSegmenter
from src.cell_tracker.adjacency import AdjacencyGraphBuilder

# CONSTANTS
TIMELAPSE_PATH = "examples/MDCK_example.tif"  # Path to timelapse TIFF file
SINGLE_FRAME_PATH = "examples/MDCK_example-1.tif"  # Single frame for comparison
FRAME_NUMBER = 0  # Frame number to process in detail
OUTPUT_DIR = "scripts/output"  # Output directory for results
USE_GPU = False  # Set to True if GPU is available
CELL_DIAMETER = None  # None for auto-detection, or specify in pixels


def visualize_single_frame(frame_data, segmentation_result, graph, frame_idx=0):
    """
    Create a visualization similar to examples/MDCK_example-1_result.png.
    
    Shows original channels, segmentation masks, flows, and adjacency graph.
    """
    fig = plt.figure(figsize=(18, 12))
    
    # Determine channel arrangement
    if frame_data.ndim == 3:
        if frame_data.shape[0] <= 3:
            # Channels first
            nuclei = frame_data[0]
            phase = frame_data[1]
            cytoplasm = frame_data[2]
        else:
            # Channels last
            nuclei = frame_data[:, :, 0]
            phase = frame_data[:, :, 1]
            cytoplasm = frame_data[:, :, 2]
    else:
        # Single channel
        nuclei = phase = cytoplasm = frame_data
    
    # Top row: Original channels
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(nuclei, cmap='gray')
    ax1.set_title('Nuclei Fluorescence')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(phase, cmap='gray')
    ax2.set_title('Phase Contrast')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(cytoplasm, cmap='gray')
    ax3.set_title('Cytoplasm Fluorescence')
    ax3.axis('off')
    
    # Bottom row: Analysis results
    ax4 = plt.subplot(2, 3, 4)
    masks = segmentation_result['masks']
    
    # Create colored masks with region IDs
    from matplotlib import colors
    unique_labels = np.unique(masks)
    n_labels = len(unique_labels)
    cmap = colors.ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(n_labels, 20))))
    
    im = ax4.imshow(masks, cmap=cmap, interpolation='nearest')
    ax4.set_title('Masks with Region IDs')
    
    # Add region labels
    for label in unique_labels:
        if label == 0:
            continue
        y, x = np.where(masks == label)
        if len(x) > 0:
            cy, cx = np.mean(y), np.mean(x)
            ax4.text(cx, cy, str(label), color='white', fontsize=10, 
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    ax4.axis('off')
    
    # Show flows
    ax5 = plt.subplot(2, 3, 5)
    flows = segmentation_result['flows'][0]  # Get the first flow field
    # Create RGB image from flows
    flow_rgb = np.zeros((flows.shape[1], flows.shape[2], 3))
    flow_rgb[:, :, 0] = (flows[0] + 1) / 2  # Normalize to 0-1
    flow_rgb[:, :, 1] = (flows[1] + 1) / 2
    ax5.imshow(flow_rgb)
    ax5.set_title('Flows')
    ax5.axis('off')
    
    # Show RAG network
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(masks, cmap=cmap, interpolation='nearest', alpha=0.3)
    
    # Calculate node positions as centroids
    pos = {}
    for label in graph.nodes():
        y, x = np.where(masks == label)
        if len(x) > 0:
            pos[label] = (np.mean(x), np.mean(y))
    
    # Draw graph
    import networkx as nx
    nx.draw_networkx_nodes(graph, pos, ax=ax6, node_size=500, node_color='white', 
                          edgecolors='black', linewidths=2)
    
    # Draw edges with weights
    edges = graph.edges()
    weights = [graph[u][v]['weight'] for u, v in edges]
    
    nx.draw_networkx_edges(graph, pos, ax=ax6, width=2, edge_color='black')
    nx.draw_networkx_labels(graph, pos, ax=ax6, font_size=10, font_weight='bold')
    
    # Add edge weights as labels
    edge_labels = {(u, v): f"{int(graph[u][v]['weight'])}" for u, v in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax6, font_size=8)
    
    ax6.set_title('RAG Network')
    ax6.axis('off')
    ax6.set_xlim(0, masks.shape[1])
    ax6.set_ylim(masks.shape[0], 0)
    
    plt.suptitle(f'Cell Segmentation and Adjacency Analysis - Frame {frame_idx}', fontsize=16)
    plt.tight_layout()
    
    return fig


def main():
    """Main test function."""
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Cell Tracker Test Script")
    print("=" * 60)
    
    # Initialize components
    print("\nInitializing segmenter and graph builder...")
    segmenter = CellposeSegmenter(
        model_type='cyto2',
        gpu=USE_GPU,
        diameter=CELL_DIAMETER,
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    
    graph_builder = AdjacencyGraphBuilder(method='boundary_length')
    
    # Test 1: Process single frame for comparison
    if Path(SINGLE_FRAME_PATH).exists():
        print(f"\nTest 1: Processing single frame {SINGLE_FRAME_PATH}")
        
        # Load image
        single_frame = tifffile.imread(SINGLE_FRAME_PATH)
        print(f"  Image shape: {single_frame.shape}")
        
        # Segment
        print("  Performing segmentation...")
        seg_result = segmenter.segment_image(single_frame, channels=[2, 0])
        
        # Build graph
        print("  Building adjacency graph...")
        graph = graph_builder.build_graph(seg_result['masks'])
        
        print(f"  Detected {graph.number_of_nodes()} cells")
        print(f"  Found {graph.number_of_edges()} adjacencies")
        
        # Save segmentation
        seg_path = os.path.join(OUTPUT_DIR, "single_frame_seg.npy")
        segmenter.save_segmentation(seg_result, seg_path)
        print(f"  Saved segmentation to {seg_path}")
        
        # Create visualization
        fig = visualize_single_frame(single_frame, seg_result, graph, frame_idx="single")
        output_path = os.path.join(OUTPUT_DIR, "single_frame_result.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to {output_path}")
    
    # Test 2: Process specific frame from timelapse
    if Path(TIMELAPSE_PATH).exists():
        print(f"\n\nTest 2: Processing frame {FRAME_NUMBER} from {TIMELAPSE_PATH}")
        
        # Load timelapse
        with tifffile.TiffFile(TIMELAPSE_PATH) as tif:
            timelapse_data = tif.asarray()
        
        print(f"  Timelapse shape: {timelapse_data.shape}")
        
        # Get specific frame
        if timelapse_data.ndim == 4:
            frame_data = timelapse_data[FRAME_NUMBER]
        elif timelapse_data.ndim == 3:
            frame_data = timelapse_data
            FRAME_NUMBER = 0
        else:
            raise ValueError(f"Unexpected timelapse shape: {timelapse_data.shape}")
        
        # Ensure correct channel arrangement
        if frame_data.shape[0] <= 3:
            frame_data = np.transpose(frame_data, (1, 2, 0))
        
        # Segment
        print("  Performing segmentation...")
        seg_result = segmenter.segment_image(frame_data, channels=[2, 0])
        
        # Build graph
        print("  Building adjacency graph...")
        graph = graph_builder.build_graph(seg_result['masks'])
        
        print(f"  Detected {graph.number_of_nodes()} cells")
        print(f"  Found {graph.number_of_edges()} adjacencies")
        
        # Print edge information
        print("\n  Edge weights (boundary lengths):")
        for u, v, data in sorted(graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]:
            print(f"    Cell {u} - Cell {v}: {data['weight']:.1f} pixels")
        
        # Save segmentation
        seg_path = os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER:04d}_seg.npy")
        segmenter.save_segmentation(seg_result, seg_path)
        print(f"\n  Saved segmentation to {seg_path}")
        
        # Create visualization
        fig = visualize_single_frame(frame_data, seg_result, graph, frame_idx=FRAME_NUMBER)
        output_path = os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER:04d}_result.png")
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to {output_path}")
        
        # Additional analysis: Find potential four-cell clusters
        print("\n  Analyzing cell clusters...")
        nodes = list(graph.nodes())
        if len(nodes) >= 4:
            # Find a four-cell cluster (cells that form a cycle)
            for i in range(min(len(nodes)-3, 5)):  # Check first few cells
                test_cells = nodes[i:i+4]
                analysis = graph_builder.analyze_four_cell_cluster(graph, test_cells)
                
                if any([analysis['ab_contact'] > 0, analysis['bc_contact'] > 0, 
                       analysis['cd_contact'] > 0, analysis['da_contact'] > 0]):
                    print(f"\n  Four-cell cluster {test_cells}:")
                    print(f"    A-C contact: {analysis['ac_contact']} pixels")
                    print(f"    B-D contact: {analysis['bd_contact']} pixels")
                    print(f"    T1 configuration: {analysis['is_t1_configuration']}")
                    break
    
    else:
        print(f"\nError: Timelapse file not found at {TIMELAPSE_PATH}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print(f"Results saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()