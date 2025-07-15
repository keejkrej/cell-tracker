"""
Example usage of the CellposeSegmenter and AdjacencyGraphBuilder classes.

This script demonstrates how to:
1. Segment cells in microscopy images using Cellpose
2. Build weighted adjacency graphs from segmentation masks
3. Track T1 transitions in four-cell clusters
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.cell_tracker.segmentation import CellposeSegmenter
from src.cell_tracker.adjacency import AdjacencyGraphBuilder


def process_single_image(image_path: str, output_dir: str = "output"):
    """Process a single microscopy image."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize segmenter
    segmenter = CellposeSegmenter(
        model_type='cyto2',
        gpu=True,  # Use GPU if available
        diameter=None,  # Auto-estimate cell diameter
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    
    # Load and segment image
    import tifffile
    image = tifffile.imread(image_path)
    
    # Segment with channels [membrane, nuclear]
    segmentation = segmenter.segment_image(image, channels=[2, 0])
    
    # Save segmentation
    segmenter.save_segmentation(
        segmentation,
        f"{output_dir}/segmentation.npy"
    )
    
    # Build adjacency graph
    graph_builder = AdjacencyGraphBuilder(method='boundary_length')
    graph = graph_builder.build_graph(segmentation['masks'])
    
    # Visualize results
    fig = graph_builder.visualize_graph(graph, segmentation['masks'])
    fig.savefig(f"{output_dir}/adjacency_graph.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return segmentation, graph


def process_timelapse(timelapse_path: str, output_dir: str = "output"):
    """Process a timelapse microscopy file and track T1 transitions."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize components
    segmenter = CellposeSegmenter(model_type='cyto2', gpu=True)
    graph_builder = AdjacencyGraphBuilder(method='boundary_length')
    
    # Segment all frames
    print("Segmenting timelapse...")
    segmentation_results = segmenter.segment_timelapse(
        timelapse_path,
        channels=[2, 0]  # [membrane, nuclear]
    )
    
    # Extract masks and build graphs
    masks = [result['masks'] for result in segmentation_results]
    graphs = graph_builder.process_timelapse(masks)
    
    print(f"Processed {len(graphs)} frames")
    
    # Save individual frame results
    for i, (seg, graph) in enumerate(zip(segmentation_results, graphs)):
        # Save segmentation
        segmenter.save_segmentation(seg, f"{output_dir}/frame_{i:04d}_seg.npy")
        
        # Save visualization
        fig = graph_builder.visualize_graph(graph, seg['masks'])
        fig.savefig(f"{output_dir}/frame_{i:04d}_graph.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    return segmentation_results, graphs


def analyze_four_cell_cluster(graphs, cell_labels=[1, 2, 3, 4]):
    """Analyze T1 transitions in a four-cell cluster across time."""
    
    graph_builder = AdjacencyGraphBuilder()
    t1_results = graph_builder.track_t1_transitions(graphs, cell_labels)
    
    # Extract time series data
    timepoints = []
    ac_contacts = []
    bd_contacts = []
    
    for result in t1_results:
        if 'error' not in result:
            timepoints.append(result['timepoint'])
            ac_contacts.append(result['ac_contact'])
            bd_contacts.append(result['bd_contact'])
    
    # Plot T1 transition dynamics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(timepoints, ac_contacts, 'o-', label='A-C contact', linewidth=2)
    ax.plot(timepoints, bd_contacts, 's-', label='B-D contact', linewidth=2)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Contact length (pixels)')
    ax.set_title('T1 Transition Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, t1_results


if __name__ == "__main__":
    # Example 1: Process single image
    print("Processing single image...")
    seg, graph = process_single_image(
        "examples/MDCK_example-1.tif",
        output_dir="output/single_frame"
    )
    
    # Example 2: Process timelapse
    print("\nProcessing timelapse...")
    seg_results, graphs = process_timelapse(
        "examples/MDCK_example.tif",
        output_dir="output/timelapse"
    )
    
    # Example 3: Analyze T1 transitions
    # Note: You need to identify the actual cell labels for your four-cell cluster
    if len(graphs) > 0:
        print("\nAnalyzing T1 transitions...")
        # This is just an example - replace with actual cell IDs
        four_cells = [1, 2, 3, 4]  
        fig, results = analyze_four_cell_cluster(graphs, four_cells)
        fig.savefig("output/t1_transitions.png", dpi=150, bbox_inches='tight')
        plt.close()