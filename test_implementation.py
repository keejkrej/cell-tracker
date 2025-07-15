import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

from src.cell_tracker.segmentation import CellposeSegmenter
from src.cell_tracker.adjacency import AdjacencyGraphBuilder


def test_with_example_data():
    """Test the implementation with provided example data."""
    
    # Test 1: Load and visualize existing segmentation
    print("Test 1: Loading existing segmentation data...")
    seg_data = np.load('examples/MDCK_example-1_seg.npy', allow_pickle=True).item()
    mask = seg_data['masks']
    
    # Build adjacency graph
    print("Building adjacency graph from existing segmentation...")
    graph_builder = AdjacencyGraphBuilder(method='boundary_length')
    graph = graph_builder.build_graph(mask)
    
    print(f"Number of cells: {graph.number_of_nodes()}")
    print(f"Number of adjacencies: {graph.number_of_edges()}")
    print("\nEdge weights (boundary lengths):")
    for u, v, data in graph.edges(data=True):
        print(f"  Cell {u} - Cell {v}: {data['weight']} pixels")
    
    # Visualize
    fig = graph_builder.visualize_graph(graph, mask)
    plt.savefig('test_existing_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 2: Process single frame with Cellpose
    print("\n\nTest 2: Segmenting single frame with Cellpose...")
    
    # Load single frame image
    img = tifffile.imread('examples/MDCK_example-1.tif')
    print(f"Image shape: {img.shape}")
    
    # Initialize segmenter
    segmenter = CellposeSegmenter(
        model_type='cyto2',
        gpu=False,  # Set to False for compatibility
        diameter=None,  # Let Cellpose estimate
        flow_threshold=0.4,
        cellprob_threshold=0.0
    )
    
    # Segment the image
    # Assuming channels are: 0=nuclear, 1=phase contrast, 2=membrane
    result = segmenter.segment_image(img, channels=[2, 0])  # membrane for cyto, nuclear
    
    # Build graph from new segmentation
    new_graph = graph_builder.build_graph(result['masks'])
    
    print(f"\nCellpose segmentation results:")
    print(f"Number of cells detected: {new_graph.number_of_nodes()}")
    print(f"Estimated diameter: {result['diams']}")
    
    # Visualize new segmentation
    fig2 = graph_builder.visualize_graph(new_graph, result['masks'])
    plt.savefig('test_cellpose_segmentation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 3: Process timelapse (if available)
    timelapse_path = Path('examples/MDCK_example.tif')
    if timelapse_path.exists():
        print("\n\nTest 3: Processing timelapse data...")
        
        # Process first 3 frames as a test
        results = segmenter.segment_timelapse(
            str(timelapse_path),
            channels=[2, 0],
            frames=[0, 1, 2]
        )
        
        print(f"Processed {len(results)} frames")
        
        # Build graphs for each frame
        masks = [r['masks'] for r in results]
        graphs = graph_builder.process_timelapse(masks)
        
        # Example: Track a hypothetical four-cell cluster
        # Note: In real usage, you'd need to identify actual cell labels
        if len(graphs) > 0 and graphs[0].number_of_nodes() >= 4:
            # Get first 4 cells as an example
            cells = list(graphs[0].nodes())[:4]
            print(f"\nTracking four-cell cluster: {cells}")
            
            t1_results = graph_builder.track_t1_transitions(graphs, cells)
            
            for result in t1_results:
                if 'error' not in result:
                    print(f"\nTimepoint {result['timepoint']}:")
                    print(f"  A-C contact: {result['ac_contact']}")
                    print(f"  B-D contact: {result['bd_contact']}")
                    print(f"  Is T1 configuration: {result['is_t1_configuration']}")
    else:
        print(f"\nTimelapse file not found at {timelapse_path}")
    
    print("\n\nAll tests completed successfully!")


if __name__ == "__main__":
    test_with_example_data()