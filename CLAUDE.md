# Cell Tracker Project - Essential Information

## Project Overview
This project segments four-cell clusters from microscopy images and generates weighted adjacency graphs to monitor T1 transitions. T1 transitions are topological changes where the length of the contacting edge between cells a and c changes (while cells b and d are not contacting).

## Key Tasks
1. **Segment timelapse microscopy data** using Cellpose Python API
2. **Generate weighted adjacency graphs** from segmentation masks
3. **Track T1 transitions** by monitoring edge lengths between cells

## Input Data Format
- **Timelapse file**: `MDCK_example.tif` with 3 pseudo-color channels:
  1. Nuclear fluorescence
  2. Whole cell phase contrast
  3. Cell membrane fluorescence
- **Example segmentation**: `MDCK_example-1_seg.npy` (generated using Cellpose SAM GUI v4)

## Technical Requirements
- Use Cellpose Python API for segmentation (https://cellpose.readthedocs.io/en/latest/api.html#cellpose.models.CellposeModel)
- Process multi-frame timelapse data
- Generate adjacency graphs similar to `examples/main.py` implementation
- Calculate edge weights based on actual boundary lengths between cells

## Implementation Approach
1. Load multi-channel timelapse TIFF data
2. Apply Cellpose segmentation to each frame
3. Construct Region Adjacency Graphs (RAG) from segmentation masks
4. Calculate edge weights as boundary lengths between adjacent cells
5. Track graph topology changes to identify T1 transitions

## Key Code References
- Example implementation: `examples/main.py`
- Graph generation utilities: `src/cell_tracker/core/graph.py`
- Example data: `examples/MDCK_example-1.tif` and `examples/MDCK_example-1_seg.npy`

## Dependencies
- cellpose (for segmentation)
- numpy, scikit-image, networkx (for graph analysis)
- tifffile (for reading microscopy data)
- matplotlib (for visualization)