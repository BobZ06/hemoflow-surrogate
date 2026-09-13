# Neural architecture verification

**Primary owner:** Letian

This verification covers `src/hemoflow/models/nets.py` and
`src/hemoflow/models/base.py`.

## Architectures

### NodeMLP

`NodeMLP` processes one surface node at a time.

Each node receives a feature vector describing the vessel geometry and flow.
The same multilayer perceptron is applied independently to every node.

The default structure is:

`12 input features → 128 → 128 → 128 → 1 output`

The hidden layers use a linear layer, layer normalization, and GELU activation.
The final linear layer converts the hidden representation into one output for
that node.

The MLP does not directly combine neighbouring grid nodes. Its output has shape
`(batch, arc, theta)`.

### VesselUNet

`VesselUNet` processes the unwrapped vessel surface as a two-dimensional grid.
It uses convolutions so that each location can combine information from nearby
locations.

The encoder repeatedly applies convolution and downsampling. Downsampling lets
the network represent information from a larger part of the vessel.

The decoder upsamples the internal grid back to the original resolution.
Skip connections preserve details from the encoder and combine them with the
larger-scale information from the decoder.

The U-Net also returns output with shape `(batch, arc, theta)`, so both
architectures can use the same training and prediction interfaces.

## Topology-aware convolutions

The unwrapped vessel surface looks like a rectangle, but its circumferential
direction is actually circular. The first and last theta columns are physical
neighbours.

The arc direction is different. The vessel has a real inlet and outlet, so the
two ends are not connected.

`MixedPadConv2d` respects these two types of boundaries:

- circular padding along the theta direction;
- replicate padding along the arc direction.

This prevents the convolution from treating the artificial theta cut as a
physical wall or boundary. Ordinary zero padding would create an artificial
seam and could produce a stripe of prediction error at the theta boundary.

The tensor layout used by the convolution is
`(batch, channels, arc, theta)`.

## Topology-aware upsampling

The decoder must also preserve the circular theta topology when it enlarges an
internal feature grid.

`topology_aware_upsample`:

1. adds circular padding along theta;
2. adds replicate padding along arc;
3. performs bilinear interpolation;
4. crops away the temporary padding.

This allows interpolation near the theta seam to use values from the opposite
side of the circular vessel surface. It prevents the decoder from reintroducing
the seam that the topology-aware convolutions removed.

## Training verification

The MLP was previously trained with:

`.\.venv\Scripts\python.exe -m hemoflow.cli train --config configs/mlp.yaml`

The verified MLP checkpoint was `mlp-53e1ed5fd3`.

Results:

- Device: `cpu`
- Parameters: `35,585`
- Epochs: `44`
- Training time: `2002.83 seconds`
- Relative L2: `0.030965`
- Median relative error: `1.879%`
- Low-shear Dice: `0.9276`

The U-Net was trained with:

`.\.venv\Scripts\python.exe -m hemoflow.cli train --config configs/unet.yaml`

The verified U-Net checkpoint was `unet-8e11d62f09`.

Results:

- Device: `cpu`
- Parameters: `490,001`
- Epochs: `18`
- Training time: `1382.68 seconds`
- Best validation loss: `0.012484`
- Relative L2: `0.030634`
- Median relative error: `1.834%`
- Low-shear Dice: `0.9259`
- Peak median relative error: `3.811%`
- Mean absolute error: `0.05951 Pa`
- Root mean squared error: `0.11937 Pa`
- Test vessels: `77`

The U-Net validation loss was still improving at epoch 18. Therefore, its result
is measured at the configured epoch limit rather than a fully converged upper
bound.

## Architecture comparison

The U-Net has approximately 13.8 times more parameters than the MLP:

`490,001 / 35,585 ≈ 13.8`

The improvement in relative L2 is small:

- MLP: `0.030965`
- U-Net: `0.030634`

The U-Net low-shear Dice is slightly lower:

- MLP: `0.9276`
- U-Net: `0.9259`

On this dataset, the extra spatial context provided by the U-Net produces little
additional benefit. The feature set already contains non-local information such
as `throat_ratio` and `downstream_of_throat`, which allows the pointwise MLP to
use information about the stenosis and downstream region.

This result does not mean that U-Net is never useful. It means that, for this
synthetic single-vessel dataset and the configured training budget, its much
larger parameter count does not provide a meaningful improvement over the MLP.

## Verification status

- [x] Explained the `NodeMLP` architecture.
- [x] Explained the `VesselUNet` architecture.
- [x] Explained circular theta padding.
- [x] Explained replicate arc padding.
- [x] Explained the artificial seam problem.
- [x] Explained topology-aware upsampling.
- [x] Trained the MLP end to end.
- [x] Trained the U-Net end to end.
- [x] Recorded the actual training and evaluation results.
- [x] Compared parameter count and prediction metrics.
- [x] Noted that the U-Net was still improving at the epoch limit.