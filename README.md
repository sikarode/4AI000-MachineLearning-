# Inverse Design of Magnetoactive Mechanical Metamaterials

Two-stage conditional diffusion model for inverse design of magnetoactive metamaterials.

- **Stage 1:** Stress-strain curves → Orientation Map (OM)
- **Stage 2:** OM + magnetic compression → Residual magnetic flux density (Br) field

## Data Format

`.data` files (pickle) containing:
- `baseCompression` / `MagCompression`: deformation gradient `F` and `Stress` arrays (4×T each)
- `orientationMap`: 2D array with NaN for void regions
- `Br`: flattened Br field (interleaved x/y components, 256×256)

## Setup

```bash
bash setup_env.sh
```

Or install manually:
```bash
pip install torch diffusers tqdm numpy matplotlib
```

## Usage

### Train
```bash
python train.py --data_dir ./data --epochs1 500 --epochs2 500 --batch_size 4
```

### Inference
```bash
python inference.py --model_dir ./output --input_file sample_1.data
python inference.py --model_dir ./output --input_dir ./test_data
```

### Test
```bash
python test_model.py --model_dir ./output --data_dir ./data
```

Runs synthetic inputs, ground-truth comparison, and diversity tests.


## Project Structure

```
train.py                      # Training launcher (two-stage)
inference.py                  # Load model and generate from conditions
test_model.py                 # Model testing and evaluation
inverse_design_diffusion.py   # Core: dataset, encoders, UNet builders
setup_env.sh                  # Environment setup script
```
